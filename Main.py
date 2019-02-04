# -*- coding: utf-8 -*-
"""
Created December 2017
@author: Clifford Donovan

Model generation and execution
This script includes:
    construction of the model architecture using Keras
    Loading of the data and labels from .npy files
    10-fold training and testing of the model on the data of choice
    "RunModelAndDropAreas" function can be given various input to alter the analysis,
    for example: RunModelAndDropAreas(num_to_drop = 3) trains and tests the model using every combination without using 3 brain regions' data
    Output is always saved to .csv files, including the attribution files

"""

from __future__ import print_function
from HelperFunctions import repeat_samples_to_chance_50
from keras import regularizers
from keras.models import Model, Input
from keras.layers import Dense, Activation, Dropout, add, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Reshape
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import csv
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
from IPython import get_ipython
import tensorflow as tf
import itertools as it
from Get_Attributions import Get_Attributions


ipython = get_ipython()
def create_model(x_train, y_train, x_test, y_test, batch_size = 100):
##### Ensure GPU is not over loaded ###
    config = tf.ConfigProto()        
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    K.set_session(tf.Session(config=config))
##### Ensure GPU is not over loaded ###

    def ResUnit(x,filters = 10, filter_size = [2,2], strides = (1,1), pool=False, pool_size = [2,2], regularization = [0.01, 0.01]):
        res = x
        if pool_size is None:
            pool = True
            pool_size = [1,1]
        out = Conv2D(filters=filters, kernel_size=[1,1], strides=(1,1), padding="same", kernel_regularizer = regularizers.l2(regularization[0]))(x)    
        out = Conv2D(filters=filters, kernel_size=filter_size, strides=strides, padding="same", kernel_regularizer = regularizers.l2(regularization[1]))(out)
        if pool:
            out = MaxPooling2D(pool_size=pool_size)(out)
            res = Conv2D(filters=filters,kernel_size=[1,1],strides=strides, padding="same")(res)
            res = MaxPooling2D(pool_size=pool_size)(res)
        out = add([res,out])
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        return out
    
        
    def ConvBlock(input, filters, kernel_size, strides, num_blocks = 1, regularization = 0.01, Max_Pool = None, Dropout_Per_Block = None, Dropout_Total = None):
        output = input
        for i in range(num_blocks):
            output = Conv2D(filters= filters, kernel_size = kernel_size, strides = strides, kernel_regularizer = regularizers.l2(regularization))(output)
            output = BatchNormalization()(output)
            output = Activation('relu')(output)
            if Dropout_Per_Block:
                output = Dropout(Dropout_Per_Block)(output)
            if Max_Pool:
                output = MaxPooling2D(Max_Pool, strides = strides)(output)
        if Dropout_Total:
            output = Dropout(Dropout_Total)(output)
        return output

    inputdata = Input((x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    print(x_train.shape)
    output = Reshape((x_train.shape[1], x_train.shape[2], x_train.shape[3]))(inputdata)  
    output = ConvBlock(output, filters = 32, kernel_size = [2,3], 
                   strides = [1,2], num_blocks = 1, Max_Pool = None, regularization =0.2)
    output = ResUnit(output, filters = 16, filter_size = [3,3], strides = [3,3], pool = True, pool_size = [1,2], regularization = [0.1, 0.16])
    output = ResUnit(output, filters = 16, filter_size = [3,3], strides = [1,1], pool = True, pool_size = [2,1], regularization = [0.09, 0.11])
    output = Flatten()(output)
    output = Dropout(0.55)(output)
    output = Dense(1, activation = 'sigmoid')(output)
    
    model = Model(inputs=inputdata,outputs=output)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=.1,
                              patience=6, min_lr=1e-8, verbose = 1)

    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = 80, validation_data=(x_test, y_test), verbose = 1, callbacks = [reduce_lr])
    score = model.evaluate(x_test, y_test)
    accuracy = score[1]
    print('Test accuracy:', accuracy)
    return model, history

#Small function to get the indices of the window of interest for the spectrogram (used for time/frequency masking)
def Get_Spectrogram_Indices(WindowOfInterest, Indices):
    OutputIdx = (np.where(Indices >= WindowOfInterest[0])[0][0], np.where(Indices <= WindowOfInterest[1])[0][-1])
    return OutputIdx

#%% Function to Average, Drop and Run the Network
'''
Area is dropped (unless num_to_drop = 0)
Network is ran and results are saved to csv

'''
def RunModelAndDropAreas(num_to_drop = 0, timewindow = (2.0, 2.5), frequencywindow = (0, 120)):
    ipython.magic("reset -f array")
    CtrlAreaNames = np.array(['BLA','DLS','DMS','OFC','PFC','VHC','VST'])
    
    #Load in the data, labels, and frequency/time indices
    Data = np.load('./Spectrograms_Data.npy', mmap_mode = 'r')
    CombinedSpecs = np.empty((round(Data.shape[0]/7), Data.shape[1], Data.shape[2], 7))
    for i in range(len(CombinedSpecs)):
        CombinedSpecs[i,:,:,:] = np.moveaxis(Data[(7*i):i*7+7,:,:], 0, 2)
    Data = CombinedSpecs
    Labels = np.load('./7AreasCombined_Labels.npy')
    batch_size = 16
    times = np.load('./CombinedSpecs_Times.npy')
    freqs = np.load('./CombinedSpecs_Freqs.npy')
    timeidx = Get_Spectrogram_Indices(timewindow, times)
    frequencyidx = Get_Spectrogram_Indices(frequencywindow, freqs)
    Data = Data[:, frequencyidx[0]:frequencyidx[1], timeidx[0]:timeidx[1], :]
    
    if num_to_drop == 0:
        #run 10 folds
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = 2)
        cvscores = []
        TrainAttributions = []
        TestAttributions = []
        for train, test in kfold.split(list(range(Data.shape[0])), Labels):
            K.clear_session()
            #Repeat samples from lacking condition so chance = 50%
            TrainData, TrainLabels, _ = repeat_samples_to_chance_50(Data[train], Labels[train])  
            TestData, TestLabels, _ = repeat_samples_to_chance_50(Data[test], Labels[test])  
            model, history = create_model(TrainData, TrainLabels,
                                          TestData, TestLabels,
                                          batch_size = batch_size)
            scores = model.evaluate(TestData, TestLabels)
    
            print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            cvscores.append(scores[1] * 100)
            filename = 'WL_' + str(num_to_drop) + 'Areas_Dropped.csv'
            TrainAttributionsthisfold = Get_Attributions(model, TrainData, penultimate_layer_idx = -10, layer_of_interest_idx = -2, colormap = 'jet', filter_indices = 0, mode = 'grad-cam')
            TestAttributionsthisfold = Get_Attributions(model, TestData, penultimate_layer_idx = -10, layer_of_interest_idx = -2, colormap = 'jet', filter_indices = 0, mode = 'grad-cam')
            TrainAttributions.append(TrainAttributionsthisfold)
            TestAttributions.append(TestAttributionsthisfold)
            with open(filename, "a", newline = '') as f:
              writer = csv.writer(f)
              writer.writerow([cvscores[-1]])
        with open(filename, "a", newline = '') as f:
            writer = csv.writer(f)
            writer.writerow([np.mean(cvscores), np.std(cvscores), '10-fold-Total'])
        TrainAttributionsFilename = 'TrainAttributions_' +str(num_to_drop) + 'Areas_Dropped.npy'
        TestAttributionsFilename = 'TestAttributions_' +str(num_to_drop) + 'Areas_Dropped.npy'
        np.save(TrainAttributionsFilename, TrainAttributions)
        np.save(TestAttributionsFilename, TestAttributions)           

    elif num_to_drop > 0: 
        combos = it.combinations(np.linspace(0,6, num = 7), num_to_drop)
        TrainAttributions = []
        TestAttributions = []
        for combo in combos:
            combo = [int(c) for c in combo]
            AreaDroppedData, AreaDroppedLabels = np.delete(Data, np.array(combo), axis = 3), Labels
            kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state = 2)
            cvscores = []
            for train, test in kfold.split(list(range(AreaDroppedData.shape[0])), AreaDroppedLabels):
                K.clear_session()
                TrainData, TrainLabels, _ = repeat_samples_to_chance_50(AreaDroppedData[train], AreaDroppedLabels[train])  
                TestData, TestLabels, _ = repeat_samples_to_chance_50(AreaDroppedData[test], AreaDroppedLabels[test])  
                
                model, history = create_model(TrainData, TrainLabels, 
                                              TestData, TestLabels,
                                              batch_size = batch_size)
                scores = model.evaluate(TestData, TestLabels)
                model.save('model_for_attributions.h5')
        
                print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
                cvscores.append(scores[1] * 100)
                filename = '' + str(num_to_drop) + 'AreasDropped.csv'
                TrainAttributionsthisfold = Get_Attributions(model, TrainData, penultimate_layer_idx = -10, layer_of_interest_idx = -2, colormap = 'jet', filter_indices = 0, mode = 'grad-cam')
                TestAttributionsthisfold = Get_Attributions(model, TestData, penultimate_layer_idx = -10, layer_of_interest_idx = -2, colormap = 'jet', filter_indices = 0, mode = 'grad-cam')
                TrainAttributions.append(TrainAttributionsthisfold)
                TestAttributions.append(TestAttributionsthisfold)
                with open(filename, "a", newline = '') as f:
                  writer = csv.writer(f)
                  writer.writerow([cvscores[-1]])
            with open(filename, "a", newline = '') as f:
                writer = csv.writer(f)
                writer.writerow([np.mean(cvscores), np.std(cvscores), CtrlAreaNames[combo]])    
            TrainAttributionsFilename = 'TrainAttributions_' +str(num_to_drop) + 'Areas_Dropped.npy'
            TestAttributionsFilename = 'TestAttributions_' +str(num_to_drop) + 'Areas_Dropped.npy'
            np.save(TrainAttributionsFilename, TrainAttributions)
            np.save(TestAttributionsFilename, TestAttributions) 

        
#%% ################################################################################
ipython.magic("reset -f array")

#RunModelAndDropAreas(num_to_drop = 0)

#For use if testing certain time windows or freq. windows:
#Timewindows = []
#for timestep in np.arange(3.0, 4.5, 0.05):
#    if timestep + 0.5 < 4.75:
#        Timewindows.append((timestep,timestep+0.5))
#
#Freqwindows = np.arange(0, 118.943, 10)
#for window in Timewindows:
#    print(window)
#    RunModelAndDropAreas(num_to_drop = 6, timewindow = window)

