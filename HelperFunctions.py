# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:47:00 2017

@author: Clifford Donovan

This script contains all of the functions that aid the main NN model script
"""
from scipy.io import loadmat
from scipy.signal import decimate
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice
from math import ceil
from tqdm import tqdm

def read_mat_files(filename):
    # This function is used to import the .mat files into python
    # Important to make sure filename string DOES NOT include '.mat'
    datafile = loadmat(filename)
    DictKeys = list(datafile.keys())
    for item in DictKeys:
        if not item.startswith('_'):
            filename = item
    datafile = datafile[filename]
    if datafile[0][0].size > 1:
        #For reading in the signal data
        try:
            if datafile[0][0].shape[2] == 3:
                data = np.zeros(shape = (datafile.shape[0], datafile[0][0].shape[0], datafile[0][0].shape[1], datafile[0][0].shape[2]))
        except:    
            data = np.zeros(shape = (datafile.shape[0], datafile[0][0].shape[0], datafile[0][0].shape[1]))
    elif datafile[0][0].size == 1:
        #For reading in the labels
        data = np.zeros(shape = (datafile.shape[0], datafile[0][0].size))
    for idx, item in np.ndenumerate(datafile):
        data[idx[0],...] = item
    return data

def repeat_samples_to_chance_50(Data, Labels, Groups = None):
    # Repeat samples from the lacking condition to ensure chance = 50%
    DifferenceFromChance = (max(sum(Labels == 1), sum(Labels == 0))) - (min(sum(Labels == 1), sum(Labels == 0)))
    LowerChanceIndex = np.where(Labels == min((sum(Labels == 1), 1), (sum(Labels == 0), 0))[1])[0]
    IndicesToCopy = choice(LowerChanceIndex, size = DifferenceFromChance)
    Data = np.concatenate((Data, Data[IndicesToCopy]), axis = 0)
    Labels = np.concatenate((Labels, Labels[IndicesToCopy]), axis = 0)
    if Groups is not None:
        Groups = np.concatenate((Groups, Groups[IndicesToCopy]), axis = 0)
    return Data, Labels, Groups

def DownsampleSignals(Data, Factor = 3):
    #Factor is the amount to downsample by (e.g. factor = 2 would cut the timesteps in half)
    if len(Data.shape) <= 2:
        DownsampledData = decimate(Data, Factor, zero_phase = True)
    else:
        DownsampledData = np.empty(shape= (Data.shape[0], ceil(Data.shape[1]/Factor), Data.shape[2]))
        for idx, sample in enumerate(Data):
            DownsampledData[idx] = decimate(sample, Factor, axis = 0, zero_phase = True)
    return DownsampledData

#Simple save and load functions to go from python variables to numpy (.npz) files stored on the hard drive
def save_train_test(x_train, x_test, y_train, y_test, trainchanceprob, testchanceprob, filename):
    np.savez_compressed(filename, 
                        x_train = x_train, 
                        x_test = x_test, 
                        y_train = y_train, 
                        y_test = y_test, 
                        trainchanceprob = trainchanceprob, 
                        testchanceprob = testchanceprob)
    return

def load_train_test(filename):
    filename = filename + '.npz'
    data = np.load(filename)
    x_train = data['x_train']
    x_test = data['x_test']
    y_train = data['y_train']
    y_test = data['y_test']
    trainchanceprob = data['trainchanceprob']
    testchanceprob = data['testchanceprob']
    return x_train, x_test, y_train, y_test, trainchanceprob, testchanceprob

# Full function to create the spectrograms from the input signals
def Make_Spectrogram(Data, fs = 2000, freqmax = 120, timewindow = (2.0, 2.5), num_outputs = 2):
    from scipy.signal import get_window
    NFFT = round(fs/2)
    noverlap = NFFT - 2
    if len(Data.shape) > 2:
        specs = []
        for i, signal in tqdm(enumerate(Data)):
            plt.ioff()
            S, f, t, _ = plt.specgram(signal[:,0], NFFT=NFFT, Fs=500, pad_to = NFFT, detrend=None,
                          window=get_window('blackmanharris', NFFT), noverlap= noverlap,
                          cmap='jet', sides='onesided', scale_by_freq=False, mode='psd', scale='dB')
            freqidx = np.where(f > freqmax)[0][0]
            timeidx = (np.where(t >= timewindow[0])[0][0], np.where(t <= timewindow[1])[0][-1])
            specs.append(S[:freqidx,timeidx[0]:timeidx[1]])
        specs = np.array(specs)
    elif len(Data.shape) == 2:
        plt.ioff()
        signal = Data
        S, f, t, _ = plt.specgram(signal[:,0], NFFT=NFFT, Fs=500, pad_to = NFFT, detrend=None,
                      window=get_window('blackmanharris', NFFT), noverlap= noverlap,
                      cmap='jet', sides='onesided', scale_by_freq=False, mode='psd', scale='dB')
        freqidx = np.where(f > freqmax)[0][0]
        timeidx = (np.where(t >= timewindow[0])[0][0], np.where(t <= timewindow[1])[0][-1])
        specs = S[:freqidx,timeidx[0]:timeidx[1]]
    elif len(Data.shape) == 1:
        plt.ioff()
        signal = Data
        S, f, t, _ = plt.specgram(signal, NFFT=NFFT, Fs=500, pad_to = NFFT, detrend=None,
                      window=get_window('blackmanharris', NFFT), noverlap= noverlap,
                      cmap='jet', sides='onesided', scale_by_freq=False, mode='psd', scale='dB')
        freqidx = np.where(f > freqmax)[0][0]
        timeidx = (np.where(t >= timewindow[0])[0][0], np.where(t <= timewindow[1])[0][-1])
        specs = S[:freqidx,timeidx[0]:timeidx[1]]
    
    freq_time_idx = (f[0], f[freqidx-1], t[timeidx[0]], t[timeidx[1]])
    
    Output = (specs, freq_time_idx, f, t)
    return Output[:num_outputs]

#Used to mask the spectrograms at certain time or frequency ranges
def Spectrogram_Masker(Data, FreqRange, TimeRange, FreqSweep = False):
    times = np.load('./WL_Spec_CombinedSpecs_Times.npy')
    freqs = np.load('./WL_Spec__CombinedSpecs_Freqs.npy')
    
    def Get_Spectrogram_Indices(WindowOfInterest, Indices):
        OutputIdx = (np.where(Indices >= WindowOfInterest[0])[0][0], np.where(Indices <= WindowOfInterest[1])[0][-1])
        return OutputIdx
    
    if FreqRange[1] - FreqRange[0] <=0 or FreqRange[0] < 0 or FreqRange[1] > 120:
        FreqRange = None
        print('Freq Range is None')
 
    if TimeRange[1] - TimeRange[0] <=0 or TimeRange[1] < 2.0 or TimeRange[1] > 2.5:
        TimeRange = None
        print('Time Range is None')
        
    if FreqSweep is True:
        basefreqidx = Get_Spectrogram_Indices((0,120), freqs)
        freqs = freqs[basefreqidx[0]:basefreqidx[1]+1]
        freqidx = Get_Spectrogram_Indices(FreqRange, freqs)
        for i, specstack in enumerate(Data):
            for j in range(specstack.shape[2]):
                Data[i, :freqidx[0], :, j] = np.zeros_like(Data[i, :freqidx[0], :, j])
                Data[i, freqidx[1]:, :, j] = np.zeros_like(Data[i, freqidx[1]:, :, j])
                
    else:
        if FreqRange is not None:  
            basefreqidx = Get_Spectrogram_Indices((0,120), freqs)
            freqs = freqs[basefreqidx[0]:basefreqidx[1]+1]
            freqidx = Get_Spectrogram_Indices(FreqRange, freqs)  
        if TimeRange is not None:        
            basetimeidx = Get_Spectrogram_Indices((2.0,2.5), times)
            times = times[basetimeidx[0]:basetimeidx[1]]
            timeidx = Get_Spectrogram_Indices(TimeRange, times)
        if TimeRange is not None and FreqRange is not None:
            for i, specstack in enumerate(Data):
                for j in range(specstack.shape[2]):
                    Data[i, freqidx[0]:freqidx[1]+1, timeidx[0]:timeidx[1]+1, j] = np.zeros((freqidx[1]+1-freqidx[0], timeidx[1]+1-timeidx[0]))
    return Data, TimeRange, FreqRange

#Uses the keras-vis toolbox to investigate layer activations - see keras-vis documentation for details.
def Get_Attributions(model, data, penultimate_layer_idx = -10, layer_of_interest_idx = -1, colormap = 'jet', filter_indices = 0, mode = 'grad-cam'):
    from vis.visualization import visualize_cam, visualize_saliency, visualize_activation, get_num_filters
    
    cmap = plt.cm.get_cmap(colormap)
    if mode == 'grad-cam':
        grads = visualize_cam(model, layer_of_interest_idx, filter_indices = filter_indices, seed_input=data, penultimate_layer_idx=penultimate_layer_idx, backprop_modifier='guided')   
        grads = (grads - np.min(grads)) / (np.max(grads) - np.min(grads))
        heatmap = cmap(grads)    
    
    elif mode == 'saliency':
        grads = visualize_saliency(model, layer_of_interest_idx, filter_indices = filter_indices, seed_input=data, backprop_modifier='guided')   
        grads = (grads - np.min(grads)) / (np.max(grads) - np.min(grads))
        heatmap = cmap(grads)    
    
    elif mode == 'activations':
        vis_activations = []
        filters = np.arange(0,get_num_filters(model.layers[layer_of_interest_idx]))
        for idx in filters:
            grads = visualize_activation(model, layer_of_interest_idx, filter_indices = idx)
            grads = (grads - np.min(grads)) / (np.max(grads) - np.min(grads))
            grads = cmap(grads)
            vis_activations.append(grads)            
        heatmap = vis_activations
    return heatmap

#Used to plot the history of the model - mainly for diagnostics
def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    return