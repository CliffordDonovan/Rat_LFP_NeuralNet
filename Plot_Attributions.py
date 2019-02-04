# -*- coding: utf-8 -*-
"""
Created December 2017

@author: Clifford Donovan

Function and script to plot the attributions obtained from model execution
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

titles = ['0 Areas Dropped - WL Attributions']
#titles = ['BLA', 'DLS','DMS','OFC','PFC','VHC','VST']
#titles = ['VST', 'VHC','PFC','OFC','DMS','DLS','BLA']
filepath = 'TestAttributions_0Areas_Dropped.npy'
#filepath = 'E:\Clifford\LFP Ubuntu Backup\Clifford - LFP Analysis\Masking Results\WL_TrainAttributions_Mask_((0, 120), (2.0, 2.25))_0Areas_Dropped.npy'
Times_Freqs = np.load('./Times_Freqs_2.0_2.5.npy')
times = Times_Freqs[0]
freqs = Times_Freqs[1]
subplotsize = (2,2)

def Plot_Attributions(filepath, times = None, freqs = None, titles = None, subplotsize = (3,3)):
    attributions = np.load(filepath)
    
    if times is not None or freqs is not None:
        x = times - 2.0
        y = freqs
    else:
        x = np.linspace(0, 0.5, attributions.shape[2])
        y = np.linspace(0, 120, attributions.shape[1])
        
    sns.set_style("dark")
    f, axes = plt.subplots(subplotsize[0], subplotsize[1], figsize=(25,25), sharex=False, sharey=False)
    dx, dy = 0.01, 0.01
    
    for i, ax in enumerate(axes.flat):
        Drawcbar = True
        if (10*i + 10) > attributions.shape[0]:
           break 
        else:
            if attributions.ndim == 5:
                z = attributions[:,:,:,:,0]
            else:
                z = attributions

            z = np.mean(z[(10*i):((10*i)+10)], axis = 0)
            z = np.mean(z, axis = 2)
            print(z.shape)
            cmap = plt.get_cmap('jet')
            levels = MaxNLocator(nbins=cmap.N).tick_values(0.4, 0.75)
            im = ax.contourf(x + dx/2., y + dy/2., z, cmap = cmap, levels = levels, vmin = 0.4, vmax = 0.75)
            ax.set_ylabel('Frequency (Hz)')
            ax.set_xlabel('Time (s)')
            for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)
            if Drawcbar:
                cb = plt.colorbar(im, ax=ax, spacing = 'proportional', ticks = np.round(np.linspace(0.4, 0.75, 10),2))
                cb.ax.tick_params(labelsize=18)
            if titles:
                ax.set_title(titles[i], fontsize = 22)
    f.tight_layout(pad = 2)
    plt.savefig('./0_Areas_Dropped_Attributions.pdf')
    
Plot_Attributions(filepath, times = times, freqs = freqs, titles = titles, subplotsize = subplotsize)
