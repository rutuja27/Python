# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:51:36 2020

@author: patilr
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def readcsvFile(filename, arr,spike_threshold):
    data_grab = csv.reader(filename, delimiter=',')
    for idx,row in enumerate(data_grab):
        if( (np.int64(row[1])/1000) >= spike_threshold):
            arr.append(np.int64(row[1])/1000)
        

def main():
    
    
    ## read csv files
    frame_dir = 'C:/Users/patilr/BIAS/misc/spike_plots/plot_data/';
    cdf_hist_list = [];
    
    
     ## experiment variables
    spike_threshold = 5;
    max_spike_threshold = 20;
    
    for i in range(0,1):
        try:
            file_handle = open(frame_dir + 'imagegrab_'+ '10000'  + '.csv', 'r+');
        except IOError:
            print (' File could not be open. Check file location')
            return -1;
        readcsvFile(file_handle, cdf_hist_list, spike_threshold);
      
    print(len(cdf_hist_list))
   
    
    ## plot hists and cdf
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout(pad=3.0)
    ax0,ax1 = axes.flatten()
    bins = np.arange(5, max_spike_threshold, max_spike_threshold/10)
    
    #histogram of height of spikes
    #get histograms
    counts, bins, patches = ax0.hist(cdf_hist_list, bins=bins, color='#0504aa',alpha=0.7, rwidth=0.85 ,align='left', log=True )
    ax0.set_xticks(bins[1:])
    ax0.set_title('Histogram of height of peaks',fontsize=10)
    
    #cumumltive frequencies
    res = stats.cumfreq(cdf_hist_list, numbins=len(bins))
    ax1.plot(bins, res.cumcount) ##width=res.binsize)
    ax1.set_title('Cumulative histogram of height of peaks',fontsize=10)
    
    
if __name__ == "__main__":
    main()