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
        if( (np.int64(row[1])/1000) >= spike_threshold and idx < 999999):
            arr.append(np.int64(row[1])/1000)
        

def main():
    
    
    ## read csv files
    dir_list = ['C:/Users/27rut/BIAS/misc/renesuas/', 'C:/Users/27rut/BIAS/misc/04_11_flir/single_camera_trig/'];
    cdf_hist_list = [];
    
    
     ## experiment variables
    min_spike_threshold = 5;
    max_spike_threshold = 20;
    step_size = 2;
    
    for frame_dir in dir_list:
        data_arr=[]
        for i in range(1,3):
            try:
                file_handle = open(frame_dir + 'spinImage_time_trial'+ str(i)  + '.csv', 'r+');
            except IOError:
                print (' File could not be open. Check file location')
                return -1;
            readcsvFile(file_handle, data_arr, min_spike_threshold);
        cdf_hist_list.append(data_arr)
        
      
    print(len(cdf_hist_list))
   
    
    ## plot hists and cdf
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout(pad=3.0)
    ax0,ax1 = axes.flatten()
    bins = np.arange(min_spike_threshold, max_spike_threshold, step_size);
    
    #histogram of height of spikes
    #get histograms
    counts1, bin_edges1  = np.histogram(cdf_hist_list[0], bins=bins)
    counts2, bin_edges2 = np.histogram(cdf_hist_list[1], bins=bins)   
    
    
    ##bar plots
    thread1 = counts1
    thread2 = counts2    
    binsize = 0.8
    labels = ['renasus', 'flir']
    
    ax0.bar(bins[0:7]-binsize/2, thread1, binsize, color = 'blue', label = bins ,align='edge', log = False)
    ax0.bar(bins[0:7]+binsize/2, thread2, binsize, color = 'green',label = bins ,align='edge', log = False)
    
    #for i in range(len(counts1)):
    #    thread = [counts1[i] , counts2[i]]
    #    ax0.bar(bins[i], thread, 0.10, label = bins[i], log = 'True')

    ax0.set_xticks(bins)
    ax0.set_ylim([0,10])
    ax0.set_title('Histogram of latency of spikes',fontsize=10)
    ax0.set_ylabel('Counts',fontsize=8)
    ax0.set_xlabel('Latency of Spikes',fontsize=8)
    
    #cumumltive frequencies
    res1 = stats.cumfreq(cdf_hist_list[0], numbins=len(bins),
                         defaultreallimits=(5,max_spike_threshold))
    res2 = stats.cumfreq(cdf_hist_list[1], numbins=len(bins), 
                         defaultreallimits=(5,max_spike_threshold))
    ax1.plot(bins, res1.cumcount, color = 'blue') 
    ax1.plot(bins, res2.cumcount, color = 'green')
    ax1.set_xticks(bins)
    ax1.set_ylim([0,max(cdf_hist_list[0])])
    ax1.set_title('Cumulative histogram of height of peaks',fontsize=10)
    ax1.set_ylabel('Counts',fontsize=8)
    ax1.set_xlabel('Latency of Spikes',fontsize=8)
    ax0.legend(labels)
    
    fig.savefig('C:/Users/27rut/BIAS/misc/04_11_flir/single_camera_trig/cdf_of_latencies.pdf')
    
if __name__ == "__main__":
    main()