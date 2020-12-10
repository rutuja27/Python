# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:51:36 2020

@author: patilr
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def readcsvFile(filename, arr,spike_threshold, inter_latency):
    
    data_grab = csv.reader(filename, delimiter=',')
    prev=0;   
    for idx,row in enumerate(data_grab):
        if(not inter_latency):
            if((np.float(row[1])/1000) >= spike_threshold):
                 arr.append(np.float(row[1])/1000)
        else: 
            if(idx == 0):
              prev =  np.float(row[1])/1000
            else:
              if(((np.float(row[1])/1000) - prev) >= spike_threshold and idx <= 199999):
                  arr.append((np.float(row[1])/1000) - prev)
              prev =   (np.float(row[1])/1000)  
            
    #print(arr)

def plot_histogram(hist_list, bins, binsize, bin_shft,  norm_factor, plt_handle):
  
    counts, bin_edges  = np.histogram(hist_list, bins=bins)
    counts = counts/norm_factor
    print(bin_shft)
    plt_handle.bar(bins[0:len(bins)-1]+bin_shft, counts, binsize, label = bins ,align='center', log = False)


def plot_cdf(hist_list, bins, norm_factor, min_spike_threshold, max_spike_threshold, plt_handle):
    
    res1 = stats.cumfreq(hist_list, numbins=len(bins),
                         defaultreallimits=(min_spike_threshold, max_spike_threshold))
    print(res1.cumcount)
    plt_handle.plot(bins, res1.cumcount/norm_factor) 
    

def main():
    
    
    ## read csv files
    dir_list = ['C:/Users/27rut/BIAS/misc/signal_slot_day_trials/single_camera/'];
    days = ['6_12_2020/', '7_12_2020/', '8_12_2020/']
    cdf_hist_list = [];
    dir_len = len(dir_list)
    
     ## experiment variables
    inter_latency=1;
    min_spike_threshold = 5;
    max_spike_threshold = 30;
    step_size = 2.0;
    no_of_frames = 200000
    no_of_trials = 5
    no_of_days = len(days)
    total_frames= no_of_frames*no_of_trials*no_of_days
    
    for frame_dir in dir_list:
        data_arr=[]
        for day in days:
            
           for i in range(1,6):
            
                try:
                    file_handle = open(frame_dir + day + 'imagegrab_f2flatency_trial'+ str(i)  + '.csv', 'r+');
                except IOError:
                    print (' File could not be open. Check file location')
                    return -1;
                readcsvFile(file_handle, data_arr, min_spike_threshold,inter_latency);
            
        cdf_hist_list.append(data_arr)    
            
    
    ## plot hists and cdf
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout(pad=3.0)
    ax0,ax1 = axes.flatten()
    bins = np.arange(min_spike_threshold, max_spike_threshold, step_size);

    bin_shift1 = np.ones(dir_len//2)
    bin_shift1 *= -1;
    bin_shift2 = np.ones(dir_len//2)
    
    if(bin_shift1.size != 0 and bin_shift2.size != 0):
        if(len(dir_list)%2==0 ):
            bin_shift = np.concatenate((bin_shift1,bin_shift2))
        else:
            bin_shift = np.concatenate((bin_shift1,0,bin_shift2))
    else:
        bin_shift= np.array([1])

    
    #binsize
    if( step_size <=  dir_len):
        binsize = step_size/dir_len;
    else:
        binsize =1.0;
        
    #get histograms    
    for i in range(0,len(dir_list)):
        print(binsize)
        bin_shft = bin_shift[i]*(binsize/2)
        print(bin_shft)
        plot_histogram(cdf_hist_list[i] , bins, binsize, bin_shft, total_frames, ax0)
    
    
    labels = ['Signal slot plugin']
    ax0.set_xticks(bins)
    ax0.set_title('Histogram of f2f latency of spikes',fontsize=10)
    ax0.set_ylabel('spikes/frames',fontsize=8)
    ax0.set_xlabel('Latency of Spikes',fontsize=8)
    
    #cumumltive frequencies
    for i in range(0,len(dir_list)):
        plot_cdf(cdf_hist_list[i], bins, total_frames, min_spike_threshold, max_spike_threshold, ax1)
    
    #print(res1.cumcount/total_frames)
    ax1.set_xticks(bins)
    ax1.set_title('Cumulative histogram of latency of peaks',fontsize=10)
    ax1.set_ylabel('spikes/frames',fontsize=8)
    ax1.set_xlabel('Latency of Spikes',fontsize=8)
    ax0.legend(labels)
    
    #fig.savefig('C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cdf_signal_slot_f2f_latencies.pdf')
    
if __name__ == "__main__":
    main()