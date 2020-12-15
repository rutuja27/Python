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

def plot_histogram(hist_list, bins, binsize, bin_shft,  norm_factor, 
                   min_spike_threshold, max_spike_threshold,plt_handle, plt_handle1):

    #histogram
    bin_edges = np.histogram_bin_edges(hist_list, bins=len(bins), range=(min_spike_threshold, max_spike_threshold))    
    counts, bin_edges  = np.histogram(hist_list, bins=bin_edges)
    
    #cdf
    cum_count = np.cumsum(counts, dtype=float)   
    total_count =cum_count[-1]
    cum_count = (total_count - cum_count )*norm_factor 
    
    #plot
    counts = counts*norm_factor
    plt_handle.bar(bins[0:len(bin_edges)-1]+bin_shft, counts, binsize, label = bins , log = False)
    plt_handle1.plot(bin_edges[:-1], cum_count) 


def plot_cdf(hist_list, bins, norm_factor, min_spike_threshold, max_spike_threshold, plt_handle):
    
    
    res1 = stats.cumfreq(hist_list, numbins=len(bins),
                         defaultreallimits=(min_spike_threshold, max_spike_threshold))
    total_count = res1.cumcount[-1]
    cum_count = total_count - res1.cumcount
    plt_handle.plot(bin_edges, cum_count*norm_factor) 
    

def main():
    
    
    ## read csv files
    dir_list = ['C:/Users/27rut/BIAS/misc/signal_slot_day_trials/single_camera/'];
    days = ['6_12_2020/', '7_12_2020/', '8_12_2020/']
    cdf_hist_list = [];
    dir_len = len(days)
    
     ## experiment variables
    inter_latency=1;
    min_spike_threshold = 5;
    max_spike_threshold = 30;
    step_size = 2.0;
    no_of_frames = 200000
    no_of_trials = 5
    framerate = 500
    no_of_days = len(days)
    norm_factor= float(framerate/(no_of_frames*no_of_trials*no_of_days))
    print(norm_factor)
    
    for frame_dir in dir_list:
        #data_arr=[]
        for day in days:
           data_arr=[] 
           for i in range(3,5):
                
                try:
                    file_handle = open(frame_dir + day + 'signal_slot_f2f_trial'+ str(i)  + '.csv', 'r+');
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

    bin_shift1 = np.array(list(reversed(range(0,dir_len//2 + 1))))
    bin_shift1 *= -1;
    bin_shift2 = np.array(list(range(1,dir_len//2 + 1)))
    
    if(bin_shift1.size != 0 and bin_shift2.size != 0):
        if(dir_len%2==0 ):
            bin_shift = np.concatenate((bin_shift1,bin_shift2))
        else:
            bin_shift = np.concatenate((bin_shift1,bin_shift2))
    else:
        bin_shift= np.array([1])

    print(bin_shift)
  
    #binsize
    if( step_size <=  dir_len):
        binsize = step_size/(len(days)+1);
    else:
        binsize =1.0;
        
    print(binsize)    
        
    #get histograms and cdf    
    for i in range(0,len(days)):
        bin_shft = bin_shift[i]*(binsize)
        plot_histogram(cdf_hist_list[i] , bins, binsize, bin_shft, norm_factor,
                          min_spike_threshold, max_spike_threshold, ax0, ax1)
    
    
   
    ax0.set_xticks(bins)
    ax0.set_title('Histogram of f2f latency of spikes - across days(2 trials)',fontsize=10)
    ax0.set_ylabel('spikes/secs',fontsize=8)
    ax0.set_xlabel('Latency of Spikes',fontsize=8)
    
    #cumumltive frequencies
    #for i in range(0,dir_len-1):
    #  if(i==1):  
    #    plot_cdf(cdf_hist_list[i], bins, norm_factor, min_spike_threshold, max_spike_threshold, ax1)
    

    ax1.set_xticks(bins)
    ax1.set_title('Cumulative histogram of latency of peaks',fontsize=10)
    ax1.set_ylabel('spikes/secs',fontsize=8)
    ax1.set_xlabel('Latency of Spikes',fontsize=8)
    
    labels = ['Day 1' , 'Day 2', 'Day 3']
    ax0.legend(labels, fontsize=7)
    
    #fig.savefig('C:/Users/27rut/BIAS/misc/signal_slot_day_trials/distribution_of_latencies_across_days(2 trials).png')
                
                
    
if __name__ == "__main__":
    main()