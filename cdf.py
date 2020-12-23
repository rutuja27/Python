# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:51:36 2020

@author: patilr
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def readcsvFile(filename, arr, max_arr, spike_threshold, inter_latency):
    
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
              if(((np.float(row[1])/1000) - prev) >= spike_threshold and idx <= 99999):
                  arr.append((np.float(row[1])/1000) - prev)
                  max_arr[idx] = max(max_arr[idx], (np.float(row[1])/1000) - prev)
              prev =  (np.float(row[1])/1000)  
            

def getMaxCam(max_cam, cam0, cam1):
    
    max_spikes = [] 
    for i in range(0, len(cam0)):
        max_cam[i] = max(cam0[i], cam1[i])
            
    max_list = [val for idx,val in enumerate(max_cam) if val != 0.0]
    return max_list;


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
    plt_handle.bar(bins[0:len(bin_edges)-1]+bin_shft, counts, binsize, label = bin_edges , log = False)
    plt_handle1.plot(bin_edges[:-1], cum_count) 


def plot_cdf(hist_list, bins, norm_factor, min_spike_threshold, max_spike_threshold, plt_handle):
    
    
    res1 = stats.cumfreq(hist_list, numbins=len(bins),
                         defaultreallimits=(min_spike_threshold, max_spike_threshold))
    total_count = res1.cumcount[-1]
    cum_count = total_count - res1.cumcount
    plt_handle.plot(bin_edges, cum_count*norm_factor) 
    

def main():
    
    
    ## read csv files
    #'C:/Users/27rut/BIAS/misc/imagegrab_day_trials/two_camera/', 
    dir_list = ['C:/Users/27rut/BIAS/misc/signal_slot_day_trials/two_camera/', 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera/'];
    days = ['21_12_2020/', '22_12_2020/', '23_12_2020/']
    file_names = ['signal_slot_f2f', 'jaaba_f2f']
    cam = ['0','1']
    cdf_hist_list = [];
    max_camera_hist_list = []
    dir_len = len(dir_list)
    
     ## experiment variables
    inter_latency=1;
    min_spike_threshold = 5;
    max_spike_threshold = 30;
    step_size = 2.0;
    no_of_frames = 100000
    no_of_trials = 5
    framerate = 500
    no_of_days = len(days)
    no_of_cam = 1 #len(cam)
    norm_factor= float(framerate/(no_of_frames*no_of_trials*no_of_days))
    print("Norm factor: " , norm_factor)
    
    cam0_list = []
    cam1_list = []
    #max_lat = np.zeros(no_of_frames,dtype=float)
    max_cam = 0
    max_lat = np.zeros(no_of_frames, dtype=float)
    max_cam0 = np.zeros(no_of_frames, dtype=float)
    max_cam1 = np.zeros(no_of_frames, dtype=float)
    
    for idx, frame_dir in enumerate(dir_list):
        cam0_list = []
        cam1_list = []
       
        for day in days:
            
            for i in range(1,no_of_trials+1):
                if(max_cam):
                    max_cam0.fill(0);
                    max_cam1.fill(0);
                    max_lat.fill(0);
                try:
                    if(no_of_cam==2):
                        cam0_handle = open(frame_dir + day + file_names[idx] + cam[0] + '_trial' + str(i)  + '.csv', 'r+')
                        cam1_handle = open(frame_dir + day + file_names[idx] + cam[1] + '_trial' + str(i)  + '.csv', 'r+')
                        readcsvFile(cam0_handle, cam0_list, max_cam0, min_spike_threshold, inter_latency);  
                        readcsvFile(cam1_handle, cam1_list, max_cam1, min_spike_threshold, inter_latency); 
                        if(max_cam):
                           max_camera_hist_list.append(getMaxCam(max_lat, max_cam0, max_cam1))
                    else:
                        cam0_handle = open(frame_dir + day + file_names[idx] + cam[0] + '_trial' + str(i)  + '.csv', 'r+')     
                        readcsvFile(cam0_handle, cam0_list, max_cam0, min_spike_threshold, inter_latency);  
                        
                except IOError:
                    print (' File could not be open. Check file location')
                    return -1;
                                
        if(no_of_cam==2):
            cdf_hist_list.append(cam0_list)
            cdf_hist_list.append(cam1_list)
        else:
            cdf_hist_list.append(cam0_list)
            
    #flatten the list  and append 
    if(no_of_cam == 2 and max_cam):
        max_camera_hist_list = [y for x in max_camera_hist_list for y in x]
        cdf_hist_list.append(max_camera_hist_list)
    
    #print(cdf_hist_list)
    ## plot hists and cdf
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout(pad=3.0)
    ax0,ax1 = axes.flatten()
    bins = np.arange(min_spike_threshold, max_spike_threshold, step_size);

    bin_shift1 = np.array(list(reversed(range(0,dir_len//2 +1))))
    bin_shift1 *= -1;
    bin_shift2 = np.array(list(range(1,dir_len//2+1)))
    
    if(bin_shift1.size != 0 and bin_shift2.size != 0):
        bin_shift = np.concatenate((bin_shift1,bin_shift2))
    else:
        bin_shift= np.array([1])

    print(bin_shift)
  
    #binsize
    if( step_size <=  dir_len):
        binsize = step_size/(dir_len+1);
    else:
        binsize =1.0;
        
    print(binsize)    
        
    #get histograms and cdf    
    for i in range(0,dir_len):
        bin_shft = bin_shift[i]*(binsize)
        plot_histogram(cdf_hist_list[i] , bins, binsize, bin_shft, norm_factor,
                          min_spike_threshold, max_spike_threshold, ax0, ax1)
        #ax0.plot(max_lat)
   
    ax0.set_xticks(bins)
    ax0.set_title('Histogram of f2f latency of spikes - 2 cameras',fontsize=10)
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
    
    labels = ['Signal Slot Cam0', 'Signal Slot Cam1', 'Max']
    ax0.legend(labels, fontsize=7)
    
    #fig.savefig('C:/Users/27rut/BIAS/misc/signal_slot_day_trials/comp_imagegrabw&woplugin_twocameras.png')
                
                
    
if __name__ == "__main__":
    main()