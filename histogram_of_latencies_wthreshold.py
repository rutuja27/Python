# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 09:36:57 2020

@author: patilr
"""

import csv
import matplotlib.pyplot as plt
import numpy as np


def readcsvFile(filename, arr):
    data_grab = csv.reader(filename, delimiter=',')
    for idx,row in enumerate(data_grab):
        arr.append(np.int64(row[1]))   

def get_peaks(arr):
    return arr > 3000


def main():


    imagegrab_file = open('C:/Users/patilr/BIAS/misc/spike_plots/plot_data/imagegrab_50000.csv','r')
    imagedispatch_file = open('C:/Users/patilr/BIAS/misc/spike_plots/plot_data/imagedispatch_cam1_50000.csv','r') 
    signal_slot_file = open('C:/Users/patilr/BIAS/misc/spike_plots/plot_data/ig_id_sl_cam1_50000.csv','r') 
    len_of_frames = 50000;
    num_of_trials = 1;
    
    latency_ig = []
    latency_id = []
    latency_sl = []
    
    #read for one thread

    readcsvFile(imagegrab_file, latency_ig)
    latency_ig = np.array(latency_ig,dtype = np.int64)
    latency_ig = np.reshape(latency_ig,(( num_of_trials, len_of_frames)))
    print(latency_ig)
    
    #read for 2 threads
    readcsvFile(imagedispatch_file, latency_id)
    latency_id = np.array(latency_id,dtype = np.int64)
    latency_id = np.reshape(latency_id,(( num_of_trials, len_of_frames)))
    
    # read for 3 threads
    readcsvFile(signal_slot_file, latency_sl)
    latency_sl = np.array(latency_sl,dtype = np.int64)
    latency_sl = np.reshape(latency_sl,(( num_of_trials, len_of_frames)))
    
    #get peaks and non peaks
    peaks_ig = get_peaks(latency_ig);
    peaks_id = get_peaks(latency_id);
    peaks_sl = get_peaks(latency_sl);
        
     #count peaks
    count_peaks_ig = np.array([np.count_nonzero(pk) for pk in peaks_ig]) 
    count_nonpeaks_ig = np.array([peaks_ig.size - count_peaks_ig])
    print(count_nonpeaks_ig)
    print(count_peaks_ig)
    
    count_peaks_id = np.array([np.count_nonzero(pk) for pk in peaks_id]) 
    count_nonpeaks_id = np.array([peaks_id.size - count_peaks_id])
    print(count_nonpeaks_id)
    print(count_peaks_id)                         
    
    count_peaks_sl = np.array([np.count_nonzero(pk) for pk in peaks_sl]) 
    count_nonpeaks_sl = np.array([peaks_sl.size - count_peaks_sl])
    print(count_nonpeaks_sl)
    print(count_peaks_sl) 
    
    labels = ['above 5 ms','below 5 ms' ]
    thread1 = [count_peaks_ig[0] , count_nonpeaks_ig[0][0]] 
    thread2 = [ count_peaks_id[0], count_nonpeaks_id[0][0]]
    thread3 = [count_peaks_sl[0], count_nonpeaks_sl[0][0]]
    x = np.arange(len(thread1))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, thread1, 0.10, label='1 thread', log=True)
    rects2 = ax.bar(x, thread2, 0.10, label='2 threads', log=True)
    rects3= ax.bar(x + width/2, thread3, 0.10, label='3 threads', log=True)
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Counts')
    ax.set_title('Histogram of Latencies')  
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.savefig('C:/Users/patilr/BIAS/misc/cvml/histogram_of_latencies.svg')
    
    fig.tight_layout()

    plt.show()

    
if __name__ == "__main__":
    main()