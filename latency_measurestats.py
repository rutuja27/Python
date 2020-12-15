# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:22:41 2020

@author: patilr
"""
import csv
import matplotlib.pyplot as plt
import numpy as np


def readcsvFile(filename, arr):
    data_grab = csv.reader(filename, delimiter=',')
    for idx,row in enumerate(data_grab):
        arr.append(np.float(row[1]))        

def get_peaks(arr):
    return arr > 5000
   
    
def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])
   
def main():
    
    ##imagegrab_file = open('C:/build-Release/Release/imagegrab_0.csv', 'r')
    imagegrab_file = open('C:/Users/27rut/BIAS/build/Release/imagegrab_f2flatency0.csv','r')
    len_of_frames = 49999
    num_of_trials = 1;
    
    latency = []   
    
    readcsvFile(imagegrab_file, latency)
    latency = np.array(latency)
    latency = np.reshape(latency,(( num_of_trials, len_of_frames)))
    
    # find peak
    peaks = get_peaks(latency)
    #print(peaks)
    
    #count peaks
    count_peaks = np.array([np.count_nonzero(pk) for pk in peaks])
    print(sum(count_peaks))
    
    #get max peaks   
    max_peaks = np.array([max(pk) for pk in latency] )
    print(max(max_peaks))
    
    #get peak indices
    peak_ind = {}
    for i, row_pk in enumerate(peaks) :
        for j,col_pk in enumerate(row_pk):
            if col_pk:
                if i not in peak_ind.keys():
                    peak_ind[i] = [j]
                else:
                    peak_ind[i].append(j)                    
                
    #peak_ind = {(i:[j] if i not in peak_ind.keys() else peak_ind[i].append[j]) if col_pk for i, row_pk in enumerate(peaks) for j,col_pk in enumerate(row_pk) }
    
    #flatten peak indices
    peak_ind_trials = [peak_ind[keys] for keys in peak_ind.keys()]                
    peak_ind_combine = [col for row in peak_ind_trials for col in row]                
    peak_ind_combine = np.array(peak_ind_combine)
    
    #get peak differences
   
    peak_diff = [np.diff(peak_ind[keys]) for keys in peak_ind.keys() ]
    peak_diff_combine = [col for row in peak_diff for col in row]
    peak_diff_combine = np.array(peak_diff_combine)

    
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout(pad=3.0)

    ax0, ax1 = axes.flatten()
    bins = np.arange(1, len_of_frames, (len_of_frames/10))
    peak_mag = np.arange(1,30000,(30000/10))
    
    
    #histogram of spacing of peaks from each other
    #get histograms
    counts1, bins1, patches1 = ax0.hist(peak_diff_combine, bins=bins, color='#0504aa',alpha=0.7, rwidth=0.85 ,align='left', log=True )
    #bin_labels(10)
    ax0.set_title('histogram of spacing of peaks from each other',fontsize=10)
    ax0.set_xticks(bins[:-1])
    ax0.set_ylim([1,1*1e4])

    
    #histogram of magnitude of peaks 
    counts2, bins2, patches2 = ax1.hist(latency[peaks], bins=peak_mag ,color='#0504aa',alpha=0.7, rwidth=0.85, align = 'left', log=True)
    ax1.set_title('histogram of magnitude of peaks',fontsize=10)
    ax1.set_xticks(peak_mag[:-1])
    ax1.set_ylim([1,1*1e4])

    fig.suptitle("Image Grab + Image Dispatch + Signal Slot Thread")
    plt.show()
    
if __name__ == "__main__":
    main()