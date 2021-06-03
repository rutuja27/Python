# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 14:51:36 2020

@author: patilr
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

    
def readConfigFile(filename, config):
    
    with open(filename, 'r', newline='') as f:
        
        config_reader = csv.reader(f , delimiter=',')
        rows = [[col for col in row ] for row in config_reader]

        keys = list(config.keys())
        for idx,row in enumerate(rows):   
           
            for idy,col in enumerate(row):
      
                if(idy == 0):
                    
                    if row[idy] == keys[idx]:
                        continue
                    else:
                        break
                 
                if config[keys[idx]] == '':
                    continue
                elif type(config[keys[idx]]) is list:
                    config[keys[idx]].append(col)
                elif type(config[keys[idx]]) is float:
                    config[keys[idx]] =  int(float(col))
                elif type(config[keys[idx]]) is int:
                    config[keys[idx]] = int(col)
                else:
                    continue;
                   

def readcsvFile(filename, arr, max_arr, spike_threshold, inter_latency, no_of_frames):
    
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
              if(((np.float(row[1])/1000) - prev) >= spike_threshold):# and idx <= no_of_frames-1):
                  arr.append((np.float(row[1])/1000) - prev)
                  max_arr[idx] = max(max_arr[idx], (np.float(row[1])/1000) - prev)
              prev =  (np.float(row[1])/1000)  
            

def getMaxCam(max_cam, cam0, cam1):
    
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
    plt_handle.plot(bins, cum_count*norm_factor) 
    

def main():
    
    
    ## read csv files
    Config = {
        
        'dir_list' : [],
        'days' : [],
        'file_names' : [],
        'cam' : [],
        'dir_len' : 0,
        'inter_latency' : 0,
        'min_spike_threshold' : 0,
        'max_spike_threshold' : 0,
        'step_size' : 0,
        'no_of_frames' : 0,
        'no_of_trials': 0,
        'framerate' : 0 ,
        'no_of_days':0,
        'no_of_cam' : 0,
            
    }
    
    filename = 'C:/Users/27rut/BIAS/scripts/python/config_files/twocamera_singlerun.csv'    
    readConfigFile(filename,Config)    
    
    dir_list = Config['dir_list'];
    days = Config['days']
    file_names = Config['file_names']
    cam = Config['cam']
    cdf_hist_list = [];
    max_camera_hist_list = []
    dir_len = len(dir_list)
    
     ## experiment variables
    inter_latency= Config['inter_latency']
    min_spike_threshold = Config['min_spike_threshold']
    max_spike_threshold = Config['max_spike_threshold']
    step_size = Config['step_size']
    no_of_frames = Config['no_of_frames']
    no_of_trials = Config['no_of_trials']
    framerate = Config['framerate']
    no_of_days = Config['no_of_days']
    no_of_cam = Config['no_of_cam']
    norm_factor = 0.003333 #float(framerate/(no_of_frames*no_of_trials*no_of_days))
    print("Norm factor: " , norm_factor)
    
    cam0_list = []
    cam1_list = []
    #max_lat = np.zeros(no_of_frames,dtype=float)
    max_cam = 2
    max_lat = np.zeros(no_of_frames, dtype=float)
    max_cam0 = np.zeros(no_of_frames, dtype=float)# gives information about the position  
    max_cam1 = np.zeros(no_of_frames, dtype=float)#of spike along with height of spike. 
    
    for idx, frame_dir in enumerate(dir_list):
        cam0_list = []
        cam1_list = []
        print(idx)
        max_camera_hist_list = []
        for day in days:
            
            for i in range(1,no_of_trials+1):
                if(max_cam):
                    max_cam0.fill(0);
                    max_cam1.fill(0);
                    max_lat.fill(0);
                try:
                    if(no_of_cam==2):
                        if(cam[idx] == '0'):
                            print(frame_dir + day + file_names[idx] + '0' + '_trial' + str(i)  + '.csv', 'r+')
                            cam0_handle = open(frame_dir + day + file_names[idx] + '0' + '_trial' + str(i)  + '.csv', 'r+')     
                            readcsvFile(cam0_handle, cam0_list, max_cam0, min_spike_threshold, inter_latency, no_of_frames);  
                            
                        elif(cam[idx] == '1'):  
                            print(frame_dir + day + file_names[idx] + '0' + '_trial' + str(i)  + '.csv', 'r+')
                            print(frame_dir + day + file_names[idx] + '1' + '_trial' + str(i)  + '.csv', 'r+') 
                            cam0_handle = open(frame_dir + day + file_names[idx] + '0' + '_trial' + str(i)  + '.csv', 'r+') 
                            cam1_handle = open(frame_dir + day + file_names[idx] + '1' + '_trial' + str(i)  + '.csv', 'r+')
                            readcsvFile(cam0_handle, cam0_list, max_cam0, min_spike_threshold, inter_latency, no_of_frames);  
                            readcsvFile(cam1_handle, cam1_list, max_cam1, min_spike_threshold, inter_latency, no_of_frames); 
                        if(cam[idx] == '1' and max_cam):
                            max_camera_hist_list.append(getMaxCam(max_lat, max_cam0, max_cam1))
                    else:
                        print(frame_dir + day + file_names[idx] + '0' + '_trial' + str(i)  + '.csv', 'r+')
                        if(cam[idx]=='0'):
                            cam0_handle = open(frame_dir + day + file_names[idx] + cam[0] + '_trial' + str(i)  + '.csv', 'r+')     
                            readcsvFile(cam0_handle, cam0_list, max_cam0, min_spike_threshold, inter_latency, no_of_frames);  
                        #if(len(days)==1):
                        #    print(len(cam0_list))
                        #    cdf_hist_list.append(cam0_list)
                        #    cam0_list = []
                except IOError:
                    print (' File could not be open. Check file location')
                    #return -1;
                                           
        if(cam[idx] == '1' and max_cam):
            #cdf_hist_list.append(cam0_list)
            #cdf_hist_list.append(cam1_list)
            max_camera_hist_list = [y for x in max_camera_hist_list for y in x]
            cdf_hist_list.append(max_camera_hist_list)
            print(len(max_camera_hist_list))     
        elif(cam[idx]=='0'):
            cdf_hist_list.append(cam0_list)
            print(len(cam0_list))
          
            
    #flatten the list  and append 
    #if(no_of_cam == 2 and max_cam):
    #    max_camera_hist_list = [y for x in max_camera_hist_list for y in x]
    #    cdf_hist_list.append(max_camera_hist_list)
    
    
    ## plot hists and cdf
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.tight_layout(pad=3.0)
    ax0,ax1 = axes.flatten()
    bins = np.arange(min_spike_threshold, max_spike_threshold, step_size);

    bin_shift1 = np.array(list(reversed(range(0,len(cdf_hist_list)//2 +1))))
    bin_shift1 *= -1;
    bin_shift2 = np.array(list(range(1,len(cdf_hist_list)//2+1)))
    
    if(bin_shift1.size != 0 and bin_shift2.size != 0):
        bin_shift = np.concatenate((bin_shift1,bin_shift2))
    else:
        bin_shift= np.array([1])

    print(bin_shift)
  
    #binsize
    if( step_size <=  len(cdf_hist_list)):
        binsize = step_size/(len(cdf_hist_list)+1);
    else:
        binsize =1.0;
        
    print(binsize)    
        
    #get histograms and cdf    
    for i in range(0,len(cdf_hist_list)):#dir_len):
        bin_shft = bin_shift[i]*(binsize)
        plot_histogram(cdf_hist_list[i] , bins, binsize, bin_shft, norm_factor,
                          min_spike_threshold, max_spike_threshold, ax0, ax1)
        #ax0.plot(max_lat)
   
    ax0.set_xticks(bins)
    ax0.set_title('Histogram of f2f latency of spikes',fontsize=10)
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
    
    labels = ['Imagegrab two camera single run',  'Imagegrab two camera trials', 'Imagegrab two camera same channel']
    ax0.legend(labels, fontsize=7)
    
    #fig.savefig('C:/Users/27rut/BIAS/misc/signal_slot_day_trials/figs/imagegrab_twocamera_singlerunVstrials.svg')
                
                
    
if __name__ == "__main__":
    main()