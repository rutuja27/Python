# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:20:08 2021

@author: 27rut
"""

import numpy as np
import csv 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def readConfigFile(filename, config):
    
    with open(filename, 'r', newline='') as f:
        
        config_reader = csv.reader(f , delimiter=',')

        keys = list(config.keys())
        print(keys)
        
        for idx, val in enumerate(config_reader):
            config[keys[idx]] = val[1:]
       
        

def readcsvFile_lat(filename, arr, arr1, inter_latency, latency_threshold):
  
    data_grab = csv.reader(filename, delimiter=',')
    count = -1    

    if not arr:
        flag_arr=0;
    else:
        flag_arr=1
        
    if not arr1:
        flag_arr1=0;
    else:
        flag_arr1=1
        
        
    for idx,row in enumerate(data_grab):
        if(not inter_latency):
            if not flag_arr:
                arr.append((np.float(row[0])))
            if not flag_arr1:
                if(len(row) == 2):
                    lat = np.float(row[1])
                    arr1.append(lat)
                    if(lat > latency_threshold):
                        count += 1  
                else:
                    lat = np.float(row[0])
                    arr1.append(lat)
                    if(lat > latency_threshold):
                        count += 1 
        else:  
            if(idx == 0):
                prev =  np.float(row[0])
            else:
                if(np.float((np.float(row[0]) - prev))/1000 > latency_threshold):
                    count += 1    
                arr1.append((np.float(row[0]) - prev)/1000)
            prev =  (np.float(row[0]))
            
    return count
    
def readcsvFile_int(filename, arr):
    data_grab = csv.reader(filename, delimiter=',')
    for idx,row in enumerate(data_grab):
        arr.append(np.int(row[0]))
        
        
## distance - distance between peaks
## height -  definition of height of a peak        
def maxPeak(latency_arr, height, distance, numFrames, framerate):
    
    peaks = find_peaks(latency_arr, height=height, distance=distance)
    latency_arr = np.array(latency_arr)

    ## indices of peaks for plugin 
    loc_of_peaks = [0] * len(latency_arr)
    for aa in peaks[0]:
        loc_of_peaks[aa] = latency_arr[aa]
    #axes.plot(loc_of_peaks, '.')
    
    ## measure number of 1 sec intervals with spikes
    peak_count=0
    num_intervals =  (numFrames//framerate)

    for i in range(0, num_intervals-1):
       count=0
       for j in range(0, (framerate)):
           if(loc_of_peaks[(i*framerate) + j] == 0):
               count += 1
           
       if(count != framerate):
           peak_count += 1      
           
    return peak_count   
 
   
         
def plot_data(arr, inter_latency, shape, color, alpha, ax_handle):
    
    sz = len(arr)
        
    if inter_latency:
        ax_handle.plot(arr[0:sz-2],'.',color=color, marker=shape, alpha=alpha, ms=8)
    else:
        ax_handle.plot(arr[0:sz-2],'.',color=color, marker=shape, alpha=alpha, ms=8)      

def main():
    
    
    ## read csv files
    Config = {
        'no_of_cam': 0,
        'cam_suffix': [], 
        'dir_len': 0,
        'dir_list': [],
        'no_of_frames': 0,
        'no_of_trials': 0,
        'framerate': 0,
        'nidaq_prefix': '',
        'f2f_prefix': '',
        'queue_prefix': '',
        'plugin_prefix': '',
        'logging_prefix': '',
        'imagegrab_prefix': ''
            
    }
                   
    filename = 'C:/Users/27rut/BIAS/scripts/python/config_files/cameragrab_singlecamera_run.csv'    
    readConfigFile(filename,Config)  
    
    fig, axes = plt.subplots(1, 1, figsize=(10,8))
    shapes= ['+', '*', 'x', 'd', '^', 'o']
    color = ['r', 'b','g', 'm', 'c', 'k']
    alpha = [0.2,0.2,0.4, 0.4, 0.6, 0.8]
    axes.set_ylim([0,30])
    axes.set_yticks(np.arange(0,30,1))
    
    cam_suffix = Config['cam_suffix']
    print(Config['no_of_cam'])
            
    
    '''for i in range(1, num_trials+1):
    
        ## get file handles for csv files
        if trials:
            imagegrab_handle1 = open(imagegrab_file1  + str(i) + '.csv', 'r+');
            imagegrab_handle2 = open(imagegrab_file2  + str(i) + '.csv', 'r+');
            plugin_handle1 =  open(plugin_file1 + str(i) + '.csv', 'r+')
            if not isJaaba:
                plugin_handle2 =  open(plugin_file2 + str(i) + '.csv', 'r+')
        else:
            imagegrab_handle1 = open(imagegrab_file1  + '.csv', 'r+');
            imagegrab_handle2 = open(imagegrab_file2  + '.csv', 'r+');
            if isPlugin:
                plugin_handle1 =  open(plugin_file1  + '.csv', 'r+')               
                if not isJaaba:
                    plugin_handle2 =  open(plugin_file2  + '.csv', 'r+')
                    
            
            if isframe2frame:
                f2f_pluginhandle1 = open(f2f_pluginfile1 + '.csv', '+r')
                if not isJaaba:
                    f2f_pluginhandle2 =  open(f2f_pluginfile2  + '.csv', 'r+')
                    f2f_pluginhandle3 =  open(f2f_pluginfile3  + '.csv', 'r+')
                    f2f_pluginhandle4 =  open(f2f_pluginfile4  + '.csv', 'r+')
                    f2f_pluginhandle5 =  open(f2f_pluginfile5 + '.csv', 'r+')
                    f2f_pluginhandle6 =  open(f2f_pluginfile6  + '.csv', 'r+')
               
    
            if isQueue:
                #image_que_handle1 = open(imagegrab_queue_file1 + '.csv', '+r') 
                #image_que_handle2 = open(imagegrab_queue_file2 + '.csv' , '+r')
                plugin_que_handle1 = open(plugin_queue_file1 + '.csv', '+r')
                
                if not isJaaba:
                    plugin_que_handle2 = open(plugin_queue_file2 + '.csv' , '+r')
    
        camgrab_time = [] ## time the camera was trigered externally 
        imagegrab_time1 = []  ##  frame frm cam1 reached the system buffers
        imagegrab_time2 = []  ##   frame from cam2 reached the system buffers
        
        plugin_time1 = []  ## time when plugin 1 processed the frame
        plugin_time2= []  ##  time when plugin 2 processed the frame
        
        peak=0;
        plugin_f2ftime1 = [] ## frame to frame latency for plugin 1 frame 
        plugin_f2ftime2 = [] ## frame to frame latency for plugin 2 frame 
        plugin_f2ftime3 = [] 
        plugin_f2ftime4 = []
        plugin_f2ftime5 = []
        plugin_f2ftime6 = []
        
        image_que_size1 = [] ## imagegrab newImageQueue size 1
        image_que_size2 = [] ## imagegrab newImagequeue size 2
        plugin_que_size1 = [] ##  pluginImageQueue size 1
        plugin_que_size2 = [] ## pluginImageQueue size 2
     
        ## read csv files 
        readcsvFile_lat(imagegrab_handle1, camgrab_time, imagegrab_time1, inter_latency, latency_threshold)
        readcsvFile_lat(imagegrab_handle2, camgrab_time, imagegrab_time2, inter_latency, latency_threshold)
        
        if isPlugin:
            readcsvFile_lat(plugin_handle1, camgrab_time,  plugin_time1, inter_latency, latency_threshold)   
            
            if not isJaaba:
                readcsvFile_lat(plugin_handle2, camgrab_time,  plugin_time2, inter_latency, latency_threshold)
                
        
        if isframe2frame:
            peak = readcsvFile_lat(f2f_pluginhandle1, [],  plugin_f2ftime1, 0, latency_threshold)
            print('Frame to frame spikes for imagger looger 1\n', peak)
        
            if not isJaaba:
                peak = readcsvFile_lat(f2f_pluginhandle2, [],  plugin_f2ftime2, 0, latency_threshold)
                print('Frame to frame spikes for imagger looger 2\n', peak)
                peak = readcsvFile_lat(f2f_pluginhandle3, [],  plugin_f2ftime3, 0, latency_threshold)
                print('Frame to frame spikes for image grab 1\n', peak)
                peak = readcsvFile_lat(f2f_pluginhandle4, [],  plugin_f2ftime4, 0, latency_threshold)
                print('Frame to frame spikes for image grab 2\n', peak)
                
                readcsvFile_lat(f2f_pluginhandle5, [],  plugin_f2ftime5, 1)
                readcsvFile_lat(f2f_pluginhandle6, [],  plugin_f2ftime6, 1)
                
        
        if isQueue:
            #readcsvFile_int(image_que_handle1,  image_que_size1)
            #readcsvFile_int(image_que_handle2,  image_que_size2)
            readcsvFile_int(plugin_que_handle1,  plugin_que_size1)
            
            if not isJaaba:
                readcsvFile_int(plugin_que_handle2, plugin_que_size2)
            
        camgrab_latency1 = []
        camgrab_latency2 = []
        camplugin_latency1 = []
        camplugin_latency2 = []
        
        
        ## Calculate the deltas for analysing different latencies 
        for idx in range(0,len(camgrab_time)):
         
            camgrab_latency1.append((imagegrab_time1[idx] - camgrab_time[idx])*0.02)
            camgrab_latency2.append((imagegrab_time2[idx] - camgrab_time[idx])*0.02)
             
            if isPlugin:
                if isJaaba:    
                
                    camplugin_latency1.append((plugin_time1[idx] - camgrab_time[idx])*0.02)
                    if(imagegrab_time1[idx] > imagegrab_time2[idx]):
                        camplugin_latency1.append((plugin_time1[idx] - imagegrab_time1[idx])*0.02)
                    else:
                        camplugin_latency1.append((plugin_time1[idx] - imagegrab_time2[idx])*0.02)
                
                else:
                    camplugin_latency1.append((plugin_time1[idx] - camgrab_time[idx])*0.02)
                    camplugin_latency2.append((plugin_time2[idx] - camgrab_time[idx])*0.02)
        
        
        ## find peaks only for image grab
        distance=10
        numPeakinterval1 = maxPeak(camgrab_latency1, latency_threshold, distance, num_frames, framerate)
        numPeakinterval2 = maxPeak(camgrab_latency2, latency_threshold, distance, num_frames, framerate)
        print('NIDAQ Latency Peaks due to cam 1', numPeakinterval1)
        print('NIDAQ Latency Peaks due to cam 2', numPeakinterval2)
        
        ## find peaks plugin 1 for NIDAQ
        if isPlugin:
            
            distance=10
            numPeakinterval1 = maxPeak(camplugin_latency1, latency_threshold, distance, num_frames, framerate)
        
            ## find peaks plugin 2 for NIDAQ
            if not isJaaba:
                numPeakInterval2 = maxPeak(camplugin_latency2, latency_threshold, distance, num_frames, framerate)
                        
            numPeakInterval = numPeakinterval1 + numPeakinterval2
            spike_rate = numPeakInterval/(num_frames/framerate)
            print(spike_rate)
              
        
        ## plot camera latency from camera trigger to System buffer 
        #plot_data(camgrab_latency1 ,inter_latency, shapes[0], color[0], 0.4, axes)
        #plot_data(camgrab_latency2 ,inter_latency, shapes[1], color[1], 0.4, axes)
            
        ## plot 
        if isPlugin:
            if isJaaba:
                plot_data(camplugin_latency1, inter_latency, shapes[1], color[1], 0.8, axes)
    
            if not isJaaba:
                plot_data(camplugin_latency1, inter_latency, shapes[1], color[1], 0.6, axes)
                plot_data(camplugin_latency2, inter_latency, shapes[0], color[0], 0.6, axes)
                
    
        if isframe2frame:
            plot_data(plugin_f2ftime1, 1, shapes[2], color[2], 0.8, axes)
            if not isJaaba:
               image_dispatch = (np.array(plugin_f2ftime5) - np.array(plugin_f2ftime3))/1000
               image_logger = (np.array(plugin_f2ftime1) - np.array(plugin_f2ftime3))/1000
               plot_data(image_dispatch, 0, shapes[0], color[0], 0.6, axes)
               plot_data(image_logger, 0, shapes[1], color[1], 0.6, axes)
               #plot_data(plugin_f2ftime2, 1, shapes[1], color[1], 0.8, axes)
               #plot_data(plugin_f2ftime3, 1, shapes[1], color[1], 0.4, axes)
               #plot_data(plugin_f2ftime4, 1, shapes[2], color[2], 0.9, axes)
               #plot_data(plugin_f2ftime5, 1, shapes[4], color[4], 0.6, axes)
               #plot_data(plugin_f2ftime6, 1, shapes[5], color[5], 0.6, axes)
    
        #if isQueue:
             #plot_data(image_que_size1, 0, shapes[2], color[2], 0.8, axes)
             #plot_data(image_que_size2, 0, shapes[3], color[3], 0.2, axes)
             #plot_data(plugin_que_size1, 0, shapes[4], color[4], 1, axes)
         
             #if not isJaaba:
             #    plot_data(plugin_que_size2, 0, shapes[5], color[5], 1, axes)
         
    #plt.plot(latency_threshold*np.ones((num_frames)))

    #['Imagegrab camera 1 to sys latency',  'Imagegrab camera 2 to sys latency', 
    #['Cameratrig-Imagegrab camera 1 to sys latency', ' Cameratrig-Imagegrab camera 2 to sys latency', 
    #  'Cameratrig-Plugin GPU Process latency', 'Plugin frame to frame GPU Process latency']
    #labels = [ 'Vidwriter function time ', 'Imagegrab function time','Logging Queue']
              
     ##['Cameratrig-Imagegrab camera 1 to sys latency', ' Cameratrig-Imagegrab camera 2 to sys latency',
             ## 'Image Queue size 1', 'Image Queue size 2','Log Queue Size 1' , 'Log Queue Size 2']
    plt.xlabel('Frames', fontsize=12)
    plt.ylabel('Milliseconds', fontsize=12)
    plt.title('Logging Queue - Cam 0 ')
    plt.legend(labels, fontsize=10, loc='upper right')
    
    #fig.savefig('C:/Users/27rut/BIAS/misc/signal_slot_day_trials/figs/vidwriterVsqueue_withimagegrab_cam0.png')'''

if __name__ == "__main__":
    main()