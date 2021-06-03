# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:20:08 2021

@author: 27rut
"""

import numpy as np
import csv 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def readcsvFile(filename, arr, arr1, inter_latency):
  
    data_grab = csv.reader(filename, delimiter=',')

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
                    arr1.append(np.float(row[1]))
                else:
                    arr1.append(np.float(row[0]))
            
def plot_data(arr, inter_latency, shape, color, alpha, ax_handle):
    
    sz = len(arr)
        
    if inter_latency:
        ax_handle.plot(arr[0:sz-2],'.',color=color, marker=shape, alpha=alpha, ms=8)
    else:
        ax_handle.plot(arr[0:sz-2],'.',color=color, marker=shape, alpha=alpha, ms=8)      

def main():
    
    inter_latency = 0
    trials = 0
    num_frames = 500000
    framerate = 500
    isJaaba = True
                       
    imagegrab_file1 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/imagegrab_cam2sys0_trial'
    imagegrab_file2 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/imagegrab_cam2sys1_trial'
    plugin_file1 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/jaaba_process_time_cam2sys0_trial'
    
    #imagegrab_file1 = 'C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cam2sys_lat/two_camera/imagegrab_cam2sys0_trial'
    #imagegrab_file2 = 'C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cam2sys_lat/two_camera/imagegrab_cam2sys1_trial'
    #plugin_file1 = 'C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cam2sys_lat/two_camera/signal_slot_time_cam2sys0_trial'
    
    if not isJaaba:
        plugin_file2 = 'C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cam2sys_lat/two_camera/signal_slot_time_cam2sys1_trial'

    if trials: 
        num_trials = 5
    else:
        num_trials = 1
        imagegrab_file1 += 'long' 
        imagegrab_file2 += 'long' 
        plugin_file1 += 'long'
        if not isJaaba:
            plugin_file2 += 'long1'
    
    fig, axes = plt.subplots(1, 1, figsize=(10,8))
    shapes= ['+', '*', 'x', 'd', '+', '^']
    color = ['r', 'b','g', 'm', 'c']
    alpha = [0.4, 0.4, 0.6, 0.8]
    axes.set_ylim([0,40])
    axes.set_yticks(np.arange(0,40,2))
    count_imagelat1=0
    count_imagelat2=0
    count_pluginlat1=0
    count_pluginlat2=0
    
    for i in range(1, num_trials+1):
    
        if trials:
            imagegrab_handle1 = open(imagegrab_file1  + str(i) + '.csv', 'r+');
            imagegrab_handle2 = open(imagegrab_file2  + str(i) + '.csv', 'r+');
            plugin_handle1 =  open(plugin_file1 + str(i) + '.csv', 'r+')
            plugin_handle2 =  open(plugin_file2 + str(i) + '.csv', 'r+')
        else:
            imagegrab_handle1 = open(imagegrab_file1  + '.csv', 'r+');
            imagegrab_handle2 = open(imagegrab_file2  + '.csv', 'r+');
            plugin_handle1 =  open(plugin_file1  + '.csv', 'r+')
            if not isJaaba:
                plugin_handle2 =  open(plugin_file2  + '.csv', 'r+')
    
    
        camgrab_time = [] ## time the camera was trigered externally 
        imagegrab_time1 = []  ##  frame frm cam1 reached the system buffers
        imagegrab_time2 = []  ##   frame from cam2 reached the system buffers
        plugin_time1 = []  ## time when plugin 1 processed the frame
        plugin_time2= []  ##  time when plugin 2 processed the frame 
    
        readcsvFile(imagegrab_handle1, camgrab_time, imagegrab_time1, inter_latency)
        readcsvFile(imagegrab_handle2, camgrab_time, imagegrab_time2, inter_latency)
        readcsvFile(plugin_handle1, camgrab_time,  plugin_time1, inter_latency)
        if not isJaaba:
            readcsvFile(plugin_handle2, camgrab_time,  plugin_time2, inter_latency)

        camgrab_latency1 = []
        camgrab_latency2 = []
        camplugin_latency1 = []
        camplugin_latency2 = []
        
        for idx in range(0,len(camgrab_time)):
         
            camgrab_latency1.append((imagegrab_time1[idx] - camgrab_time[idx])*0.02)
            camgrab_latency2.append((imagegrab_time2[idx] - camgrab_time[idx])*0.02)
                
            if isJaaba:    
                camplugin_latency1.append((plugin_time1[idx] - camgrab_time[idx])*0.02)
            else:
                camplugin_latency1.append((plugin_time1[idx] - camgrab_time[idx])*0.02)
                camplugin_latency2.append((plugin_time2[idx] - camgrab_time[idx])*0.02)
            
            if (((imagegrab_time1[idx] - camgrab_time[idx])*0.02) > 5):
                count_imagelat1 += 1
            if (((imagegrab_time2[idx] - camgrab_time[idx])*0.02) > 5):
                count_imagelat2 += 1
            if(imagegrab_time1[idx] > imagegrab_time2[idx]):
                if (((plugin_time1[idx] - imagegrab_time1[idx])*0.02) > 5):
                    count_pluginlat1 += 1
            else:
                if((((plugin_time1[idx] - imagegrab_time2[idx])*0.02) > 5)):
                    count_pluginlat1 += 1
            if not isJaaba:
                if(((plugin_time2[idx] - imagegrab_time2[idx])*0.02) > 5):
                    count_pluginlat2 += 1    
                if(((plugin_time1[idx] - imagegrab_time1[idx])*0.02) > 5):
                    count_pluginlat1 += 1       
          

        ## find peaks plugin 1
        peaks1 = find_peaks(camplugin_latency1, height = 5, distance=10)
        camplugin_latency1 = np.array(camplugin_latency1)
        
        ## find peaks plugin 2
        if not isJaaba:
            peaks2 = find_peaks(camplugin_latency2, height = 5, distance=10)
            camplugin_latency2 = np.array(camplugin_latency2)
       
        ## peaks for plugin 1
        loc_of_peaks1 = [0] * len(camplugin_latency1)
        for aa in peaks1[0]:
            loc_of_peaks1[aa] = camplugin_latency1[aa]
        axes.plot(loc_of_peaks1, '.')
        
        ## peaks for plugin 2
        if not isJaaba:
            loc_of_peaks2 = [0] * len(camplugin_latency2)
            for aa in peaks2[0]:
               loc_of_peaks2[aa] = camplugin_latency2[aa]
            axes.plot(loc_of_peaks2, '.')
            
        ## measure peak intervals
        peak_count=0
        num_intervals =  (num_frames//framerate)
        for i in range(0, num_intervals-1):
            count1=0
            count2=0
            for j in range(0, (framerate)):
                if(loc_of_peaks1[(i*framerate) + j] == 0):
                    count1 += 1
                if not isJaaba:
                    if(loc_of_peaks2[(i*framerate) + j] == 0):
                        count2 += 1
                
            if(count1 != framerate):
                peak_count += 1
            if not isJaaba:
                if(count2 != framerate):
                    peak_count += 1
        print(peak_count)
        print(peak_count/(num_frames/framerate))        
          
        print("Latency Spikes due to camera 1: ", count_imagelat1)
        print("Latency Spikes due to camera 2: ", count_imagelat2)
        print("Latency Spikes due to plugin1: ", count_pluginlat1)
        print("Latency Spikes due to plugin2: ", count_pluginlat2)
        
        #plot_data(camgrab_latency1 ,inter_latency, shapes[0], color[0], alpha[2], axes)
        #plot_data(camgrab_latency2 ,inter_latency, shapes[1], color[1], alpha[1], axes)
        plot_data(camplugin_latency1, inter_latency, shapes[2], color[2], 0.2, axes)
        #plot_data(camplugin_latency2, inter_latency, shapes[3], color[3], 0.2, axes)
            
    plt.plot(5*np.ones((num_frames)))

    #['Imagegrab camera 1 to sys latency',  'Imagegrab camera 2 to sys latency', 
    #['Cameratrig-Imagegrab camera 1 to sys latency', ' Cameratrig-Imagegrab camera 2 to sys latency', 
    labels = ['Spike Peaks', 'Spikes due to plugin']
             #  'Cameratrig-Plugin no GPU Process latency1', 'Cameratrig-Plugin no GPU Process latency2']
    plt.xlabel('Frames', fontsize=12)
    plt.ylabel('Milliseconds', fontsize=12)
    plt.title('Camera Trigger-Plugin with Gpu Process Latency Spike Peaks')
    plt.legend(labels, fontsize=10, loc='upper right')
 
    fig.savefig('C:/Users/27rut/BIAS/misc/signal_slot_day_trials/figs/camtrig-withgpu_twocam_process_latency_wspikespeaks.pdf')

if __name__ == "__main__":
    main()