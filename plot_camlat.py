# -*- coding: utf-8 -*-
"""
Created on Mon May 17 10:20:08 2021

@author: 27rut
"""

import numpy as np
import csv 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def readcsvFile_lat(filename, arr, arr1, inter_latency):
  
    data_grab = csv.reader(filename, delimiter=',')
    count = 0    

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
        else:  
            if(idx == 0):
              prev =  np.float(row[0])
            else:
                if(np.float((np.float(row[0]) - prev))/1000 > 5):
                    count += 1    
                arr1.append((np.float(row[0]) - prev)/1000)
            prev =  (np.float(row[0]))
    #print(count)
    
def readcsvFile_int(filename, arr):
    data_grab = csv.reader(filename, delimiter=',')
    for idx,row in enumerate(data_grab):
        arr.append(np.int(row[0]))
        
            
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
    isframe2frame = False
    isQueue = True
                       
    imagegrab_file1 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/single_plugin/queue_size/imagegrab_cam2sys0_trial'
    imagegrab_file2 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/single_plugin/queue_size/imagegrab_cam2sys1_trial'
    plugin_file1 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/single_plugin/queue_size/jaaba_process_time_cam2sys0_trial'
    
    #imagegrab_file1 = 'C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cam2sys_lat/two_camera/imagegrab_cam2sys0_trial'
    #imagegrab_file2 = 'C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cam2sys_lat/two_camera/imagegrab_cam2sys1_trial'
    #plugin_file1 = 'C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cam2sys_lat/two_camera/signal_slot_time_cam2sys0_trial'
    
    if not isJaaba:
        #plugin_file2 = 'C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cam2sys_lat/two_camera/signal_slot_time_cam2sys1_trial'
        plugin_file2 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/single_plugin/queue_size/jaaba_process_time_cam2sys1_trial'


    if isframe2frame:
        #f2f_pluginfile1 = 'C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cam2sys_lat/two_camera/signal_slot_time_f2f0_trial'
        #f2f_pluginfile2 = 'C:/Users/27rut/BIAS/misc/signal_slot_day_trials/cam2sys_lat/two_camera/signal_slot_time_f2f1_trial'
    
        f2f_pluginfile1 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/single_plugin/new_long/jaaba_process_time_camf2f0_trial'
      

    if isQueue:
        imagegrab_queue_file1 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/single_plugin/queue_size/imagegrab_queue_0_trial'
        imagegrab_queue_file2 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/single_plugin/queue_size/imagegrab_queue_1_trial'
        jaaba_queue_file1 = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera_lat/single_plugin/queue_size/jaaba_queue_size0_trial'                      

    if trials: 
        num_trials = 5
    else:
        num_trials = 1
        imagegrab_file1 += 'long3' 
        imagegrab_file2 += 'long3' 
        plugin_file1 += 'long3'
        
        if isframe2frame:
            f2f_pluginfile1 += 'long5'
            if not isJaaba:
               f2f_pluginfile2 += 'long3'
        
        if isQueue:
            imagegrab_queue_file1 += 'long3'
            imagegrab_queue_file2 += 'long3'
            jaaba_queue_file1 += 'long3'
            
        if not isJaaba:
            plugin_file2 += 'long25'
    
    fig, axes = plt.subplots(1, 1, figsize=(10,8))
    shapes= ['+', '*', 'x', 'd', '^', 'o']
    color = ['r', 'b','g', 'm', 'c', 'k']
    alpha = [0.2,0.2,0.4, 0.4, 0.6, 0.8]
    axes.set_ylim([0,30])
    axes.set_yticks(np.arange(0,30,1))
    
    count_imagelat1=0
    count_imagelat2=0
    count_pluginlat1=0
    count_pluginlat2=0
    
    for i in range(1, num_trials+1):
    
        if trials:
            imagegrab_handle1 = open(imagegrab_file1  + str(i) + '.csv', 'r+');
            imagegrab_handle2 = open(imagegrab_file2  + str(i) + '.csv', 'r+');
            plugin_handle1 =  open(plugin_file1 + str(i) + '.csv', 'r+')
            if not isJaaba:
                plugin_handle2 =  open(plugin_file2 + str(i) + '.csv', 'r+')
        else:
            imagegrab_handle1 = open(imagegrab_file1  + '.csv', 'r+');
            imagegrab_handle2 = open(imagegrab_file2  + '.csv', 'r+');
            plugin_handle1 =  open(plugin_file1  + '.csv', 'r+')
            
            if isframe2frame:
               f2f_pluginhandle1 = open(f2f_pluginfile1 + '.csv', '+r')
               
            if not isJaaba:
                plugin_handle2 =  open(plugin_file2  + '.csv', 'r+')
                if isframe2frame:
                    f2f_pluginhandle2 = open(f2f_pluginfile2 + '.csv', '+r')
                
            if isQueue:
                image_que_handle1 = open(imagegrab_queue_file1 + '.csv', '+r') 
                image_que_handle2 = open(imagegrab_queue_file2 + '.csv' , '+r')
                jaaba_que_handle1 = open(jaaba_queue_file1 + '.csv', '+r')
    
        camgrab_time = [] ## time the camera was trigered externally 
        imagegrab_time1 = []  ##  frame frm cam1 reached the system buffers
        imagegrab_time2 = []  ##   frame from cam2 reached the system buffers
        
        plugin_time1 = []  ## time when plugin 1 processed the frame
        plugin_time2= []  ##  time when plugin 2 processed the frame
        
        plugin_f2ftime1 = [] ## frame to frame latency for plugin 1 frame 
        plugin_f2ftime2 = [] ## frame to frame latency for plugin 2 frame 
        
        image_que_size1 = [] ## imagegrab newImageQueue size 1
        image_que_size2 = [] ## imagegrab newImagequeue size 2
        jaaba_que_size1 = [] ## jaaba pluginImageQueue size 1
    
        readcsvFile_lat(imagegrab_handle1, camgrab_time, imagegrab_time1, inter_latency)
        readcsvFile_lat(imagegrab_handle2, camgrab_time, imagegrab_time2, inter_latency)
        readcsvFile_lat(plugin_handle1, camgrab_time,  plugin_time1, inter_latency)
        
        
        if isframe2frame:
            readcsvFile_lat(f2f_pluginhandle1, camgrab_time,  plugin_f2ftime1, 1)
        
        if not isJaaba:
            readcsvFile_lat(plugin_handle2, camgrab_time,  plugin_time2, inter_latency)
            
            if isframe2frame:
                readcsvFile_lat(f2f_pluginhandle2, camgrab_time,  plugin_f2ftime2, 1)
        
        if isQueue:
            readcsvFile_int(image_que_handle1,  image_que_size1)
            readcsvFile_int(image_que_handle2,  image_que_size2)
            readcsvFile_int(jaaba_que_handle1,  jaaba_que_size1)
            
        camgrab_latency1 = []
        camgrab_latency2 = []
        camplugin_latency1 = []
        camplugin_latency2 = []
        
        for idx in range(0,len(camgrab_time)):
         
            camgrab_latency1.append((imagegrab_time1[idx] - camgrab_time[idx])*0.02)
            camgrab_latency2.append((imagegrab_time2[idx] - camgrab_time[idx])*0.02)
                
            if isJaaba:    
                
                #camplugin_latency1.append((plugin_time1[idx] - camgrab_time[idx])*0.02)
                if(imagegrab_time1[idx] > imagegrab_time2[idx]):
                    camplugin_latency1.append((plugin_time1[idx] - camgrab_time[idx])*0.02)
                else:
                    camplugin_latency1.append((plugin_time1[idx] - camgrab_time[idx])*0.02)
                
            else:
                camplugin_latency1.append((plugin_time1[idx] - imagegrab_time1[idx])*0.02)
                camplugin_latency2.append((plugin_time2[idx] - imagegrab_time2[idx])*0.02)
            
            if (((imagegrab_time1[idx] - camgrab_time[idx])*0.02) > 5):
                count_imagelat1 += 1
            if (((imagegrab_time2[idx] - camgrab_time[idx])*0.02) > 5):
                count_imagelat2 += 1
                
            if isJaaba:    
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
        peaks1 = find_peaks(camplugin_latency1, height = 6, distance=10)
        camplugin_latency1 = np.array(camplugin_latency1)
        
        ## find peaks plugin 2
        if not isJaaba:
            peaks2 = find_peaks(camplugin_latency2, height = 6, distance=10)
            camplugin_latency2 = np.array(camplugin_latency2)
       
        ## peaks for plugin 1
        loc_of_peaks1 = [0] * len(camplugin_latency1)
        for aa in peaks1[0]:
            loc_of_peaks1[aa] = camplugin_latency1[aa]
        #axes.plot(loc_of_peaks1, '.')
        
        ## peaks for plugin 2
        if not isJaaba:
            loc_of_peaks2 = [0] * len(camplugin_latency2)
            for aa in peaks2[0]:
               loc_of_peaks2[aa] = camplugin_latency2[aa]
            #axes.plot(loc_of_peaks2, '.')
            
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
                continue
            if not isJaaba:
                if(count2 != framerate):
                    peak_count += 1
        print(peak_count)
        print(peak_count/(num_frames/framerate))   
          
        print("Latency Spikes due to camera 1: ", count_imagelat1)
        print("Latency Spikes due to camera 2: ", count_imagelat2)
        print("Latency Spikes due to plugin1: ", count_pluginlat1)
        if not isJaaba:
            print("Latency Spikes due to plugin2: ", peaks2[0].size)
        
        plot_data(camgrab_latency1[2460:2500] ,inter_latency, shapes[0], color[0], 0.2, axes)
        plot_data(camgrab_latency2[2460:2500] ,inter_latency, shapes[1], color[1], 0.2, axes)
        if isJaaba:
            plot_data(camplugin_latency1[2460:2500], inter_latency, shapes[2], color[2], 0.8, axes)
        
        if not isJaaba:
            plot_data(camplugin_latency2, inter_latency, shapes[2], color[2], 0.2, axes)
            if isframe2frame:
                plot_data(plugin_f2ftime1, 1, shapes[3], color[3], 0.6, axes)
        
        if isframe2frame:
           plot_data(plugin_f2ftime1, 1, shapes[2], color[2], 0.6, axes)
           if not isJaaba:
               plot_data(plugin_f2ftime2, 1, shapes[3], color[3], 0.6, axes)
        
        if isQueue:
             plot_data(image_que_size1[2460:2500], 0, shapes[3], color[3], 0.8, axes)
             plot_data(image_que_size2[2460:2500], 0, shapes[4], color[4], 0.8, axes)
             plot_data(jaaba_que_size1[2460:2500], 0, shapes[5], color[5], 0.4, axes)
             
    #plt.plot(5*np.ones((num_frames)))

    #['Imagegrab camera 1 to sys latency',  'Imagegrab camera 2 to sys latency', 
    #['Cameratrig-Imagegrab camera 1 to sys latency', ' Cameratrig-Imagegrab camera 2 to sys latency', 
    #  'Cameratrig-Plugin GPU Process latency', 'Plugin frame to frame GPU Process latency']
    labels = ['Cameratrig-Imagegrab camera 1 to sys latency', ' Cameratrig-Imagegrab camera 2 to sys latency', 'Jaaba Plugin end to end latency',
              'Image queue 1','Image Queue 2','Jaaba Plugin Queue']
    plt.xlabel('Frames', fontsize=12)
    plt.ylabel('Milliseconds', fontsize=12)
    plt.title('Jaaba Queue Vs Latency')
    plt.legend(labels, fontsize=10, loc='upper right')
 
    fig.savefig('C:/Users/27rut/BIAS/misc/signal_slot_day_trials/figs/jaaba_queVslat.pdf')

if __name__ == "__main__":
    main()