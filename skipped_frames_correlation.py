# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:22:25 2022

@author: 27rut
"""

import numpy as np 
import csv
import matplotlib.pyplot as plt
import re

def return_fighandle():
    fig, ax_handle = set_plot_var('', 1)
    return ax_handle

def copy_camtrig(lat_data1, lat_data2):
    
    lat_data2.lat_camtrig = lat_data1.lat_camtrig

def setFlags(flag_prefix) :
    
    if flag_prefix != '':
        return 1
    else:
        return 0

class LatencyData:  
 
    def __init__(self, arr1, arr2, arr3, arr4, arr5, arr6):
        # initializing instance variable
        self.lat_nidaq = arr1
        self.lat_f2f = arr2
        self.lat_queue = arr3
        self.lat_camtrig = arr4
        self.lat_nidaq_filt = arr5
        self.process_time= arr6
        
class BiasConfigMode:
          
    def __init__(self, flag1, flag2,flag3,flag4):
        # initializing instance variable
        self.isCamOnly = flag1
        self.islogging = flag2
        self.isPlugin =  flag3
        self.isJaaba  =  flag4
        
class LatencyMetric:
    
    def __init__(self, flag1, flag2,flag3, flag4):
        # initializing instance variable
        self.isnidaq=flag1;
        self.isframetoframe=flag2
        self.isqueue=flag3
        self.isnidaqThres=flag4

    
def set_plot_var(queue_prefix, no_of_trials):
     
    if(no_of_trials > 1):
            
        fig, axes = plt.subplots(no_of_trials, 1, figsize=(12,10))
        ax = axes[0].get_gridspec()
       
    else:
        fig, axes = plt.subplots(1, 1, figsize=(20,10))
        ax = axes.get_gridspec()
        
    
    fig.subplots_adjust(hspace=0.5)

    # if no_of_trials/ > 1:
    #     axes[ax.nrows-1, ax.ncols-2] = plt.subplot2grid((rows, cols), (2,0), colspan = 2)
        
    if queue_prefix:
        plt.setp(axes, yticks = np.arange(0,20,1), ylim = [0,20])
    else:
        plt.setp(axes, yticks = np.arange(0,20,2), ylim=[0,20])
   
    return fig, axes

def readcsvFile_nidaq(filename, arr_lat, arr_cam, cam_id, plugin_prefix):
    
    #if cam_id == 1 and plugin_prefix == 'jaaba_plugin':
    #    return
        
    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')

    for idx, row in enumerate(data_grab):

        if cam_id == 0:
            
            arr_cam[idx] = ((np.float(row[0]))) ## will store the count corresponding to camera trigger 
            arr_lat[idx] = (((np.float(row[1])-np.float(row[0])) * 0.02)) ## latency calculation between 
                                               ## event and camera trigger, fast clock is 50khz
                                               ## hence multiplying factor is (1/50khz- period) 0.02 to calculate latency
                                                
        else:
            arr_lat[idx] = (((np.float(row[1]) - arr_cam[idx] ) * 0.02))
        
                         
    fhandle.close()

    
def readcsvFile_int(filename, arr, cam_id, plugin_prefix):
    
    #if cam_id == 1 and plugin_prefix == 'jaaba_plugin':
    #    return   
    
    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')
    
    for idx,row in enumerate(data_grab):
        arr[idx] = np.int(row[0])                    
    fhandle.close()
    
def readcsvFile_float(filename, arr, cam_id, plugin_prefix):

    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')

    for idx,row in enumerate(data_grab):
        arr[idx] = np.float(row[0])                    
    fhandle.close()
    
def readcsvFile_f2f(filename, arr, f2f_flag, cam_id, plugin_prefix):
 
    #if cam_id == 1 and plugin_prefix == 'jaaba_plugin':
    #    return   
 
    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')
    for idx,row in enumerate(data_grab):
                     
        if f2f_flag:
            if idx == 0: 
                prev = np.float(row[0])
            else:
                arr[idx] = (np.float(row[0]) - prev)/1000
                prev = np.float(row[0])
        else:
            arr[idx] = (np.float(row[0])/1000)
            
    fhandle.close()       
            
def readLatency_data(lat_dat, testconfig, lat_metric, biasmode_prefix, \
                     cam_id):
    
    no_of_trials = np.int(testconfig['no_of_trials'])
    numFrames = np.int(testconfig['numFrames'])
    plugin_prefix = testconfig['plugin_prefix']
    f2f_flag = 0
    
    path_dir = testconfig['dir_list'][0] 
    
    if(numFrames > 100000):
        trial_suffix = '_long_trial'
    else:
        trial_suffix = '_short_trial'

    
    for i in range(0, no_of_trials):     
    
        ## read latency readings from nidaq
        if lat_metric.isnidaq:
            print(biasmode_prefix)
            filename = path_dir + testconfig['nidaq_prefix'] \
            + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
            testconfig['date'] + '/' + biasmode_prefix + '_' + \
            testconfig['nidaq_prefix'] +  'cam' + testconfig['cam_suffix'][cam_id]\
            + trial_suffix + str(i+1) + '.csv'
            print(filename)
            readcsvFile_nidaq(filename, lat_dat.lat_nidaq[i], lat_dat.lat_camtrig[i], \
                          cam_id, plugin_prefix)
                
        # read latency frame to frame from pc timings
        if lat_metric.isframetoframe:
            
            if f2f_flag:

                filename = path_dir + testconfig['f2f_prefix'] \
                + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
                testconfig['date'] + '/' + biasmode_prefix + '_' + \
                testconfig['f2f_prefix'] +  'cam' + testconfig['cam_suffix'][cam_id]\
                + trial_suffix + str(i+1) + '.csv'
                
                readcsvFile_f2f(filename, lat_dat.lat_f2f[i], 1, cam_id, plugin_prefix)
            else:
                    
                filename = path_dir + testconfig['nidaq_prefix'] \
                + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
                testconfig['date'] + '/' + biasmode_prefix + '_' + \
                "process_time" +  'cam' + testconfig['cam_suffix'][cam_id]\
                + trial_suffix + str(i+1) + '.csv'
                
                readcsvFile_f2f(filename, lat_dat.lat_f2f[i], 0, cam_id, plugin_prefix)
            
        # read queue size
        if lat_metric.isqueue:
            
            filename = path_dir + testconfig['queue_prefix'] \
            + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
            testconfig['date'] + '/'+ biasmode_prefix + '_' + \
            testconfig['queue_prefix'] +  'cam' + \
            testconfig['cam_suffix'][cam_id] + trial_suffix + str(i+1) + '.csv'  
    
            readcsvFile_int(filename, lat_dat.lat_queue[i], cam_id, plugin_prefix)
            
        if lat_metric.isnidaqThres:
            filename = path_dir + testconfig['nidaq_prefix'] \
            + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
            testconfig['date'] + '/' + biasmode_prefix + '_' + 'nidaq_thres' + 'cam' + \
            testconfig['cam_suffix'][cam_id] + trial_suffix + str(i+1) + '.csv'

            #print(filename)
            readcsvFile_float(filename, lat_dat.lat_nidaq_filt[i], cam_id, plugin_prefix)
                     
                    
def readConfigFile(filename, config):
    
    with open(filename, 'r', newline='') as f:
        
        config_reader = csv.reader(f , delimiter=',')
        keys = list(config.keys()) ### keys in configdata structure
        rows = [[col for col in row ] for row in config_reader] ## key-value pair in csv file
       
        if len(rows) == len(keys):
            pass
        else:
            print(len(rows))
            print(len(keys))
            print('key-value pair unbalanced')
        
        for idx,row in enumerate(rows):   
 
            for idy,col in enumerate(row):
                   
                if idy == 0:
                    if row[idy] == keys[idx]:
                        continue
                    else:
                        print(row[idx])
                        break
                if type(config[keys[idx]]) is str:
                    if keys[idx] == 'git_commit':
                        col = col[1:]
                    config[keys[idx]] = str(col)
                elif type(config[keys[idx]]) is list:
                    col = col.strip("[, ], ' ")
                    col = re.split(',', col)
                    if col[0] == '':
                        config[keys[idx]] = []
                    else:
                        config[keys[idx]] = col
                elif type(config[keys[idx]]) is float:
                    config[keys[idx]] =  float(col)
                elif type(config[keys[idx]]) is int:
                    config[keys[idx]] = int(col)
                else:  
                    continue;
                    
def correlate_skips(lat_data1, lat_data2, lat_data3, testconfig):
    
    no_of_trials = testconfig['no_of_trials']
    numFrames = testconfig['numFrames']
    total_skips_misses=0
    
    fig, ax = plt.subplots()
    # for i in range(0, no_of_trials): 
    #     print('skip correlation score for trial no', i)
    #     total_skips_misses = np.sum(np.subtract(lat_data1.lat_queue[i],lat_data2.lat_queue[i]))
    #     print(total_skips_misses)
    for i in range(0, no_of_trials):  
        plt.plot(lat_data1.lat_nidaq[i][:], 'P',  alpha=0.3, markersize=8)
        #plt.plot(lat_data2.lat_nidaq_filt[i][:], 'x',  alpha=0.4,markersize=8)
        #plt.plot(lat_data3.lat_nidaq[i][:], 'o',  alpha=0.5,markersize=6)
        #plt.plot(lat_data1.lat_queue[i][20800:21000], '.',  alpha=0.3)
        #plt.plot(lat_data2.lat_queue[i][20800:21000], '.', alpha=0.6)  
        
    plt.show()   

def filterf2f(lat_data1, lat_data2,lat_data3, testconfig):

    no_of_trials = testconfig['no_of_trials']
    numFrames = testconfig['numFrames']
  
    f2f_filt_1 = []
    f2f_filt_2 = []
    f2f_filt_3 = []
    
    for i in range(0, no_of_trials):
        lat_data1.lat_f2f[i][lat_data1.lat_f2f[i] < 4.0] = 0.0
        lat_data2.lat_f2f[i][lat_data2.lat_f2f[i] < 4.0] = 0.0
        lat_data3.lat_f2f[i][lat_data3.lat_f2f[i] < 4.0] = 0.0
        #correlate_skips(lat_data1.lat_f2f, lat_data2.lat_f2f, testconfig)    
        
    plt.show()  

def maxPeak(latency_arr, height, distance, latency_filt_arr, testconfig):
    
    peaks = []
    numFrames = len(latency_arr)

    i = 0         
    while i < numFrames:
        bout_count=0
        if latency_arr[i] >= height:
            peaks.append(i)
            latency_filt_arr[i] = latency_arr[i]
            #print(i, np.array(latency_filt_arr[i]))
            i += 1  

            ## getting rid of the tail of spikes
            while i < numFrames and latency_arr[i] >= height:
                latency_filt_arr[i] = 0.0
                bout_count +=1 
                i+=1
                
        else :
            latency_filt_arr[i] = latency_arr[i]
            i += 1
            
        testconfig['max_skippedFrames_nidaq'] = max(int(testconfig['max_skippedFrames_nidaq']), bout_count)           
            
    # ## get filtered indexes
    # for aa in peaks:
    #     latency_filt_arr[aa] = latency_arr[aa]

    return [len(peaks), peaks]     

def plot_raw_data(arr1, arr2, arr3, shape, color, alpha, labels, ax_handle,\
                  no_of_trials, latency_threshold, numFrames, title,\
                  cam_id, legend):
         
    if cam_id == 0:
        color_id = 0
    else:
        color_id = 3 

    if(no_of_trials > 1):
        
        ax = ax_handle[0].get_gridspec()
 
        for ix in range(0, ax.nrows):
            
            if cam_id == 0:
                
                if (ix) == 0:
                    ax_handle[ix].plot(arr1[ix], '.', 
                                       color=color[color_id] ,\
                                       marker=shape[ix], \
                                       alpha=1, ms=12, label='Cam' + str(cam_id))    
                else:
                    ax_handle[ix].plot(arr1[ix], '.', 
                                       color=color[color_id],\
                                       marker=shape[ix], \
                                       alpha=1, ms=12)
            else:
                if (ix) == 0:
                    ax_handle[ix].plot(arr1[ix], '.', 
                                       color=color[color_id] ,\
                                       marker=shape[ix], \
                                       alpha=1, ms=12, label='Cam' + str(cam_id)) 
                else:
                    ax_handle[ix].plot(arr1[ix], '.', 
                                       color=color[color_id] ,\
                                       marker=shape[ix],\
                                       alpha=1, ms=12,label= '_nolegend_')
            ax_handle[ix].title.set_text('Trial ' +  str(ix + 1))
            ax_handle[ix].tick_params(axis='x', labelsize=12)
            ax_handle[ix].tick_params(axis='y', labelsize=12)
            #ax_handle[ix].plot(np.array(latency_threshold*np.ones(numFrames)),\
            #                      label='_nolegend_')
        plt.suptitle(title,fontsize=17)                
        plt.setp(ax_handle[:], xlabel = 'Frames', ylabel ='Milliseconds')
    else:
        #ax = ax_handle.get_gridspec()
        ax_handle.plot(arr1[0][:], '.', \
                                color=color[0],\
                                marker=shape[0], \
                                alpha=0.4, ms=12, label='Cam' + str(cam_id))
        ax_handle.plot(arr2[0][:], '.', \
                                color=color[1],\
                                marker=shape[1], \
                                alpha=0.6, ms=12, label='Cam' + str(cam_id))
        ax_handle.plot(arr3[0][:], '.', \
                         color=color[2],\
                         marker=shape[2], \
                         alpha=0.8, ms=12, label='Cam' + str(cam_id))
        ax_handle.plot(np.array(latency_threshold*np.ones(25)),\
                      label='_nolegend_')   
        plt.suptitle(title + str(cam_id),fontsize=17)    
        plt.setp(ax_handle, xlabel = 'Frames', ylabel ='Milliseconds')
        plt.legend(legend)
    
def plot_thres_data_singletrial(arr1, arr2, arr3, shape, color, alpha, ax_handle,\
                  no_of_trials, latency_threshold, numFrames, title,\
                  cam_id, legend):
    
    ax_handle.plot(arr1[:], '.', \
                                color=color,\
                                marker=shape[4], \
                                alpha=0.4, ms=12, label='Cam' + str(cam_id))
    ax_handle.plot(arr2[:], '.', \
                                color=color,\
                                marker=shape[1], \
                                alpha=0.6, ms=12, label='Cam' + str(cam_id))
    '''ax_handle.plot(arr3[:], '.', \
                          color=color,\
                          marker=shape[2], \
                          alpha=0.8, ms=12, label='Cam' + str(cam_id))'''
    ax_handle.plot(np.array(latency_threshold*np.ones(numFrames)),\
                      label='_nolegend_')     
    plt.suptitle(title + str(cam_id),fontsize=17)
    plt.setp(ax_handle, xlabel = 'Frames', ylabel ='Milliseconds')
    plt.legend(legend)
    plt.show()
        
def plot_matching_data(arr1, arr2, arr3, testconfig, cam_id):
    
    fig, ax_handle = set_plot_var('',1) 
    labels = ['nidaq latency spike imagegrab', 'nidaq latency spike imagedispatch', 'nidaq latency spike jaaba']
    shape= ['+', '*', 'x', '.', '^', 'o']
    color = ['r', 'b','g', 'm', 'c', 'k']
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]    
    title= 'Latencies from different threads Camera '
    legend = ['Imagegrab', 'Imagedispatch', 'Jaaba']
    
    numFrames = testconfig['numFrames']
    no_of_trials = testconfig['no_of_trials']
    latency_threshold = testconfig['latency_threshold']
    
    for i in range(4,5):
        
        plot_thres_data_singletrial(arr1.lat_f2f[i], arr2.lat_f2f[i],
                                    arr3.lat_nidaq_filt[i], shape, color[3],
                                    alpha, ax_handle, 1, 
                                    latency_threshold, numFrames,
                                    title, cam_id, legend)
        #plt.suptitle(title + str(cam_id),fontsize=17)
        #plt.setp(ax_handle, xlabel = 'Frames', ylabel ='Milliseconds')
        #plt.legend(legend)

        
def count_latencies(lat_data1, lat_data2, lat_data3, cam_id, testconfig):
    
    no_of_trials = testconfig['no_of_trials']
    latency_threshold = testconfig['latency_threshold']
     
    for i in range(0,no_of_trials):

        if testconfig['nidaq_prefix'] == 'nidaq':

            count_spikes_imagegrab = 0
            count_spikes_imagedispatch = 0
            count_spikes_jaaba = 0

            [count_peaks, loc_peaks] = maxPeak(lat_data1.lat_nidaq_filt[i], \
                                               latency_threshold, 10, lat_data1.lat_nidaq_filt[i], testconfig)

            count_spikes_imagegrab = count_peaks
            #np.count_nonzero(lat_data1.lat_nidaq[i][:] > 4)

            [count_peaks, loc_peaks] = maxPeak(lat_data2.lat_nidaq_filt[i], \
                                               latency_threshold, 10, lat_data2.lat_nidaq_filt[i], testconfig)

            count_spikes_imagedispatch = count_peaks
            #np.count_nonzero(lat_data2.lat_nidaq[i][:] > 4)

            [count_peaks, loc_peaks] = maxPeak(lat_data3.lat_nidaq[i], \
                                               latency_threshold, 10, lat_data3.lat_nidaq_filt[i], testconfig)

            count_spikes_jaaba =  count_peaks #np.count_nonzero(lat_data3.lat_nidaq[i][:] > 4)

            print('Trial i ', i)
            print('Imagegrab spikes nidaq', count_spikes_imagegrab)
            print('Imagedispatch spikes nidaq', count_spikes_imagedispatch)
            print('Jaaba spikes frame to nidaq', count_spikes_jaaba)

        if testconfig['f2f_prefix'] == 'f2f':

            count_spikes_imagegrab = 0
            count_spikes_imagedispatch = 0
            count_spikes_jaaba = 0
            count_spikes_imagegrab = np.count_nonzero(lat_data1.lat_f2f[i][:] > 4)
            count_spikes_imagedispatch = np.count_nonzero(lat_data2.lat_f2f[i][:] > 4)
            count_spikes_jaaba = np.count_nonzero(lat_data3.lat_f2f[i][:] > 4)
            print('Trial i ', i)
            print('Imagegrab spikes f2f', count_spikes_imagegrab)
            print('Imagedispatch spikes f2f', count_spikes_imagedispatch)
            print('Jaaba spikes f2f', count_spikes_jaaba)


        
def compare_lat_thres_nidaq(lat_data1, lat_data2, lat_data3, cam_id, testconfig,
                            ax_handle, color_id):
    
    no_of_trials = testconfig['no_of_trials']
    numFrames = testconfig['numFrames']
    latency_threshold = testconfig['latency_threshold']
    
    # mismatch_peaks_cam0 = []
    # mismatch_peaks_cam1 = []
    
    mismatch_peaks_imagegrab = no_of_trials*[numFrames*[0.0]]
    mismatch_peaks_imagedispatch = no_of_trials*[numFrames*[0.0]]   
    mismatch_peaks_jaaba = no_of_trials*[numFrames*[0.0]]
        
    for i in range(0,1):
        
        # plotting variables
        # fig, ax = plt.subplots()
        #fig, ax_handle = set_plot_var('',1) 
        labels = ['nidaq latency spike imagegrab', 'nidaq latency spike imagedispatch', 'nidaq latency spike jaaba']
        shape= ['+', '*', 'x', '.', '^', 'o']
        color = ['r', 'b','g', 'm', 'c', 'k']
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]    
        title= 'Latencies from different threads Camera '
        legend = ['Imagegrab', 'Imagedispatch', 'Jaaba']
    
        ## flags to see if spike in each thread for each trial 
        isImagegrab=1
        isImagedispatch=1
        isJaaba=1
        
        #count spikes in each thread for each trial
        imagegrab_spikes =0
        imagedispatch_spikes=0
        jaaba_spikes=0
        
        imagegrab_spike_idx = []
        imagedispatch_spike_idx = []
        jaaba_spike_idx = []
        
        if not np.any(lat_data1.lat_nidaq_filt[i][:]):
            print('No imagegrab')
            return
    
        if not np.any(lat_data2.lat_nidaq_filt[i][:]):
            print('No imagedispatch')
            isImagedispatch = 0
    
        if not np.any(lat_data3.lat_nidaq_filt[i][:]):
            print('No jaaba')
            isJaaba = 0
               
        if isImagegrab:
            imagegrab_spike_idx = np.argwhere(lat_data1.lat_nidaq[i][:] > 4).flatten()
        if isImagedispatch:
            imagedispatch_spike_idx = np.argwhere(lat_data2.lat_nidaq[i][:] > 4).flatten()
        if isJaaba:
            jaaba_spike_idx = np.argwhere(lat_data3.lat_nidaq[i][:] > 4).flatten()


        ## if no latency spike is present 
        if imagegrab_spike_idx.size == 0:
            if imagedispatch_spike_idx.size==0:
                if jaaba_spike_idx.size==0:
                    print('Skipping Trial:', i)
                    continue
                else:
                    print('delay in jaaba')
                    for val in jaaba_spike_idx: 
                        if lat_data3.lat_nidaq_filt[i][val] - lat_data1.lat_nidaq_filt[i][val] > 1.0:
                            mismatch_peaks_jaaba[i][val] = lat_data3.lat_nidaq_filt[i][val]
                            jaaba_spikes += 1
            else:
                for val in imagedispatch_spike_idx: 
                    if lat_data2.lat_nidaq_filt[i][val] - lat_data1.lat_nidaq_filt[i][val] > 1.0: 
                        mismatch_peaks_imagedispatch[i][val] = lat_data2.lat_nidaq_filt[i][val]
                        imagedispatch_spikes += 1
        else:     
            if isImagegrab:
               set1 = set(imagegrab_spike_idx.flatten())
            if isImagedispatch:
               set2 = set(imagedispatch_spike_idx.flatten())
            if isJaaba:
               set3 = set(jaaba_spike_idx.flatten())
            total_spikes = 0
            
            if isImagegrab and isImagedispatch:
                mismatch_1 = set1 ^ set2
            if isImagegrab and isJaaba:
                mismatch_2 = set1 ^ set3
            
            ## check if delay in frames with latency in imagegrab thread
            for val in imagegrab_spike_idx:
                total_spikes += 1
                if isImagedispatch:
                    if lat_data2.lat_nidaq_filt[i][val] > 4.0:
                        if lat_data2.lat_nidaq_filt[i][val] - lat_data1.lat_nidaq_filt[i][val] > 1.0:
                          print('Delay in image dispatch while having delay in imagegrab')
                          mismatch_peaks_imagedispatch[i][val] = lat_data2.lat_nidaq_filt[i][val]
                          imagedispatch_spikes += 1
                        else:
                           if isJaaba:    
                             if lat_data3.lat_nidaq_filt[i][val] > 4.0:
                               if lat_data3.lat_nidaq_filt[i][val] - lat_data1.lat_nidaq_filt[i][val] > 1.0:
                                   print('Delay in jaaba while having delay in imagegrab')
                                   mismatch_peaks_jaaba[i][val] = lat_data3.lat_nidaq_filt[i][val]
                                   jaaba_spikes += 1
                               else:
                                   mismatch_peaks_imagegrab[i][val] = lat_data1.lat_nidaq[i][val]
                                   imagegrab_spikes += 1
                             else:
                                print('Something is jaaba wrong')
                           else:
                              mismatch_peaks_imagegrab[i][val] = lat_data1.lat_nidaq[i][val]
                              imagegrab_spikes+=1
                    else:
                        print('Something is imagedispatch wrong')
            
            # check for latency in imagedispatch not in imagegrab
            if isImagegrab and isImagedispatch:
              if len(mismatch_1) != 0:
                for val in mismatch_1:
                    if lat_data2.lat_nidaq_filt[i][val] > 4.0:
                        if lat_data2.lat_nidaq_filt[i][val] - lat_data1.lat_nidaq_filt[i][val] > 1.0:
                            total_spikes += 1
                            imagedispatch_spikes += 1 
                            mismatch_peaks_imagedispatch[i][val] = lat_data2.lat_nidaq_filt[i][val]
            
            # check for latency in jaaba not in imagegrab and imagedispatch
            if isImagegrab and isJaaba:
              if len(mismatch_2) != 0:
                for val in mismatch_2:
                    if lat_data3.lat_nidaq_filt[i][val] > 4.0:
                        if lat_data3.lat_nidaq_filt[i][val] - lat_data2.lat_nidaq_filt[i][val] > 1.0:
                            total_spikes += 1
                            jaaba_spikes += 1
                            mismatch_peaks_jaaba[i][val] = lat_data3.lat_nidaq_filt[i][val]
             
            # print('imageDispatch spike idx', imagedispatch_spike_idx)
            print('imagegrab_spikes nidaq', imagegrab_spikes)
            print('imagedispatch spikes nidaq', imagedispatch_spikes)
            print('jaaba spikes nidaq', jaaba_spikes)
            
            
            # len_peaks,peaks1 = maxPeak(lat_data1.lat_nidaq_filt[i], testconfig['latency_threshold'], 10,
            #         max_peaks_imagegrab[i],testconfig)
            # len_peaks, peaks2 = maxPeak(lat_data2.lat_nidaq_filt[i], testconfig['latency_threshold'], 10,
            #          max_peaks_imagegrab[i],testconfig)
            # len_peaks, peaks3 = maxPeak(lat_data3.lat_nidaq_filt[i], testconfig['latency_threshold'], 10,
            #         max_peaks_imagegrab[i],testconfig)
             

            plot_thres_data_singletrial(mismatch_peaks_imagegrab[i], mismatch_peaks_imagedispatch[i], 
                         mismatch_peaks_jaaba[i], shape, color[color_id], alpha, ax_handle,
                         1, latency_threshold, numFrames,title, cam_id, legend)

    
def compare_lat_thres_f2f(lat_data1, lat_data2, lat_data3, cam_id, testconfig, 
                          ax_handle, color_id):
    
    no_of_trials = testconfig['no_of_trials']
    numFrames = testconfig['numFrames']
    latency_threshold = testconfig['latency_threshold']
    
    # mismatch_peaks_cam0 = []
    # mismatch_peaks_cam1 = []
    
    mismatch_peaks_imagegrab = no_of_trials*[numFrames*[0.0]]
    mismatch_peaks_imagedispatch = no_of_trials*[numFrames*[0.0]]   
    mismatch_peaks_jaaba = no_of_trials*[numFrames*[0.0]]
        
    for i in range(0,1):
        
        # plotting variables
        # fig, ax = plt.subplots()
        #fig, ax_handle = set_plot_var('',1) 
        shape= ['+', '*', 'x', '.', '^', 'o']
        color = ['r', 'b','g', 'm', 'c', 'k']
        alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]    
        title= 'Latencies from different threads Camera '
        legend = ['Imagegrab', 'Imagedispatch', 'Jaaba']
    
        ## flags to see if spike in each thread for each trial 
        isImagegrab=1
        isImagedispatch=1
        isJaaba=1
        
        #count spikes in each thread for each trial
        imagegrab_spikes=0
        imagedispatch_spikes=0
        jaaba_spikes=0
        total_spikes = 0
        
        imagegrab_spike_idx = []
        imagedispatch_spike_idx = []
        jaaba_spike_idx = []
        
        #filterf2f(lat_data1, lat_data2, lat_data3, testconfig)
  
        if not np.any(lat_data1.lat_f2f[i][:]):
            print('No imagegrab')
            return
    
        if not np.any(lat_data2.lat_f2f[i][:]):
            print('No imagedispatch')
            isImagedispatch = 0
    
        if not np.any(lat_data3.lat_f2f[i][:]):
            print('No jaaba')
            isJaaba = 0
        
        if isImagegrab:       
            imagegrab_spike_idx = np.argwhere(lat_data1.lat_f2f[i][:] > 4).flatten()
        if isImagedispatch:
            imagedispatch_spike_idx = np.argwhere(lat_data2.lat_f2f[i][:] > 4).flatten()
        if isJaaba:
            jaaba_spike_idx = np.argwhere(lat_data3.lat_f2f[i][:] > 4).flatten()
        
        print('trial ', i)  

        ## if no latency spike is present 
        if imagegrab_spike_idx.size == 0:
            if imagedispatch_spike_idx.size==0:
                if jaaba_spike_idx.size==0:
                    print('Skipping Trial:', i)
                    continue
                else:
                    for val in jaaba_spike_idx: 
                        if lat_data3.lat_f2f[i][val] - lat_data1.lat_f2f[i][val] > 1.0:
                            print('Latency only in Jaaba')
                            mismatch_peaks_jaaba[i][val] = lat_data3.lat_f2f[i][val]
                            jaaba_spikes += 1
            else:
                for val in imagedispatch_spike_idx: 
                    if lat_data2.lat_f2f[i][val] - lat_data1.lat_f2f[i][val] > 1.0: 
                        print('Latency Only in Imagedispatch')
                        mismatch_peaks_imagedispatch[i][val] = lat_data2.lat_f2f[i][val]
                        imagedispatch_spikes += 1
        else:     
            if isImagegrab:
               set1 = set(imagegrab_spike_idx.flatten())
            if isImagedispatch:
               set2 = set(imagedispatch_spike_idx.flatten())
            if isJaaba:
               set3 = set(jaaba_spike_idx.flatten())
            
            
            if isImagegrab and isImagedispatch:
                mismatch_1 = set1 ^ set2
            if isImagegrab and isJaaba:
                mismatch_2 = set1 ^ set3
            
            ## check if delay in frames with latency in imagegrab thread
            for val in imagegrab_spike_idx:
                total_spikes += 1
                if isImagedispatch:
                    if lat_data2.lat_f2f[i][val] > 4.0:
                        if lat_data2.lat_f2f[i][val] - lat_data1.lat_f2f[i][val] > 1.0:
                            print('Delay in image dispatch while having delay in imagegrab')
                            mismatch_peaks_imagedispatch[i][val] = lat_data2.lat_f2f[i][val]
                            imagedispatch_spikes += 1
                        
                        mismatch_peaks_imagegrab[i][val] = lat_data1.lat_f2f[i][val]
                        imagegrab_spikes+=1
                else:
                    mismatch_peaks_imagegrab[i][val] = lat_data1.lat_f2f[i][val]
                    imagegrab_spikes +=1
            
            # check for latency in imagedispatch not in imagegrab
            if isImagegrab and isImagedispatch:
                if len(mismatch_1) != 0:
                    for val in mismatch_1:
                        if lat_data2.lat_f2f[i][val] > 4.0 and val !=0:
                            if lat_data2.lat_f2f[i][val] - lat_data1.lat_f2f[i][val] > 1.0:
                                total_spikes += 1
                                imagedispatch_spikes += 1 
                                mismatch_peaks_imagedispatch[i][val] = lat_data2.lat_f2f[i][val]
                                print(lat_data2.lat_f2f[i][val])
            
            
            print('imagegrab_spike_idx', imagegrab_spike_idx)
            print('imagedispatch_spike_idx', isImagedispatch)
            print('imagegrab_spikes f2f', imagegrab_spikes)
            print('imagedispatch f2f', imagedispatch_spikes)
            #print('jaaba spikes f2f', jaaba_spikes
            print('total spikes', total_spikes)
            
            #print(mismatch_peaks_imagegrab[i])
            plot_thres_data_singletrial(mismatch_peaks_imagegrab[i], mismatch_peaks_imagedispatch[i], 
                         mismatch_peaks_jaaba[i], shape, color[color_id], alpha, ax_handle,
                         1, latency_threshold, numFrames,title, cam_id, legend)
    
def compare_mismatch(lat_data1, lat_data2, 
                   lat_data3, cam_id, Config):    

    fig, ax_handle = set_plot_var('',1)
    color_id = 1
    compare_lat_thres_nidaq(lat_data1, lat_data2, 
                    lat_data3, cam_id, Config, ax_handle, color_id)
    color_id = 2
    compare_lat_thres_f2f(lat_data1, lat_data2, 
                    lat_data3, cam_id, Config, ax_handle, color_id)    
    
    title= 'Latencies from different threads Camera '
    legend = ['Imagegrab', 'Imagedispatch', 'Jaaba']
    
    plt.suptitle(title + str(cam_id),fontsize=17)    
    plt.setp(ax_handle, xlabel='Frames', ylabel='Milliseconds')
    plt.legend(legend)

def main():
    
    config_file  = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/config_files/short/jaaba_plugin_multicamera_shorttrial_run_b67a7_6_13_2022.csv'
     
    Config = {
    
        'filename':'',
        'numCameras': 0,
        'cam_suffix':[], 
        'dir_len': 0,
        'dir_list': [],
        'numFrames': 0,
        'no_of_trials': 0,
        'framerate': 0,
        'latency_threshold':0.0,
        'cam_dir': '',
        'nidaq_prefix': '',
        
        'f2f_prefix': '',
        'queue_prefix': '',
        'plugin_prefix': '',
        'logging_prefix': '',
        'framegrab_prefix': '',
        'git_commit': '',
        'date': '',
        'count_latencyspikes_nidaq':[],
        'average_normspikes_nidaq':[],
        'mean_spikes_nidaq': [],
        'std_spikes_nidaq': [],
        'spikes_per_sec_nidaq' : [],
        'max_skippedFrames_nidaq': 0,
        'fracIntwspikes_nidaq': [],
        'count_latencyspikes_f2f':[],
        'average_normspikes_f2f':[],
        'mean_spikes_f2f': [],
        'std_spikes_f2f': [],
        'spikes_per_sec_f2f': [],
        'fracIntwspikes_f2f': [],    
         
    }
                   
    ##
    readConfigFile(config_file, Config)
    
    no_of_trials = Config['no_of_trials']
    numFrames = Config['numFrames']
    numCameras = Config['numCameras']    
    plugin_prefix = Config['plugin_prefix']
    
    isPlugin = 0
    isJaaba = 0
    islogging = 0
    isCamOnly = 0
    
    ## mode to run in BIAS
    plugin_prefix = Config['plugin_prefix']
    logging_prefix = Config['logging_prefix']
    framegrab_prefix = Config['framegrab_prefix'] 
    
    ## set BIAS mode configuration flags
    islogging = setFlags(logging_prefix)
    isCamOnly = setFlags(framegrab_prefix)
    isPlugin = setFlags(plugin_prefix)
    
    if plugin_prefix == 'jaaba_plugin':
        isJaaba = 1
        
    latency_metric = LatencyMetric(1, 1, 0, 1)
    
    biasConfig_mode = BiasConfigMode(isCamOnly, islogging, isPlugin, isJaaba)

    latency_data_imagegrab_cam1 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                np.array(no_of_trials*[numFrames * [0.0]]), \
                                np.array(no_of_trials*[numFrames * [0.0]]), \
                                np.array(no_of_trials*[numFrames * [0.0]]), \
                                np.array(no_of_trials*[numFrames * [0.0]]), \
                                np.array(no_of_trials*[numFrames * [0.0]]))
        
    if numCameras == 2:
        latency_data_imagegrab_cam2 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]),\
                                            np.array(no_of_trials*[numFrames * [0.0]]),\
                                            np.array(no_of_trials*[numFrames * [0.0]]),\
                                            np.array(no_of_trials*[numFrames * [0.0]]),\
                                            np.array(no_of_trials*[numFrames * [0.0]]),\
                                            np.array(no_of_trials*[numFrames * [0.0]]))
                    
    latency_data_imagedispatch_cam1 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                np.array(no_of_trials*[numFrames * [0.0]]))
        
    if numCameras == 2:
        latency_data_imagedispatch_cam2 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]),\
                                            np.array(no_of_trials*[numFrames * [0.0]]),\
                                            np.array(no_of_trials*[numFrames * [0.0]]),\
                                            np.array(no_of_trials*[numFrames * [0.0]]),\
                                            np.array(no_of_trials*[numFrames * [0.0]]),\
                                            np.array(no_of_trials*[numFrames * [0.0]]))

    latency_data_jaaba_cam1 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                            np.array(no_of_trials*[numFrames * [0.0]]), \
                            np.array(no_of_trials*[numFrames * [0.0]]), \
                            np.array(no_of_trials*[numFrames * [0.0]]), \
                            np.array(no_of_trials*[numFrames * [0.0]]), \
                            np.array(no_of_trials*[numFrames * [0.0]]))
    
    if numCameras == 2:
        latency_data_jaaba_cam2 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]),\
                                        np.array(no_of_trials*[numFrames * [0.0]]),\
                                        np.array(no_of_trials*[numFrames * [0.0]]),\
                                        np.array(no_of_trials*[numFrames * [0.0]]),\
                                        np.array(no_of_trials*[numFrames * [0.0]]),\
                                        np.array(no_of_trials*[numFrames * [0.0]]))
    
    bias_config=''
    if plugin_prefix:
        bias_config = Config['plugin_prefix']
          
        print(bias_config)    
        cam_id = 0
        readLatency_data(latency_data_jaaba_cam1, Config, latency_metric,\
                          bias_config, cam_id)
    
        if numCameras > 1:        
            copy_camtrig(latency_data_jaaba_cam1, latency_data_jaaba_cam2)
            cam_id = 1
            readLatency_data(latency_data_jaaba_cam2, Config, latency_metric,\
                          bias_config, cam_id)
            
    if framegrab_prefix:
        bias_config = Config['framegrab_prefix']    
        
    print(bias_config)    
    cam_id = 0
    readLatency_data(latency_data_imagegrab_cam1, Config, latency_metric,\
                      bias_config, cam_id)
    
    if numCameras > 1:    
        copy_camtrig(latency_data_imagegrab_cam1, latency_data_imagegrab_cam2)
        cam_id = 1
        readLatency_data(latency_data_imagegrab_cam2, Config, latency_metric,\
                      bias_config, cam_id)  
   
    bias_config = 'imagedispatch'
    print(bias_config)    
    cam_id=0    
    readLatency_data(latency_data_imagedispatch_cam1, Config, latency_metric,\
                      bias_config, cam_id)

    if numCameras > 1:        
        copy_camtrig(latency_data_imagedispatch_cam1, latency_data_imagedispatch_cam2)
        cam_id = 1
        readLatency_data(latency_data_imagedispatch_cam2, Config, latency_metric,\
                      bias_config, cam_id)

    #ax_handle = return_fighandle()
    # filterf2f(latency_data_imagegrab_cam1, latency_data_imagedispatch_cam1, 
    #             latency_data_jaaba_cam1, Config)
    
    # filterf2f(latency_data_imagegrab_cam2, latency_data_imagedispatch_cam2, 
    #             latency_data_jaaba_cam2, Config)
            
    print("Analyse Camera 0 ")
    cam_id=0
    #correlate_skips(latency_data_imagegrab_cam1,
    #                latency_data_imagedispatch_cam1,
    #                latency_data_jaaba_cam1, Config)
    

    #compare_mismatch(latency_data_imagegrab_cam1, latency_data_imagedispatch_cam1, 
    #                latency_data_jaaba_cam1, cam_id, Config)
    #compare_lat_thres_nidaq(latency_data_imagegrab_cam1, latency_data_imagedispatch_cam1,
    #               latency_data_jaaba_cam1, cam_id, Config, ax_handle, 1)
    
    # compare_lat_thres_f2f(latency_data_imagegrab_cam1, latency_data_imagedispatch_cam1, 
    #               latency_data_jaaba_cam1, cam_id, Config, ax_handle, color_id)

    
    print("Analyse Camera 1 ")
    #if numCameras > 1:
        #cam_id=1
        #correlate_skips(latency_data_imagegrab_cam2,
        #                latency_data_imagedispatch_cam2, 
        #                latency_data_jaaba_cam2, 
        #                Config)

        #compare_mismatch(latency_data_imagegrab_cam2, latency_data_imagedispatch_cam2, 
        #            latency_data_jaaba_cam2, cam_id, Config)
        #compare_lat_thres_nidaq(latency_data_imagegrab_cam2, latency_data_imagedispatch_cam2,
        #           latency_data_jaaba_cam2, cam_id, Config, ax_handle, 1)
        
        # compare_lat_thres_f2f(latency_data_imagegrab_cam2, latency_data_imagedispatch_cam2, 
        #           latency_data_jaaba_cam2, cam_id, Config, ax_handle, color_id)
        
    
    ## plot data
    plot_matching_data(latency_data_imagegrab_cam1, latency_data_imagedispatch_cam1,
                  latency_data_jaaba_cam1, Config, 0)
    plot_matching_data(latency_data_imagegrab_cam2, latency_data_imagedispatch_cam2,
                  latency_data_jaaba_cam2, Config, 1)
    
    #count_latencies(latency_data_imagegrab_cam1, latency_data_imagedispatch_cam1,
    #                latency_data_jaaba_cam1, 0, Config)
    #count_latencies(latency_data_imagegrab_cam2, latency_data_imagedispatch_cam2,
    #                latency_data_jaaba_cam2, 1,Config)
    
    
if __name__ == "__main__":
    
    main()