# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 20:47:52 2021

@author: 27rut
"""

import numpy as np 
import csv
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
from collections import namedtuple
import re
import write_csvconfigfiles as wcs
import plotting_code as pc


class LatencyData:  
 
    def __init__(self, arr1, arr2, arr3, arr4, arr5, arr6):
        # initializing instance variable
        self.lat_nidaq = arr1
        self.lat_f2f = arr2
        self.lat_queue = arr3
        self.lat_camtrig = arr4
        self.lat_nidaq_filt = arr5
        self.process_time= arr6
          
class LatencyMetric:

    
    def __init__(self, flag1, flag2,flag3):
        # initializing instance variable
        self.isnidaq=flag1;
        self.isframetoframe=flag2
        self.isqueue=flag3
    
    
class BiasConfigMode:
          
    def __init__(self, flag1, flag2,flag3,flag4):
        # initializing instance variable
        self.isCamOnly = flag1
        self.islogging = flag2
        self.isPlugin =  flag3
        self.isJaaba  =  flag4
    
            
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
                        print(row[idy])
                        print(keys[idx])
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
                    
def readData(testconfig, lat_metric, lat_data_cam1, lat_data_cam2, bias_config):

    cam_suffix = testconfig['cam_suffix']
    cam_id = int(cam_suffix[0])
    readLatency_data(lat_data_cam1, testconfig, lat_metric,\
                      bias_config, cam_id)

    if testconfig['numCameras'] == 2:
         
        copy_camtrig(lat_data_cam1, lat_data_cam2)
        cam_id = int(cam_suffix[1])
        readLatency_data(lat_data_cam2, testconfig, lat_metric,\
                          bias_config, cam_id)

        
def readcsvFile_nidaq(filename, arr_lat, arr_cam, cam_id, plugin_prefix):

    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')

    for idx, row in enumerate(data_grab):

        arr_cam[idx] = ((np.float(row[0]))) ## will store the count corresponding to camera trigger
        arr_lat[idx] = (((np.float(row[1])-np.float(row[0])) * 0.02)) ## latency calculation between
                                               ## event and camera trigger, fast clock is 50khz
                                               ## hence multiplying factor is (1/50khz- period) 0.02 to calculate latency
        #print((np.float(row[1])-np.float(row[0])) * 0.02)
                         
    fhandle.close()
    
    
def readcsvFile_f2f(filename, arr, f2f_flag, cam_id, plugin_prefix):
 
    if cam_id == 1 and plugin_prefix == 'jaaba_plugin':
        return
 
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
        
    
def readcsvFile_int(filename, arr, cam_id, plugin_prefix):
    
    if cam_id == 1 and plugin_prefix == 'jaaba_plugin':
        return
    
    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')
    
    for idx,row in enumerate(data_grab):
        arr[idx] = np.int(row[0])                    
    fhandle.close()


def readLatency_data(lat_dat, testconfig, lat_metric, biasmode_prefix, \
                     cam_id):

    no_of_trials = np.int(testconfig['no_of_trials'])
    numFrames = np.int(testconfig['numFrames'])
    plugin_prefix = testconfig['plugin_prefix']
    
    path_dir = testconfig['dir_list'][0] 
    
    if(numFrames > 100000):
        trial_suffix = '_long_trial'
    else:
        trial_suffix = '_short_trial'

    for i in range(0, no_of_trials):
        # read latency readings from nidaq
        print('Trial', i)
        if lat_metric.isnidaq:

            filename = path_dir + testconfig['nidaq_prefix'] \
            + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
            testconfig['date'] + '/' + biasmode_prefix + '_' + \
            testconfig['nidaq_prefix'] +  'cam' + testconfig['cam_suffix'][cam_id]\
            + trial_suffix + str(i+1) + '.csv'
            
            readcsvFile_nidaq(filename, lat_dat.lat_nidaq[i], lat_dat.lat_camtrig[i], \
                          cam_id, plugin_prefix)

        ##read latency readings from computer time stamps
        if lat_metric.isframetoframe:
            
            filename = path_dir + testconfig['f2f_prefix'] \
            + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' +\
            testconfig['date'] + '/' + biasmode_prefix + '_' + \
            testconfig['f2f_prefix'] +  'cam' + testconfig['cam_suffix'][cam_id] \
            + trial_suffix + str(i+1) + '.csv'
    
            readcsvFile_f2f(filename, lat_dat.lat_f2f[i], 1, cam_id, plugin_prefix)
            
            filename = path_dir + testconfig['f2f_prefix'] \
            + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' +\
            testconfig['date'] + '/' + biasmode_prefix + '_' + \
            testconfig['f2f_prefix'] +  'cam' + testconfig['cam_suffix'][cam_id] \
            + trial_suffix + str(i+1) + '.csv'
    
            readcsvFile_f2f(filename, lat_dat.process_time[i], 0, cam_id, plugin_prefix)
            
    
        ## read queue size 
        if lat_metric.isqueue:
            
            filename = path_dir + testconfig['queue_prefix'] \
            + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
            testconfig['date'] + '/'+ biasmode_prefix + '_' + \
            testconfig['queue_prefix'] +  'cam' + \
            testconfig['cam_suffix'][cam_id] + trial_suffix + str(i+1) + '.csv'  
    
            readcsvFile_int(filename, lat_dat.lat_queue[i], cam_id, plugin_prefix)
            
            
def process_data(testconfig, lat_metric, lat_data_cam1, lat_data_cam2, 
                 bias_config, debug_flag):

    numCameras = testconfig['numCameras']
    cam_suffix = testconfig['cam_suffix']

    readData(testconfig, lat_metric, lat_data_cam1, lat_data_cam2, bias_config)
    cam_id = int(cam_suffix[0])
    meanNumberSpikes(lat_data_cam1, testconfig, lat_metric, cam_id, debug_flag)

    if debug_flag:
        logging_function(testconfig, cam_id)

    if numCameras == 1 :
        plot_data(lat_data_cam1, None, lat_metric, bias_config, testconfig,\
                  debug_flag)
    
    if numCameras == 2:
        
        cam_id = int(cam_suffix[1])      
        meanNumberSpikes(lat_data_cam2, testconfig, lat_metric, cam_id, debug_flag)
        
        if debug_flag:
            logging_function(testconfig, cam_id)

        plot_data(lat_data_cam1, lat_data_cam2, lat_metric, \
                  bias_config, testconfig, debug_flag)


def analyze_data(testconfig, lat_metric, \
                 biasmode_config, debug_flag):
    
    numCameras = testconfig['numCameras']
    numFrames = int(testconfig['numFrames'])
    no_of_trials = testconfig['no_of_trials']
    cam_suffix = testconfig['cam_suffix']
    
    ## mode to run in BIAS
    plugin_prefix = biasmode_config.isPlugin
    logging_prefix = biasmode_config.islogging
    framegrab_prefix = biasmode_config.isCamOnly

    if framegrab_prefix:
        bias_config = testconfig['framegrab_prefix']

        latency_data_cam1 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]))
        
            
        if numCameras==2:
            latency_data_cam2 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]))

        process_data(testconfig, lat_metric, latency_data_cam1, latency_data_cam2, \
                     bias_config, debug_flag)

    if logging_prefix:

        bias_config = testconfig['logging_prefix']
        latency_data_cam1 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]))
        
            
        if numCameras == 2:
            latency_data_cam2 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]))


        process_data(testconfig, lat_metric, latency_data_cam1, latency_data_cam2, \
                     bias_config, debug_flag)
        
    
    if plugin_prefix:

        bias_config = testconfig['plugin_prefix']
        latency_data_cam1 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]))

        if numCameras == 2:
            latency_data_cam2 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]))

        process_data(testconfig, lat_metric, latency_data_cam1, latency_data_cam2, \
                     bias_config, debug_flag)
        

def setFlags(flag_prefix) :
    
    if flag_prefix != '':
        return 1
    else:
        return 0
    
    
def count_numSpikes(lat_dat, testconfig, lat_metric, cam_id):
    
    no_of_trials = testconfig['no_of_trials']
    latency_threshold = testconfig['latency_threshold']
    numFrames = testconfig['numFrames']
    calc_mean = 1
    avg=0.0
    print('*************', latency_threshold)

    if(lat_metric.isframetoframe and len(testconfig['count_latencyspikes_f2f']) != 0):
        for trial_id in range(0,no_of_trials):
            for i in range(0, numFrames):

                if(lat_dat.lat_f2f[trial_id][i] == None):
                    continue
        
                if(lat_dat.lat_f2f[trial_id][i] > latency_threshold):
                    testconfig['count_latencyspikes_f2f'][cam_id][trial_id] += 1

            testconfig['average_normspikes_f2f'][cam_id][trial_id] = \
                    float(testconfig['count_latencyspikes_f2f'][cam_id][trial_id])/ float(numFrames)      
             

    if(lat_metric.isnidaq and len(testconfig['count_latencyspikes_nidaq']) != 0):
        for trial_id in range(0,no_of_trials):
            if len(lat_dat.lat_nidaq_filt) != 0:
                [count_peaks, loc_peaks] = maxPeak(lat_dat.lat_nidaq[trial_id],\
                latency_threshold, 10, lat_dat.lat_nidaq_filt[trial_id],testconfig)

                print('Count Peaks Jaaba', count_peaks)
                if len(testconfig['count_latencyspikes_nidaq']) < no_of_trials:
                    testconfig['count_latencyspikes_nidaq'][cam_id][trial_id] = int(count_peaks)
            
                # bad name choice
                if len(testconfig['average_normspikes_nidaq']) < no_of_trials:
                    testconfig['average_normspikes_nidaq'][cam_id][trial_id] = \
                    float(testconfig['count_latencyspikes_nidaq'][cam_id][trial_id]) / float(numFrames)

            if calc_mean and testconfig['framegrab_prefix'] == 'imagegrab':
                avg += np.sum(lat_dat.lat_nidaq[trial_id][:])/float(numFrames)

        if calc_mean and testconfig['framegrab_prefix'] == 'imagegrab':
            avg = avg/float(no_of_trials)
            print('The average latency for imagegrab is ', avg)
            
def count_nsecintervals_wspikes(arr, testconfig, cam_id, interval_length):
    
    no_of_trials = int(testconfig['no_of_trials'])
    numFrames = int(testconfig['numFrames'])
    latency_threshold = testconfig['latency_threshold']
    
    count_intervalswspikes = np.array(no_of_trials*[0])
    count_intervals = 0;
    numIntervals = (numFrames//interval_length)
    noframes = interval_length * numIntervals 
    left_over = numFrames - noframes
    
    if left_over != 0:
        numIntervals += 1
   
    for trial_id in range(0, no_of_trials):
        count_intervals=0
        for i in range(0,  noframes, interval_length): 
         
            for j in range(0, interval_length-1): 
                if(arr[trial_id][i] == None):
                    continue
                
                if(arr[trial_id][i+j] >= latency_threshold):               
                    count_intervals += 1
                    break;
        for j in range(0, left_over):
            if(arr[trial_id][noframes+j] >= latency_threshold):               
                count_intervals += 1
                break;
                    
        count_intervalswspikes[trial_id] = count_intervals            
     
    count_intervalswspikes = count_intervalswspikes/numIntervals  

    if 1:
        print('fraction of intervals with spikes', count_intervalswspikes)

    return count_intervalswspikes
    
                            
## distance - distance between peaks
## height -  definition of height of a peak        
def maxPeak(latency_arr, height, distance, latency_filt_arr, testconfig):
    
    peaks = []
    numFrames = len(latency_arr)
    i = 0    
     
    while i < numFrames:
        bout_count=0
        if latency_arr[i] > height:
            peaks.append(i)
            latency_filt_arr[i] = latency_arr[i]
            i += 1
            while i < numFrames and latency_arr[i] >= height:
                latency_filt_arr[i] = 2.2
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
    

def copy_camtrig(lat_data1, lat_data2):
    
    lat_data2.lat_camtrig = lat_data1.lat_camtrig
  
    
def set_plot_var(queue_prefix, no_of_trials):
     
    if(no_of_trials > 1):
            
        fig, axes = plt.subplots(no_of_trials, 1, figsize=(14,12))
        ax = axes[0].get_gridspec()
       
    else:
        fig, axes = plt.subplots(1, 1, figsize=(22,20))
        ax = axes.get_gridspec()

    fig.subplots_adjust(hspace=0.5)

    # if no_of_trials/ > 1:
    #     axes[ax.nrows-1, ax.ncols-2] = plt.subplot2grid((rows, cols), (2,0), colspan = 2)
        
    if queue_prefix:
        plt.setp(axes, yticks = np.arange(0,20,1), ylim = [0,20])
    else:
        plt.setp(axes, yticks = np.arange(0,30,2), ylim=[0,20])
   
    return fig, axes
  

def plot_raw_data(arr, shape, color, alpha, labels, ax_handle,\
                  no_of_trials, latency_threshold, numFrames, title, cam_id):
         
    if cam_id == 0:
        color_id = 0
    else:
        color_id = 3 

    if no_of_trials > 1:
        
        ax = ax_handle[0].get_gridspec()
 
        for ix in range(0, ax.nrows):
            
            if cam_id == 0:
                
                if ix == 0:
                    ax_handle[ix].plot(arr[ix], '.', 
                                       color=color[color_id],\
                                       marker=shape[ix],\
                                       alpha=1, ms=8, label='Cam' + str(cam_id))
                else:
                    ax_handle[ix].plot(arr[ix], '.', 
                                       color=color[color_id],\
                                       marker=shape[ix],\
                                       alpha=1, ms=8)
            else:
                if ix == 0:
                    ax_handle[ix].plot(arr[ix], '.', 
                                       color=color[color_id] ,\
                                       marker=shape[ix],\
                                       alpha=1, ms=8, label='Cam' + str(cam_id))
                else:
                    ax_handle[ix].plot(arr[ix], '.', 
                                       color=color[color_id] ,\
                                       marker=shape[ix],\
                                       alpha=1, ms=8,label= '_nolegend_')
            ax_handle[ix].set_title('Trial ' + str(ix + 1))
            ax_handle[ix].tick_params(axis='x', labelsize=10)
            ax_handle[ix].tick_params(axis='y', labelsize=10)
            ax_handle[ix].plot(np.array(latency_threshold*np.ones(numFrames)),\
                                  label='_nolegend_')
        plt.suptitle(title + ' Cam' + str(cam_id),fontsize=17)
        plt.setp(ax_handle[:], xlabel = 'Frames', ylabel ='Milliseconds')
    else:
        ax = ax_handle.get_gridspec()
        ax_handle.plot(arr[0], '.', \
                                color=color[color_id] ,\
                                marker=shape[0],\
                                alpha=1, ms=8, label='Cam' + str(cam_id))
        ax_handle.plot(np.array(latency_threshold*np.ones(numFrames)),\
                      label='_nolegend_')
        plt.suptitle(title + ' Cam' + str(cam_id), fontsize=17)
        plt.setp(ax_handle, xlabel='Frames', ylabel='Milliseconds')
    plt.show()
    
    
def matching_peaks(lat_arr_nidaq, lat_arr_f2f, shape, color, alpha,\
                          labels, ax_handle, no_of_trials,
                          latency_threshold, numFrames, title, cam_id):
    
    
    no_of_mismatches = np.array(no_of_trials*[0], np.int32)
    index_mismatches = []
    
    for trial_id in range(0, no_of_trials):
        trial_mismatches = []
        for frame_id in range(1, numFrames):
            if abs(lat_arr_nidaq[trial_id][frame_id] - \
                   lat_arr_f2f[trial_id][frame_id]) > latency_threshold :
                       no_of_mismatches[trial_id] += 1
                       trial_mismatches.append(frame_id)
                     
        index_mismatches.append(trial_mismatches)          
        print('Mismatched indexes', index_mismatches)  
        if no_of_mismatches[trial_id] > 0:
            print('Trial ' + str(trial_id) + ' has mismatches ' + str(no_of_mismatches[trial_id]))
            
   
    if no_of_trials >  1:    
        ax = ax_handle[0].get_gridspec()
        for ix in range(0, ax.nrows):
  
            ax_handle[ix].plot(abs(lat_arr_nidaq[ix][1:] - \
                                      lat_arr_f2f[ix][1:]),\
                         '.', color=color[0] , \
                         marker=shape[ix], \
                         alpha=1, label='Red')
            ax_handle[ix].plot(np.array(latency_threshold*np.ones(numFrames)),\
                                       label='_nolegend_')  
        plt.suptitle(title,fontsize=17)
        plt.setp(ax_handle[:], xlabel = 'Frames', ylabel ='Milliseconds')                
    else:
        ax_handle[0].plot(abs(lat_arr_nidaq[0][1:] - \
                       lat_arr_f2f[0][1:]), '.', \
                       color=color[0] ,\
                       marker=shape[0], \
                       alpha=1, ms=8, label='Cam' + str(cam_id)) 
        ax_handle[0].plot(np.array(latency_threshold*np.ones(numFrames)),\
                       label='_nolegend_') 
        plt.suptitle(title,fontsize=17)    
        plt.setp(ax_handle, xlabel = 'Frames', ylabel ='Milliseconds')    
        
def compare_peaks(lat_arr_nidaq, lat_arr_f2f, shape, color, alpha, \
                          labels, ax_handle,no_of_trials, 
                          latency_threshold, numFrames,\
                          title, cam_id):
    
    if no_of_trials > 1:    
        ax = ax_handle[0].get_gridspec()
        for ix in range(0, ax.nrows):
           
            ax_handle[ix].plot(lat_arr_nidaq[ix] ,\
                         '.', color=color[cam_id] , \
                         marker=shape[ix+cam_id], \
                         alpha=1, label='Cam' + str(cam_id))
            ax_handle[ix].plot(lat_arr_f2f[ix] ,\
                         '.', color=color[cam_id] , \
                         marker=shape[ix+cam_id], \
                         alpha=1, label='Cam' + str(cam_id))    
            #ax_handle[ix].plot(np.array(latency_threshold*np.ones(numFrames)),\
            #                       label='_nolegend_')  
        plt.suptitle(title,fontsize=17)
        plt.setp(ax_handle[:], xlabel = 'Frames', ylabel ='Milliseconds')                
    else:
        ax_handle.plot(lat_arr_nidaq[0],\
                        '.', \
                       color=color[cam_id] ,\
                       marker=shape[cam_id], \
                       alpha=1, ms=8, label='Cam' + str(cam_id)) 
        ax_handle.plot(lat_arr_f2f[0] ,\
                                 '.', color=color[0+cam_id] , \
                                 marker=shape[0+cam_id], \
                                 alpha=1, label='Cam' + str(cam_id))        
        #ax_handle.plot(np.array(latency_threshold*np.ones(numFrames)),\
        #               label='_nolegend_') 
        plt.suptitle(title,fontsize=17)    
        plt.setp(ax_handle, xlabel = 'Frames', ylabel ='Milliseconds')  
        

def plot_queue_length(lat_data_cam1, lat_data_cam2, shape, color, alpha, \
                      ax_handle, no_of_trials, 
                      numCameras, numFrames,\
                      title, plugin_prefix):
    
    if no_of_trials > 1:    
        ax = ax_handle[0].get_gridspec()
        for ix in range(0, ax.nrows):

            ax_handle[ix].plot(lat_data_cam1.lat_queue[ix],\
                          '.', color=color[ix] , \
                          marker=shape[ix], \
                          alpha=0.5)

            ax_handle[ix].plot(lat_data_cam2.lat_queue[ix] ,\
                          '.', color=color[ix+1] , \
                          marker=shape[ix+1], \
                          alpha=1)    
           
           
        plt.suptitle(title,fontsize=17)
        plt.setp(ax_handle[:], xlabel = 'Frames', ylabel ='No. of frames in queue')                
    else:
        ax_handle.plot(lat_data_cam1.lat_queue[0],\
                        '.', \
                        color=color[0] ,\
                        marker=shape[0], \
                        alpha=0.5, ms=8) 
            
        if numCameras == 2:    
            ax_handle.plot(lat_data_cam2.lat_queue[0],\
                            '.', \
                            color=color[1] ,\
                            marker=shape[1], \
                            alpha=1, ms=8)     
        plt.suptitle(title,fontsize=17)    
        plt.setp(ax_handle, xlabel = 'Frames', ylabel ='No. of frames in queue')  
    
            
def plot_data(lat_data_cam1, lat_data_cam2, lat_metric, bias_config, testconfig,\
              debug_flag):
      
    no_of_trials = int(testconfig['no_of_trials'])
    numFrames = int(testconfig['numFrames'])
    numCameras = int(testconfig['numCameras'])
    plugin_prefix = testconfig['plugin_prefix']
    latency_threshold = testconfig['latency_threshold']
    
    shape = ['+', '*', 'x', '.', '^', 'o']
    color = ['r', 'b','g', 'm', 'c', 'k']
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]    
    labels = [['Cam 1', 'Cam 2'], ['Cam 1']]
    marker_size=6

    if(numFrames > 100000):
        trial_suffix = '_long_trial'
    else:
        trial_suffix = '_short_trial'
    
    if lat_metric.isnidaq:
        print('BIAS mode', bias_config)
        figsaveFile = testconfig['dir_list'][0] + testconfig['nidaq_prefix'] + '/' + \
                        testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + testconfig['date'] +\
                        '/' + bias_config\
                        + '_' + testconfig['nidaq_prefix'] + trial_suffix + '_' + str(testconfig['cam_dir'])
        
        if no_of_trials == 1:
            title = bias_config + ' multi camera Nidaq latency timestamps' + '-' + trial_suffix
        else:
            title = bias_config + ' multi camera Nidaq latency timestamps' + '-' + trial_suffix

        if numCameras == 2: 
            arr1 = lat_data_cam1.lat_nidaq
            arr2 = lat_data_cam2.lat_nidaq

            fig, axes = pc.set_plot_var('', 1,numFrames, 20)
            pc.plot_raw_single_axis(arr1, arr2,shape, color, alpha, labels[0], axes, 1,\
                              latency_threshold, numFrames, title, 0,marker_size)
            if not debug_flag:
                fig.savefig(figsaveFile + 'cam0' + '.png')
            
        else:
            arr1 = lat_data_cam1.lat_nidaq
            fig, axes = set_plot_var('', no_of_trials)
            pc.plot_raw_data(arr1, shape, color, alpha, labels[1], axes, 1,\
                      latency_threshold, numFrames, title, 0,marker_size)
            if not debug_flag:
                fig.savefig(figsaveFile + 'cam0' + '.png')
        plt.show()
    if lat_metric.isframetoframe:

        figsaveFile = testconfig['dir_list'][0] + testconfig['f2f_prefix'] + '/' + \
                      testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + testconfig['date'] + \
                      '/' + bias_config + '_' + testconfig['f2f_prefix'] + trial_suffix + '_' \
                      + str(testconfig['cam_dir'])

        if no_of_trials == 1:
            title = testconfig['f2f_prefix'] + ' ' + testconfig['cam_dir'] + \
                ' ' + bias_config + trial_suffix
        else:
            title = testconfig['f2f_prefix'] + ' ' + testconfig['cam_dir'] + \
                ' ' + bias_config + trial_suffix

        if numCameras == 2: 
            
            arr1 = lat_data_cam1.lat_f2f
            arr2 = lat_data_cam2.lat_f2f
            arr3 = lat_data_cam1.process_time

            # plotting variables
            fig, axes = set_plot_var('', no_of_trials)
            pc.plot_raw_single_axis(arr1,arr2, shape, color, alpha, labels, axes, no_of_trials,\
                          latency_threshold, numFrames, title, 0,marker_size)
            if not debug_flag:
                fig.savefig(figsaveFile + 'cam0' + '.png')

        else:
            arr1 = lat_data_cam1.lat_f2f
            pc.plot_raw_data(arr1, shape, color, alpha, labels, axes, no_of_trials,\
                      latency_threshold, numFrames, title, 0,marker_size)
            if not debug_flag:
                fig.savefig(figsaveFile + 'cam0' + '.png')
        plt.show()
                
        fig.legend(axes, labels, loc = 'upper right')

        #diff plots
        fig, axes = set_plot_var('', no_of_trials)
        if no_of_trials == 1:
            title = testconfig['nidaq_prefix']  + ' and ' +  testconfig['f2f_prefix'] + \
                ' ' + testconfig['cam_dir'] + \
                ' ' + bias_config + trial_suffix + ' diff'
        else:
            title = testconfig['nidaq_prefix']  + ' and ' +  testconfig['f2f_prefix'] + \
                ' ' +  testconfig['cam_dir'] + \
                ' ' + bias_config + trial_suffix +' diff'
        matching_peaks(lat_data_cam1.lat_nidaq_filt, lat_data_cam1.lat_f2f,\
                       shape, color, alpha, labels[0], axes, no_of_trials,\
                       latency_threshold, numFrames, title, 0)   
        
        if not debug_flag:    
            if no_of_trials == 1:    
                fig.savefig(testconfig['dir_list'][0] + testconfig['f2f_prefix'] + '/' \
                        + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + testconfig['date'] +'/' + \
                        bias_config  + \
                        '_' + testconfig['f2f_prefix'] + 'and_' + testconfig['nidaq_prefix'] \
                          + trial_suffix + '_diff_' + str(testconfig['cam_dir']) + \
                        'cam' + '.png')
            else:
                fig.savefig(testconfig['dir_list'][0] + testconfig['f2f_prefix'] + '/' \
                            + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + testconfig['date'] + '/' + \
                            bias_config \
                            + '_' + testconfig['f2f_prefix'] + 'and_' + testconfig['nidaq_prefix'] + \
                            trial_suffix + '_diff_' + str(testconfig['cam_dir']) + 
                            'cam' + '.png')
                
        ## f2f vs nidaq filt
        fig, axes = set_plot_var('', no_of_trials)
        if no_of_trials == 1:
            title = testconfig['nidaq_prefix']  + ' and ' +  testconfig['f2f_prefix'] + \
                ' ' + testconfig['cam_dir'] + \
                ' ' + bias_config + trial_suffix + ' compare peaks'
        else:
            title = testconfig['nidaq_prefix']  + ' and ' +  testconfig['f2f_prefix'] + \
                ' ' +  testconfig['cam_dir'] + \
                ' ' + bias_config + trial_suffix + ' compare peaks'
        compare_peaks(lat_data_cam1.lat_nidaq_filt, lat_data_cam1.lat_f2f,
                                  shape, color, alpha, labels[0], axes, no_of_trials,\
                                  latency_threshold, numFrames, title, 0)
            
        if not debug_flag:    
            if no_of_trials == 1:    
                fig.savefig(testconfig['dir_list'][0] + testconfig['f2f_prefix'] + '/' \
                        + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + testconfig['date'] +'/' + \
                        bias_config  + \
                        '_' + testconfig['f2f_prefix'] + 'and_' + testconfig['nidaq_prefix'] \
                          + trial_suffix + '_compare_peaks_' + str(testconfig['cam_dir']) + \
                        'cam' + '.png')
            else:
                fig.savefig(testconfig['dir_list'][0] + testconfig['f2f_prefix'] + '/' \
                            + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + testconfig['date'] + '/' + \
                            bias_config \
                            + '_' + testconfig['f2f_prefix'] + 'and_' + testconfig['nidaq_prefix'] + \
                            trial_suffix + '_compare_peaks_' + str(testconfig['cam_dir']) + 
                            'cam' + '.png')  
    
    if lat_metric.isqueue:
        
        ## queue vs nidaq
        fig, axes = set_plot_var('', no_of_trials)
        if no_of_trials == 1:
            title = testconfig['nidaq_prefix']  + ' and ' +  testconfig['queue_prefix'] + \
                ' ' + testconfig['cam_dir'] + \
                ' ' + bias_config + trial_suffix + ' skipped Frames'
        else:
            title = testconfig['nidaq_prefix']  + ' and ' +  testconfig['queue_prefix'] + \
                ' ' +  testconfig['cam_dir'] + \
                ' ' + bias_config + trial_suffix + ' skipped Frames'
        # compare_peaks(lat_data_cam1.lat_nidaq, lat_data_cam1.lat_queue,
        #                           shape, color, alpha, labels[0], axes, no_of_trials,\
        #                           latency_threshold, numFrames, title, 0)
        # compare_peaks(lat_data_cam2.lat_nidaq, lat_data_cam2.lat_queue,
        #                           shape, color, alpha, labels[0], axes, no_of_trials,\
        #                           latency_threshold, numFrames, title, 1)    
            
        if not debug_flag:    
            if no_of_trials == 1:    
                fig.savefig(testconfig['dir_list'][0] + testconfig['queue_prefix'] + '/' \
                        + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + testconfig['date'] +'/' + \
                        bias_config  + \
                        '_' + testconfig['queue_prefix'] + 'and_' + testconfig['nidaq_prefix'] \
                          + trial_suffix + '_compare_peaks_' + str(testconfig['cam_dir']) + \
                        'cam' + '.png')
            else:
                fig.savefig(testconfig['dir_list'][0] + testconfig['queue_prefix'] + '/' \
                            + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + testconfig['date'] + '/' + \
                            bias_config \
                            + '_' + testconfig['queue_prefix'] + 'and_' + testconfig['nidaq_prefix'] + \
                            trial_suffix + '_compare_peaks_' + str(testconfig['cam_dir']) + 
                            'cam' + '.png')  
                            
        
        
def meanNumberSpikes(lat_data, testconfig, lat_metric, cam_id, debug_flag):
    
    no_of_trials = testconfig['no_of_trials']
    numFrames = testconfig['numFrames']
    framerate = testconfig['framerate']
    filename = testconfig['filename']
    numCameras = testconfig['numCameras']

    if lat_metric.isnidaq:
        
        if len(testconfig['mean_spikes_nidaq']) < numCameras:
            testconfig['mean_spikes_nidaq'] = np.array(int(testconfig['numCameras'])*[0.0])
            testconfig['std_spikes_nidaq'] = np.array(int(testconfig['numCameras'])*[0.0])
            testconfig['spikes_per_sec_nidaq'] = np.array(int(testconfig['numCameras'])*[0.0])
        
        if len(testconfig['count_latencyspikes_nidaq']) < numCameras:    
            testconfig['count_latencyspikes_nidaq'] = np.array(int(testconfig['numCameras'])*[no_of_trials * [0]])
            testconfig['average_normspikes_nidaq'] = np.array(int(testconfig['numCameras'])*[no_of_trials * [0.0]])
     
    if lat_metric.isframetoframe:
       
        if len(testconfig['mean_spikes_f2f']) < numCameras: ## check if string is empty
            testconfig['mean_spikes_f2f'] = np.array(int(testconfig['numCameras'])*[0.0])
            testconfig['std_spikes_f2f'] = np.array(int(testconfig['numCameras'])*[0.0])
            testconfig['spikes_per_sec_f2f'] = np.array(int(testconfig['numCameras'])*[0.0])
            
        
        if len(testconfig['count_latencyspikes_f2f']) < numCameras: ## check if string is empty
            testconfig['count_latencyspikes_f2f'] = np.array(int(testconfig['numCameras'])*[no_of_trials * [0]])
            testconfig['average_normspikes_f2f'] = np.array(int(testconfig['numCameras'])*[no_of_trials * [0.0]])

    count_numSpikes(lat_data, testconfig, lat_metric, cam_id)
    stddevNumSpikes(testconfig, lat_metric, cam_id)
    mean_nsecintervals_wspikes(lat_data, testconfig, lat_metric, cam_id, framerate)
        
    if lat_metric.isnidaq:    
        testconfig['mean_spikes_nidaq'][cam_id] = sum(testconfig['average_normspikes_nidaq'][cam_id])/no_of_trials
        testconfig['spikes_per_sec_nidaq'][cam_id] = sum((testconfig['count_latencyspikes_nidaq'][cam_id]/numFrames)*framerate)/no_of_trials
        
        if not debug_flag:
            wcs.appendWriteConfigFile(filename, testconfig, 'count_latencyspikes_nidaq', testconfig['count_latencyspikes_nidaq'])
            wcs.appendWriteConfigFile(filename, testconfig, 'average_normspikes_nidaq', testconfig['average_normspikes_nidaq'])
            wcs.appendWriteConfigFile(filename, testconfig, 'mean_spikes_nidaq', testconfig['mean_spikes_nidaq'])
            wcs.appendWriteConfigFile(filename, testconfig, 'std_spikes_nidaq', testconfig['std_spikes_nidaq'])
            wcs.appendWriteConfigFile(filename, testconfig, 'spikes_per_sec_nidaq', testconfig['spikes_per_sec_nidaq'])
            wcs.appendWriteConfigFile(filename, testconfig, 'max_skippedFrames_nidaq', testconfig['max_skippedFrames_nidaq'])
            wcs.appendWriteConfigFile(filename, testconfig, 'fracIntwspikes_nidaq', testconfig['fracIntwspikes_nidaq'])
            
    
    if lat_metric.isframetoframe:
        
        testconfig['mean_spikes_f2f'][cam_id] = sum(testconfig['average_normspikes_f2f'][cam_id])/no_of_trials
        testconfig['spikes_per_sec_f2f'][cam_id] = sum((testconfig['count_latencyspikes_f2f'][cam_id]/numFrames)*framerate)/no_of_trials
        if not debug_flag:
            wcs.appendWriteConfigFile(filename, testconfig, 'mean_spikes_f2f', testconfig['mean_spikes_f2f']) 
            wcs.appendWriteConfigFile(filename, testconfig, 'std_spikes_f2f', testconfig['std_spikes_f2f'])
            wcs.appendWriteConfigFile(filename, testconfig, 'spikes_per_sec_f2f', testconfig['spikes_per_sec_f2f'])
            wcs.appendWriteConfigFile(filename, testconfig, 'fracIntwspikes_f2f', testconfig['fracIntwspikes_f2f'])
        
        
def mean_nsecintervals_wspikes(lat_data, testconfig, lat_metric, cam_id, interval_length):

    numCameras = int(testconfig['numCameras'])
    no_of_trials = int(testconfig['no_of_trials'])
    nframes = int(testconfig['numFrames'])
    latency_threshold = testconfig['latency_threshold']     
    
    if len(testconfig['fracIntwspikes_nidaq']) < numCameras:
        testconfig['fracIntwspikes_nidaq'] = np.array(numCameras*[0.0], np.float)
        
    if len(testconfig['fracIntwspikes_f2f']) < numCameras:  
        testconfig['fracIntwspikes_f2f'] = np.array(numCameras*[0.0], np.float)
        
        
    if lat_metric.isframetoframe:
        testconfig['fracIntwspikes_f2f'][cam_id] = sum(count_nsecintervals_wspikes(lat_data.lat_f2f,\
                                                   testconfig, cam_id, interval_length)) / no_of_trials
        

    if lat_metric.isnidaq:
        testconfig['fracIntwspikes_nidaq'][cam_id] = sum(count_nsecintervals_wspikes(lat_data.lat_nidaq,\
                                                   testconfig,cam_id, interval_length)) / no_of_trials
        
            
def stddevNumSpikes(testconfig, lat_metric, cam_id):    
        
    lat_nidaq_std=0.0
    lat_f2f_std=0.0
    
    if lat_metric.isnidaq:
        lat_nidaq_std = np.std(testconfig['average_normspikes_nidaq'][cam_id], axis=0)
        testconfig['std_spikes_nidaq'][cam_id] = lat_nidaq_std

    if lat_metric.isframetoframe:
        lat_f2f_std = np.std(testconfig['average_normspikes_f2f'][cam_id], axis=0)
        testconfig['std_spikes_f2f'][cam_id] = lat_f2f_std
       
    
def logging_function(testconfig, cam_id):

    if testconfig['nidaq_prefix']:
        print('Camera id ', cam_id)
        print('spike counts per trial nidaq cam ',  testconfig['count_latencyspikes_nidaq'])
        print('average normalized spikes per trial nidaq cam ',  testconfig['average_normspikes_nidaq'])
        print('mean spikes across trials - nidaq cam ', testconfig['mean_spikes_nidaq'])
        print('std spikes across trials - nidaq cam ' , testconfig['std_spikes_nidaq'])
        print('spikes per sec - nidaq cam ' , testconfig['spikes_per_sec_nidaq'])
        print('fraction of intervals with spikes - nidaq cam ', testconfig['fracIntwspikes_nidaq'])
        print('\n')
   
 
    if testconfig['f2f_prefix']:
        print('Camera id ', cam_id)
        print('spike counts per trial f2f cam ', testconfig['count_latencyspikes_f2f'])
        print('average normalized spikes per trial f2f cam ',  testconfig['average_normspikes_f2f'])
        print('mean spikes across trials - f2f cam ', testconfig['mean_spikes_f2f'])
        print('std spikes across trials - f2f cam ', testconfig['std_spikes_f2f'])
        print('spikes per sec - f2f cam ' , testconfig['spikes_per_sec_f2f'])
        print('fraction of intervals with spikes - f2f cam ', testconfig['fracIntwspikes_f2f'])
        print('\n')
        

def main():

    ## read csv files
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
                   
    ## read configuration file
    filename = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/config_files/short/jaaba_plugin_multicamera_shorttrial_run_5e468_2_6_2023.csv'
    readConfigFile(filename,Config)  

    ## Experiment related configuration
    numCameras = Config['numCameras']
    numFrames = Config['numFrames']
    no_of_trials = Config['no_of_trials']
    framerate  = Config['framerate']
    cam_suffix = Config['cam_suffix']
    latency_threshold = Config['latency_threshold']

    ## timing configurations
    nidaq_prefix = Config['nidaq_prefix']
    f2f_prefix = Config['f2f_prefix']
    queue_prefix  = Config['queue_prefix'] 
    
    isnidaq = 0
    isfram2frame = 0
    isqueue = 0 
    
    ## mode to run in BIAS
    plugin_prefix = Config['plugin_prefix']
    logging_prefix = Config['logging_prefix']
    framegrab_prefix = Config['framegrab_prefix']  
    
    isPlugin = 0
    isJaaba = 0
    islogging = 0
    isCamOnly = 0

    ## set metrics flag
    isnidaq = setFlags(nidaq_prefix)
    isframetoframe = setFlags(f2f_prefix)
    isqueue = setFlags(queue_prefix)
    latency_metric = LatencyMetric(isnidaq, isframetoframe, isqueue)
    
    ## set BIAS mode configuration flags
    islogging = setFlags(logging_prefix)
    isCamOnly = setFlags(framegrab_prefix)
    isPlugin = setFlags(plugin_prefix)
    if plugin_prefix == 'jaaba_plugin':
        isJaaba = 1

    biasConig_mode = BiasConfigMode(isCamOnly, islogging, isPlugin, isJaaba) 

    ## Debug flag
    debug = False

    analyze_data(Config, latency_metric, \
                 biasConig_mode, debug)
                    
        
if __name__ == "__main__":
    main()