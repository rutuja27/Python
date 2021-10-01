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


class LatencyData:  
 
    def __init__(self, arr1, arr2, arr3, arr4, arr5):
        # initializing instance variable
        self.lat_nidaq = arr1
        self.lat_f2f = arr2
        self.lat_queue = arr3
        self.lat_camtrig = arr4
        self.lat_nidaq_filt = arr5
                
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
        rows = [[col for col in row ] for row in config_reader]

        keys = list(config.keys())
        for idx,row in enumerate(rows):   
           
            for idy,col in enumerate(row):
      
                if(idy == 0):               
                    if row[idy] == keys[idx]:
                        continue
                    else:
                        break               
                if type(config[keys[idx]]) is str:
                    config[keys[idx]] = str(col)
                elif type(config[keys[idx]]) is list:
                    col = col.strip("[, ], ' ")
                    col = re.split(',', col)
                    config[keys[idx]] = col
                elif type(config[keys[idx]]) is float:
                    config[keys[idx]] =  float(col)
                elif type(config[keys[idx]]) is int:
                    config[keys[idx]] = int(col)
                else:
                    #print(type(config[keys[idx]]))
                    continue;

        
def readcsvFile_nidaq(filename, arr_lat, arr_cam, cam_id):
    
    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')
    
    for idx, row in enumerate(data_grab):
           
        if cam_id == 0:

            arr_cam[idx] = ((np.float(row[0]))) ## will store the count corresponding to camera trigger 
            arr_lat[idx] = (((np.float(row[1])-np.float(row[0])) * 0.02)) ## latency calculation between 
                                               ## event and camera trigger, fast clock is 50khz
                                               ## hence multiplying factor is (1/50khz- period) 0.02 to calculate latency
            #print((np.float(row[1])-np.float(row[0])) * 0.02)                                        
        else:
            arr_lat[idx] = (((np.float(row[1]) - arr_cam[idx] ) * 0.02))
            #print((np.float(row[1]) - arr_cam[idx] ) * 0.02)
                         
    fhandle.close()
    
    
def readcsvFile_f2f(filename, arr, f2f_flag):
 
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
        
    
def readcsvFile_int(filename, arr):
    
    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')
    
    for idx,row in enumerate(data_grab):
        arr[idx] = np.int(row[0])                    
    fhandle.close()


def readLatency_data(lat_dat, testconfig, lat_metric, biasmode_prefix, \
                     cam_id):
    
    no_of_trials = np.int(testconfig['no_of_trials'])
    
    if(no_of_trials == 1):
        trial_suffix = '_long'
    else:
        trial_suffix = '_trial'
    
        
    for i in range(0, no_of_trials):     
    
        ## read latency readings from nidaq 

        if lat_metric.isnidaq:
        
            filename = testconfig['dir_list'][0] + testconfig['nidaq_prefix'] + \
            '/' + testconfig['cam_dir'] + '/' + biasmode_prefix + '_' + testconfig['nidaq_prefix'] +  'cam' + \
            testconfig['cam_suffix'][cam_id] + trial_suffix + str(i+1) + '.csv'
            
            readcsvFile_nidaq(filename, lat_dat.lat_nidaq[i], lat_dat.lat_camtrig[i], \
                          cam_id)

        ##read latency readings from computer time stamps
        if lat_metric.isframetoframe:
            
            filename = testconfig['dir_list'][0] + testconfig['f2f_prefix'] \
            + '/' + testconfig['cam_dir'] + '/' + biasmode_prefix + '_' + testconfig['f2f_prefix'] +  'cam' + \
            testconfig['cam_suffix'][cam_id] + trial_suffix + str(i+1) + '.csv'
    
            readcsvFile_f2f(filename, lat_dat.lat_f2f[i], 1)
            
    
        ## read queue size 
        if lat_metric.isqueue:
            
            filename = testconfig['dir_list'][0] + testconfig['queue_prefix'] + \
            '/' + testconfig['cam_dir'] + '/'+ biasmode_prefix + '_' + testconfig['queue_prefix'] +  'cam' + \
            testconfig['cam_suffix'][cam_id] + trial_suffix + str(i+1) + '.csv'  
    
            readcsvFile_int(filename, lat_dat.lat_queue[i])
    
                
def setFlags(flag_prefix) :
    
    if flag_prefix != '':
        return 1
    else:
        return 0
    
    
def count_numSpikes(arr, testconfig, lat_metric, cam_id, arr_filt):
    
    no_of_trials = testconfig['no_of_trials']
    latency_threshold = testconfig['latency_threshold']

    if(lat_metric.isframetoframe and len(testconfig['count_latencyspikes_f2f']) != 0):
        for trial_id in range(0,no_of_trials):
            for i in range(0,len(arr[0])):
                if(arr[trial_id][i] == None):
                    continue
        
                if(arr[trial_id][i] > latency_threshold):
                    testconfig['count_latencyspikes_f2f'][cam_id][trial_id] += 1


    if(lat_metric.isnidaq and len(testconfig['count_latencyspikes_nidaq']) != 0):
        for trial_id in range(0,no_of_trials):
            if len(arr_filt) != 0:
               [count_peaks, loc_peaks] = maxPeak(arr[trial_id],\
               latency_threshold, 10, arr_filt[trial_id])
               testconfig['count_latencyspikes_nidaq'][cam_id][trial_id]  = count_peaks  
    
            
def count_nsecintervals_wspikes(arr, testconfig, cam_id, interval_length):
    
    no_of_trials = int(testconfig['no_of_trials'])
    nframes = int(testconfig['numFrames'])
    latency_threshold = testconfig['latency_threshold']
    
    count_intervalswspikes = np.array(no_of_trials*[0])
    count_intervals = 0;
   
    for trial_id in range(0, no_of_trials):
        count_intervals=0
        for i in range(0,  nframes, interval_length):     
            for j in range(0, interval_length-1): 
                if(arr[trial_id][i] == None):
                    continue
            
                if(arr[trial_id][i+j] >= latency_threshold):               
                    count_intervals += 1
                    break;
        count_intervalswspikes[trial_id] = count_intervals            
     
    numIntervals = nframes/interval_length     
    count_intervalswspikes = count_intervalswspikes/numIntervals  

    return count_intervalswspikes
    
                            
## distance - distance between peaks
## height -  definition of height of a peak        
def maxPeak(latency_arr, height, distance, latency_filt_arr):
    
    peaks = find_peaks(latency_arr, height=height, distance=distance)
    latency_arr = np.array(latency_arr)
        
    for i in range(0,len(latency_arr)):
        
        if latency_arr[i] < height:
            latency_filt_arr[i] = latency_arr[i]
        else:
            latency_filt_arr[i] = 2.2
            
    ## get filtered indexes
    for aa in peaks[0]:
        latency_filt_arr[aa] = latency_arr[aa]

    return [len(peaks[0]), peaks[0]]
    

def copy_camtrig(lat_data1, lat_data2):
    
    lat_data2.lat_camtrig = lat_data1.lat_camtrig
  
    
def set_plot_var(no_of_trials):
     
    fig, axes = plt.subplots(3, 2, figsize=(12,10))
    ax = axes[0,0].get_gridspec()
    fig.subplots_adjust(hspace=0.5)

    if no_of_trials == 5:
        axes[ax.nrows-1, ax.ncols-2] = plt.subplot2grid((3,2), (2,0), colspan = 2)
        
    plt.setp(axes, yticks = np.arange(0,20,2), ylim=[0,20])
   
    
        
    return fig, axes
  

def plot_raw_data(arr, shape, color, alpha, labels, ax_handle,\
                  no_of_trials, numFrames, title, cam_id):
    
    ax = ax_handle[0,0].get_gridspec()
    
    if cam_id == 0:
       color_id = 0
    else:
        color_id = 3 
    
    
    for ix in range(0, ax.nrows):
        idx = ix*ax.ncols
        for iy in range(0, ax.ncols):
            
            if (idx + iy) < (no_of_trials): 
            
                if cam_id == 0:
                    
                    if (idx + iy) == 0:
                        ax_handle[ix, iy].plot(arr[idx + iy], '.', 
                                           color=color[color_id] ,\
                                           marker=shape[idx + iy], \
                                           alpha=1, ms=8, label='Cam' + str(cam_id))    
                    else:
                        ax_handle[ix, iy].plot(arr[idx + iy], '.', 
                                           color=color[color_id] ,\
                                           marker=shape[idx + iy], \
                                           alpha=1, ms=8)
                else:
                    if (idx + iy) == 0:
                        ax_handle[ix, iy].plot(arr[idx + iy], '.', 
                                           color=color[color_id] ,\
                                           marker=shape[idx + iy], \
                                           alpha=1, ms=8, label='Cam' + str(cam_id)) 
                    else:
                        ax_handle[ix, iy].plot(arr[idx + iy], '.', 
                                           color=color[color_id] ,\
                                           marker=shape[idx + iy], \
                                           alpha=1, ms=8,label= '_nolegend_')
                ax_handle[ix, iy].title.set_text('Trial ' +  str(idx+ iy+ 1))
                ax_handle[ix, iy].tick_params(axis='x', labelsize=12)
                ax_handle[ix, iy].tick_params(axis='y', labelsize=12)
                ax_handle[ix, iy].plot(np.array(5*np.ones(numFrames)),\
                                       label='_nolegend_')
                
    plt.suptitle(title,fontsize=17)
    plt.setp(ax_handle[:], xlabel = 'Frames', ylabel ='Milliseconds')
    
    
def matching_peaks(lat_arr_nidaq, lat_arr_f2f, shape, color, alpha, \
                          labels, ax_handle,
                          no_of_trials, numFrames, title, cam_id):
    
    ax = ax_handle[0,0].get_gridspec()
    
    no_of_mismatches = np.array(no_of_trials*[0],np.int32)
    index_mismatches = np.array(no_of_trials*[0],np.int32)
    
    for trial_id in range(0, no_of_trials):
        for frame_id in range(1, numFrames):
            if abs(lat_arr_nidaq[trial_id][frame_id] - \
                   lat_arr_f2f[trial_id][frame_id]) > 5.0 :
                       no_of_mismatches[trial_id] += 1
                       np.append(index_mismatches[trial_id],frame_id)
                       print(frame_id)
        if no_of_mismatches[trial_id] > 0:
            print('Trial ' + str(trial_id) + ' has mismatches')
            print('Trial ' + str(trial_id) + 'mismatched indexes', \
                  index_mismatches[trial_id])
            
    for ix in range(0, ax.nrows):
        idx = ix*ax.ncols
        for iy in range(0, ax.ncols):
            if (idx + iy) < (no_of_trials): 
                ax_handle[ix,iy].plot(abs(lat_arr_nidaq[idx+iy][1:] - \
                                          lat_arr_f2f[idx+iy][1:]),\
                             '.', color=color[0] , \
                             marker=shape[idx + iy], \
                             alpha=1, label='Red')
                ax_handle[ix, iy].plot(np.array(5*np.ones(numFrames)),\
                                       label='_nolegend_')    
    
             
    plt.suptitle(title,fontsize=17)
    plt.setp(ax_handle[:], xlabel = 'Frames', ylabel ='Milliseconds')
    
        
def plot_data(lat_data_cam1, lat_data_cam2, lat_metric, testconfig):
      
    no_of_trials = testconfig['no_of_trials']
    numFrames = testconfig['numFrames'] 
    numCameras = testconfig['numCameras']
    
    shape= ['+', '*', 'x', '.', '^', 'o']
    color = ['r', 'b','g', 'm', 'c', 'k']
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]    
    labels = [['Cam 1', 'Cam 2'], ['Cam 1']] 
    
    if lat_metric.isnidaq:
        
        ## plotting variables
        fig, axes = set_plot_var(no_of_trials) 
        
        if no_of_trials == 1:
            title = testconfig['nidaq_prefix']  + ' ' +  testconfig['cam_dir'] + \
                ' ' + testconfig['framegrab_prefix'] + ' long trial'
        else:
            title = testconfig['nidaq_prefix']  + ' ' +  testconfig['cam_dir'] + \
                ' ' + testconfig['framegrab_prefix'] + ' trials'
                
        
        
        if numCameras == 2: 
            arr1 = lat_data_cam1.lat_nidaq
            arr2 = lat_data_cam2.lat_nidaq
            plot_raw_data(arr1, shape, color, alpha, labels[0], axes, no_of_trials,\
                      numFrames, title, 0)
            
            plot_raw_data(arr2, shape, color, alpha, labels[0], axes, no_of_trials,\
                      numFrames, title, 1)
                
            
        else:
            arr1 = lat_data_cam1.lat_nidaq
            plot_raw_data(arr1, shape, color, alpha, labels[1], axes, no_of_trials,\
                      numFrames, title, 0)

        if no_of_trials == 1:    
            fig.savefig(testconfig['dir_list'][0] + testconfig['nidaq_prefix'] + '/' + \
                    testconfig['cam_dir'] + testconfig['framegrab_prefix'] \
                    + '_' + testconfig['nidaq_prefix'] + '_long_trial_' + str(testconfig['cam_dir'])+ \
                       'cam' + '.png')
        else:
            fig.savefig(testconfig['dir_list'][0] + testconfig['nidaq_prefix'] + '/' + \
                    testconfig['cam_dir'] + testconfig['framegrab_prefix'] \
                    + '_' + testconfig['nidaq_prefix'] + '_trials_' + str(testconfig['cam_dir'])+ \
                       'cam' + '.png')
        
                
    if lat_metric.isframetoframe:
        
        ## plotting variables
        fig, axes = set_plot_var(no_of_trials)
        
        if no_of_trials == 1:
            title = testconfig['f2f_prefix']  + ' ' + testconfig['cam_dir'] + \
               ' ' + testconfig['framegrab_prefix'] + 'long trial'
        else:
            title = testconfig['f2f_prefix']  + ' ' + testconfig['cam_dir'] + \
                ' ' + testconfig['framegrab_prefix'] + 'trials'
             

        if numCameras == 2: 
            
            arr1 = lat_data_cam1.lat_f2f
            arr2 = lat_data_cam2.lat_f2f
        
            plot_raw_data(arr1, shape, color, alpha, labels, axes, no_of_trials,\
                      numFrames, title, 0)
            plot_raw_data(arr2, shape, color, alpha, labels, axes, no_of_trials,\
                      numFrames, title, 1)
        else:
            arr1 = lat_data_cam1.lat_f2f
            plot_raw_data(arr1, shape, color, alpha, labels, axes, no_of_trials,\
                      numFrames, title, 0)
                
        fig.legend(axes, labels, loc = 'upper right')
        if no_of_trials == 1:    
            fig.savefig(testconfig['dir_list'][0] + testconfig['f2f_prefix'] + '/' + \
                    testconfig['cam_dir'] + testconfig['framegrab_prefix'] \
                    + '_' + testconfig['f2f_prefix'] + '_long_trial_' + str(testconfig['cam_dir'])+ \
                       'cam' + '.png')
        else:
            fig.savefig(testconfig['dir_list'][0] + testconfig['f2f_prefix'] + '/' + \
                        testconfig['cam_dir'] + testconfig['framegrab_prefix'] \
                         + '_' + testconfig['f2f_prefix'] + '_trials_' + str(testconfig['cam_dir'])+ \
                        'cam' + '.png')
    
        ## diff plots 
        fig, axes = set_plot_var(no_of_trials)
        if no_of_trials == 1:
            title = testconfig['nidaq_prefix']  + ' and ' +  testconfig['f2f_prefix'] + \
                ' ' + testconfig['cam_dir'] + \
                ' ' + testconfig['framegrab_prefix'] + ' long trial diff'
        else:
            title = testconfig['nidaq_prefix']  + ' and ' +  testconfig['f2f_prefix'] + \
                ' ' +  testconfig['cam_dir'] + \
                ' ' + testconfig['framegrab_prefix'] + ' trials diff'
        matching_peaks(lat_data_cam1.lat_nidaq_filt, lat_data_cam1.lat_f2f,
                                  shape, color, alpha, labels[0], axes, no_of_trials,\
                                  numFrames, title, 0)


def meanNumberSpikes(lat_data, testconfig, lat_metric, cam_id):
    
    no_of_trials = testconfig['no_of_trials']
    numFrames = testconfig['numFrames']
    framerate = testconfig['framerate']
   
    if lat_metric.isnidaq:
        
        if  len(testconfig['mean_spikes_nidaq']) == 0:
            testconfig['mean_spikes_nidaq'] = np.array(int(testconfig['numCameras'])*[0.0])
            testconfig['spikes_per_sec_nidaq'] = np.array(int(testconfig['numCameras'])*[0.0])
        
        if len(testconfig['count_latencyspikes_nidaq']) == 0:    
            testconfig['count_latencyspikes_nidaq'] = np.array(int(testconfig['numCameras'])*[no_of_trials * [0]])
        
        print(cam_id)
        count_numSpikes(lat_data.lat_nidaq, testconfig, lat_metric, \
                            cam_id, lat_data.lat_nidaq_filt)
           
        testconfig['mean_spikes_nidaq'][cam_id] = sum(testconfig['count_latencyspikes_nidaq'][cam_id])/no_of_trials
        
        testconfig['spikes_per_sec_nidaq'][cam_id] = sum((testconfig['count_latencyspikes_nidaq'][cam_id]/numFrames)*framerate)/no_of_trials


    if lat_metric.isframetoframe:
        
        if len(testconfig['mean_spikes_f2f']) == 0:
             testconfig['mean_spikes_f2f'] = np.array(int(testconfig['numCameras'])*[0.0])
             testconfig['spikes_per_sec_f2f'] = np.array(int(testconfig['numCameras'])*[0.0])
        
        if len(testconfig['count_latencyspikes_f2f']) == 0:
            testconfig['count_latencyspikes_f2f'] = np.array(int(testconfig['numCameras'])*[no_of_trials * [0]])
            
        count_numSpikes(lat_data.lat_f2f, testconfig, lat_metric, \
                            cam_id, [])
            
        testconfig['mean_spikes_f2f'][cam_id] = sum(testconfig['count_latencyspikes_f2f'][cam_id])/no_of_trials
        
        testconfig['spikes_per_sec_f2f'][cam_id] = sum((testconfig['count_latencyspikes_f2f'][cam_id]/numFrames)*framerate)/no_of_trials
    
    
        
def mean_nsecintervals_wspikes(lat_data, testconfig, lat_metric, cam_id, interval_length):

    numCameras = int(testconfig['numCameras'])
    no_of_trials = int(testconfig['no_of_trials'])
    nframes = int(testconfig['numFrames'])
    latency_threshold = testconfig['latency_threshold']     
    
    if len(testconfig['fracIntwspikes_nidaq']) == 0:
        testconfig['fracIntwspikes_nidaq'] = np.array(numCameras*[0], np.float)
        
    if len(testconfig['fracIntwspikes_f2f']) == 0:  
        testconfig['fracIntwspikes_f2f'] = np.array(numCameras*[0], np.float)
        
        
    if lat_metric.isframetoframe:
        testconfig['fracIntwspikes_f2f'][cam_id] = sum(count_nsecintervals_wspikes(lat_data.lat_f2f,\
                                                   testconfig,cam_id, interval_length)) / no_of_trials
        

    if lat_metric.isnidaq:
        testconfig['fracIntwspikes_nidaq'][cam_id] = sum(count_nsecintervals_wspikes(lat_data.lat_nidaq,\
                                                   testconfig,cam_id, interval_length)) / no_of_trials
                                                                                    
    
def stddevNumSpikes(lat_data, testconfig):

    no_of_trials = testconfig['no_of_trials']
    
    lat_nidaq = np.array(lat_data.lat_nidaq)
    lat_f2f = np.array(lat_data.lat_f2f)
    
    lat_nidaq_std = np.std(lat_nidaq, axis=0)
    lat_f2f_std = np.std(lat_f2f, axis=0)
    
    testconfig['std_spikes_nidaq'] = lat_nidaq_std
    testconfig['std_spikes_f2f'] = lat_f2f_std
    
def logging_function(testconfig):

    if testconfig['nidaq_prefix']:
        print('spike counts per trial nidaq cam 1', testconfig['count_latencyspikes_nidaq'])
        print('avg spikes - nidaq cam 1', testconfig['mean_spikes_nidaq'])
        print('avg spikes per sec per trial - nidaq cam 1', testconfig['spikes_per_sec_nidaq'])
        print('fraction of intervals with spikes - nidaq cam 1', testconfig['fracIntwspikes_nidaq'])
        print('\n')
   
 
    if testconfig['f2f_prefix']:
        print(testconfig['count_latencyspikes_f2f'])
        print('avg count of spikes per trial - f2f cam 1', testconfig['mean_spikes_f2f'])
        print('avg spikes per sec per trial - f2f cam 1', testconfig['spikes_per_sec_f2f'])
        print('fraction of intervals with spikes - f2f cam 1', testconfig['fracIntwspikes_f2f'])
        print('\n')
        

def main():
    
    ## read csv files
    Config = {
        
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
        'count_latencyspikes_nidaq':[],
        'mean_spikes_nidaq': [],
        'std_spikes_nidaq': [],
        'spikes_per_sec_nidaq' : [],
        'fracIntwspikes_nidaq': [],
        'count_latencyspikes_f2f':[], 
        'mean_spikes_f2f': [],
        'std_spikes_f2f': [],
        'spikes_per_sec_f2f': [],
        'fracIntwspikes_f2f': []
            
    }
                   
    ## read configuration file
    filename = 'C:/Users/27rut/BIAS/scripts/python/config_files/cameragrab_singlecamera_run.csv'    
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
    if plugin_prefix == 'jaaba':
        isJaaba = 1
    biasConig_mode = BiasConfigMode(isCamOnly, islogging, isPlugin, isJaaba) 

    ## Debug flag
    debug = True

    ## data allocation arrays
    latency_data_cam1 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                    np.array(no_of_trials*[numFrames * [0.0]]), \
                                    np.array(no_of_trials*[numFrames * [0.0]]), \
                                    np.array(no_of_trials*[numFrames * [0.0]]), \
                                    np.array(no_of_trials*[numFrames * [0.0]]))
    
    
    if Config['numCameras'] == 2:
        latency_data_cam2 = LatencyData(np.array(no_of_trials*[numFrames * [0.0]]),\
                                        np.array(no_of_trials*[numFrames * [0.0]]),\
                                        np.array(no_of_trials*[numFrames * [0.0]]),\
                                        np.array(no_of_trials*[numFrames * [0.0]]),\
                                        np.array(no_of_trials*[numFrames * [0.0]]))
           
    if isCamOnly:
                
        cam_id = int(cam_suffix[0])
              
        readLatency_data(latency_data_cam1, Config, latency_metric, framegrab_prefix, cam_id)
       
        meanNumberSpikes(latency_data_cam1, Config, latency_metric, cam_id)
        
        mean_nsecintervals_wspikes(latency_data_cam1, Config, latency_metric, cam_id, framerate)
                 
        #print('Trial data nidaq cam1 ', latency_data_cam1.lat_nidaq)
        #print('Trial data f2f cam 1', latency_data_cam1.lat_f2f)
        #print('Trial data queue cam 1 ', latency_data_cam1.lat_queue)

        if debug:
            logging_function(Config)
            
        if numCameras == 1:
            plot_data(latency_data_cam1, None, latency_metric, Config)
        
        if Config['numCameras'] == 2:
            
            copy_camtrig(latency_data_cam1, latency_data_cam2)
            
            cam_id = int(cam_suffix[1])
            
            print(latency_data_cam2.lat_nidaq_filt)
            readLatency_data(latency_data_cam2, Config, latency_metric, framegrab_prefix,\
                             cam_id)
            
            meanNumberSpikes(latency_data_cam2, Config, latency_metric, cam_id)
            
            mean_nsecintervals_wspikes(latency_data_cam2, Config, latency_metric, cam_id,\
                                       framerate)
            
            #print('Trial data nidaq cam 2', latency_data_cam2.lat_nidaq)
            #print('Trial data f2f cam 2', latency_data_cam2.lat_f2f)
            #print('Trial data queue cam 2', latency_data_cam2.lat_queue)
            
            if debug:
                logging_function(Config)
        
        
            plot_data(latency_data_cam1, latency_data_cam2, latency_metric, Config)
             
            
            
        
        
if __name__ == "__main__":
    main()