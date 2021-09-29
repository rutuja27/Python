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
 
    def __init__(self, arr1, arr2, arr3, arr4):
        # initializing instance variable
        self.lat_nidaq = arr1
        self.lat_f2f = arr2
        self.lat_queue = arr3
        self.lat_camtrig = arr4
    
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
                     cam_id, ax_handle):
    
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
    
def count_numSpikes(arr, testconfig, lat_metric, cam_id , ax_handle):
    

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
            [count_peaks, loc_peaks] = maxPeak(arr[trial_id],\
            latency_threshold, 10, ax_handle)
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
def maxPeak(latency_arr, height, distance, ax_handle):
    
    peaks = find_peaks(latency_arr, height=height, distance=distance)
    latency_arr = np.array(latency_arr)

    ## indices of peaks 
    loc_of_peaks = [0] * len(latency_arr)
    for aa in peaks[0]:
        loc_of_peaks[aa] = latency_arr[aa]
    #ax_handle.plot(loc_of_peaks, '.')
    
    return [len(peaks[0]), loc_of_peaks]
    

def copy_camtrig(lat_data1, lat_data2):
    
    lat_data2.lat_camtrig = lat_data1.lat_camtrig
    

def plot_data(arr, shape, color, alpha, ax_handle):
    
    sz = len(arr)      
    ax_handle.plot(arr,'.',color=color, marker=shape, alpha=alpha, ms=8)


def meanNumberSpikes(lat_data, testconfig, lat_metric, cam_id, ax_handle):
    
    no_of_trials = testconfig['no_of_trials']
    numFrames = testconfig['numFrames']
    framerate = testconfig['framerate']
   

    if lat_metric.isnidaq:
        
        if  len(testconfig['mean_spikes_nidaq']) == 0:
            testconfig['mean_spikes_nidaq'] = np.array(int(testconfig['numCameras'])*[0.0])
            testconfig['spikes_per_sec_nidaq'] = np.array(int(testconfig['numCameras'])*[0.0])
        
        if len(testconfig['count_latencyspikes_nidaq']) == 0:    
            testconfig['count_latencyspikes_nidaq'] = np.array(int(testconfig['numCameras'])*[no_of_trials * [0]])
        
        count_numSpikes(lat_data.lat_nidaq  , testconfig, lat_metric, \
                            cam_id, ax_handle)
           
        testconfig['mean_spikes_nidaq'][cam_id] = sum(testconfig['count_latencyspikes_nidaq'][cam_id])/no_of_trials
        
        testconfig['spikes_per_sec_nidaq'][cam_id] = sum((testconfig['count_latencyspikes_nidaq'][cam_id]/numFrames)*framerate)/no_of_trials


    if lat_metric.isframetoframe:
        
        if len(testconfig['mean_spikes_f2f']) == 0:
             testconfig['mean_spikes_f2f'] = np.array(int(testconfig['numCameras'])*[0.0])
             testconfig['spikes_per_sec_f2f'] = np.array(int(testconfig['numCameras'])*[0.0])
        
        if len(testconfig['count_latencyspikes_f2f']) == 0:
            testconfig['count_latencyspikes_f2f'] = np.array(int(testconfig['numCameras'])*[no_of_trials * [0]])
            
        count_numSpikes(lat_data.lat_f2f, testconfig, lat_metric, \
                            cam_id, ax_handle)
            
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
    std_nidaq=0.0
    std_f2f=0.0
    
    lat_nidaq = np.array(lat_data.lat_nidaq)
    lat_f2f = np.array(lat_data.lat_f2f)
    
    lat_nidaq_std = np.std(lat_nidaq, axis=0)
    lat_f2f_std = np.std(lat_f2f, axis=0)
    
    testconfig['std_spikes_nidaq'] = lat_nidaq_std
    testconfig['std_spikes_f2f'] = lat_f2f_std
        

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
    filename = 'C:/Users/27rut/BIAS/scripts/python/config_files/cameragrab_multicamera_run.csv'    
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

    fig, axes = plt.subplots(1, 1, figsize=(10,8))
    shapes= ['+', '*', 'x', '.', '^', 'o']
    color = ['r', 'b','g', 'm', 'c', 'k']
    alpha = [0.2,0.2,0.4, 0.4, 0.6, 0.8]
    axes.set_ylim([0,30])
    axes.set_yticks(np.arange(0,30,1))

    latency_data_cam1 = LatencyData(np.array(no_of_trials*[numFrames * [None]]), np.array(no_of_trials*[numFrames * [None]]),
                                    np.array(no_of_trials*[numFrames * [None]]), np.array(no_of_trials*[numFrames * [None]]))
    
    
    if Config['numCameras'] == 2:
        latency_data_cam2 = LatencyData(np.array(no_of_trials*[numFrames * [None]]), np.array(no_of_trials*[numFrames * [None]]),
                                       np.array(no_of_trials*[numFrames * [None]]), np.array(no_of_trials*[numFrames * [None]]))
           
    if isCamOnly:
        
        cam_id = int(cam_suffix[0])
        
        readLatency_data(latency_data_cam1, Config, latency_metric, framegrab_prefix, cam_id, axes)
       
        meanNumberSpikes(latency_data_cam1, Config, latency_metric, cam_id, axes)
        
        mean_nsecintervals_wspikes(latency_data_cam1, Config, latency_metric, cam_id, framerate)
                 
        #print('Trial data nidaq cam1 ', latency_data_cam1.lat_nidaq)
        #print('Trial data f2f cam 1', latency_data_cam1.lat_f2f)
        #print('Trial data queue cam 1 ', latency_data_cam1.lat_queue)

        #plot_data(latency_data_cam1.lat_nidaq[4], shapes[0], color[0], alpha[0], axes)
        #plot_data(latency_data_cam1.lat_f2f[4], shapes[1], color[1], alpha[1], axes)
        #plot_data(latency_data_cam1.lat_nidaq[1], shapes[1], color[1], alpha[1], axes)
        #plot_data(latency_data_cam1.lat_nidaq[2], shapes[2], color[2], alpha[2], axes)
        #plot_data(latency_data_cam1.lat_nidaq[3], shapes[3], color[3], alpha[3], axes)
        
        print('spike counts per trial nidaq cam 1', Config['count_latencyspikes_nidaq'])
        print('avg spikes - nidaq cam 1', Config['mean_spikes_nidaq'])
        print('avg spikes per sec per trial - nidaq cam 1', Config['spikes_per_sec_nidaq'])
        print('fraction of intervals with spikes - nidaq cam 1', Config['fracIntwspikes_nidaq'])
        
        print('\n')
        print('\n')
        
        print(Config['count_latencyspikes_f2f'])
        print('avg count of spikes per trial - f2f cam 1', Config['mean_spikes_f2f'])
        print('avg spikes per sec per trial - f2f cam 1', Config['spikes_per_sec_f2f'])
        print('fraction of intervals with spikes - f2f cam 1', Config['fracIntwspikes_f2f'])
        
        print('\n')
        print('\n')
        
        if Config['numCameras'] == 2:
            copy_camtrig(latency_data_cam1, latency_data_cam2)
            
            cam_id = int(cam_suffix[1])
            
            readLatency_data(latency_data_cam2, Config, latency_metric, framegrab_prefix, cam_id, axes)
            
            meanNumberSpikes(latency_data_cam2, Config, latency_metric, cam_id, axes)
            
            mean_nsecintervals_wspikes(latency_data_cam2, Config, latency_metric, cam_id, framerate)
            
            #print('Trial data nidaq cam 2', latency_data_cam2.lat_nidaq)
            #print('Trial data f2f cam 2', latency_data_cam2.lat_f2f)
            #print('Trial data queue cam 2', latency_data_cam2.lat_queue)
        
            
            print('spike counts per trial nidaq cam 2' , Config['count_latencyspikes_nidaq'])
            print('avg spikes - nidaq cam 2', Config['mean_spikes_nidaq'])
            print('avg spikes per sec per trial - nidaq cam 2', Config['spikes_per_sec_nidaq'])
            print('avg fraction of intervals with spikes cam 2 - nidaq',Config['fracIntwspikes_nidaq']) 
            
            print('\n')
            print('\n')
            
            print('spike counts per trial f2f cam 2', Config['count_latencyspikes_f2f'])
            print('avg count of spikes per trial - f2f cam 2', Config['mean_spikes_f2f'])
            print('avg spikes per sec per trial - f2f cam 2', Config['spikes_per_sec_f2f'])
    
                                    
            print('avg fraction of intervals with spikes cam 2 - f2f', Config['fracIntwspikes_f2f'])
                      
        #plot_data(latency_data_cam2.lat_nidaq[0], shapes[0], color[0], alpha[0], axes)
        #plot_data(latency_data_cam2.lat_f2f[0], shapes[1], color[1], alpha[1], axes)
        #plot_data(latency_data_cam1.lat_f2f[2], shapes[2], color[2], alpha[2], axes)
        #plot_data(latency_data_cam1.lat_f2f[3], shapes[3], color[3], alpha[3], axes)
                   
        
        plt.plot(np.array(5*np.ones(numFrames)))
        
if __name__ == "__main__":
    main()