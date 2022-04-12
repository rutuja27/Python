# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:22:25 2022

@author: 27rut
"""

import numpy as np 
import csv
import matplotlib.pyplot as plt
import re

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
    
    def __init__(self, flag1, flag2,flag3):
        # initializing instance variable
        self.isnidaq=flag1;
        self.isframetoframe=flag2
        self.isqueue=flag3

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
            #print((np.float(row[1])-np.float(row[0])) * 0.02)                                        
        else:
            arr_lat[idx] = (((np.float(row[1]) - arr_cam[idx] ) * 0.02))
            #print((np.float(row[1]) - arr_cam[idx] ) * 0.02)
                         
    fhandle.close()

    
def readcsvFile_int(filename, arr, cam_id, plugin_prefix):
    
    #if cam_id == 1 and plugin_prefix == 'jaaba_plugin':
    #    return   
    
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
    
        ## read latency readings from nidaq
        if lat_metric.isnidaq:
        
            filename = path_dir + testconfig['nidaq_prefix'] \
            + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
            testconfig['date'] + '/' + biasmode_prefix + '_' + \
            testconfig['nidaq_prefix'] +  'cam' + testconfig['cam_suffix'][cam_id]\
            + trial_suffix + str(i+1) + '.csv'
            
            readcsvFile_nidaq(filename, lat_dat.lat_nidaq[i], lat_dat.lat_camtrig[i], \
                          cam_id, plugin_prefix)
        ## read queue size 
        if lat_metric.isqueue:
            
            filename = path_dir + testconfig['queue_prefix'] \
            + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
            testconfig['date'] + '/'+ biasmode_prefix + '_' + \
            testconfig['queue_prefix'] +  'cam' + \
            testconfig['cam_suffix'][cam_id] + trial_suffix + str(i+1) + '.csv'  
    
            readcsvFile_int(filename, lat_dat.lat_queue[i], cam_id, plugin_prefix)
                     
                    
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
            print(rows[idx])
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
                    
def correlate_skips(lat_data1, lat_data2, testconfig):
    
    no_of_trials = testconfig['no_of_trials']
    numFrames = testconfig['numFrames']
    total_skips_misses=0
    
    fig, ax = plt.subplots()
    for i in range(0, no_of_trials): 
        print('skip correlation score for trial no', i)
        total_skips_misses = np.sum(np.subtract(lat_data1.lat_queue[i],lat_data2.lat_queue))
    print(total_skips_misses)    
    plt.show()                    

def main():
    
    config_file  = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/config_files/short/jaaba_plugin_multicamera_shorttrial_run_7881e_4_12_2022.csv'
     
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
        
    latency_metric = LatencyMetric(0, 0, 1)
    
    biasConfig_mode = BiasConfigMode(isCamOnly, islogging, isPlugin, isJaaba) 
    
    if plugin_prefix:
        bias_config = Config['plugin_prefix']
        
    print(bias_config)    
    cam_id = 0
    readLatency_data(latency_data_jaaba_cam1, Config, latency_metric,\
                      bias_config, cam_id)
        
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
        
    copy_camtrig(latency_data_imagegrab_cam1, latency_data_imagegrab_cam2)
    cam_id = 1
    readLatency_data(latency_data_imagegrab_cam2, Config, latency_metric,\
                      bias_config, cam_id)    
    
    
    correlate_skips(latency_data_imagegrab_cam2, latency_data_jaaba_cam2, Config)
    correlate_skips(latency_data_imagegrab_cam1, latency_data_jaaba_cam1, Config)

if __name__ == "__main__":
    
    main()         