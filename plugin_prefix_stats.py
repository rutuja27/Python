# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:18:29 2021

@author: 27rut
"""

import numpy as np
import read_csvconfigfile as rcs

def match_skipped_frames(testconfig, lat_metric, lat_grab_data_cam1, lat_grab_data_cam2, \
                         lat_plugin_data_cam1, lat_plugin_data_cam2, \
                         biasConig_mode):
    
    no_of_trials = testconfig['no_of_trials']
    
    arr_grab_cam1 = lat_grab_data_cam1.lat_queue
    arr_grab_cam2 = lat_grab_data_cam2.lat_queue
    
    arr_plugin_cam1 = lat_plugin_data_cam1.lat_queue
    arr_plugin_cam2 = lat_plugin_data_cam2.lat_queue
    
    print(arr_plugin_cam2)
    
    for i in range(0, no_of_trials):
        
        if(not np.array_equal(arr_grab_cam1[i], arr_plugin_cam1[i])):
            print('skipped frames do not match cam1 ')
            
            
        if(not np.array_equal(arr_grab_cam2[i], arr_plugin_cam2[i])):
            print('skipped frames do not match cam2 ')
            


def main():

    Config = {
        
        'filename':'',
        'numCameras': 0,
        'cam_suffix':[], 
        'dir_len': 0,
        'dir_list': [],
        'numFrames': 0,
        'no_of_trials': 0,
        'framerate': 0.0,
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
    
    dir_path = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/config_files/'
    configfile_prefix = 'jaaba_plugin'
    cam_dir = 'multi'
    trial_type = 'short'
    
    flag_std = False
    
    filepath = dir_path + trial_type + '/' +  configfile_prefix + '_' + cam_dir  + 'camera' + '_' + trial_type + 'trial_run_'
    
    fileList = 'd2b00_11_19_2021'
    
    filename = filepath + fileList + '.csv'      
    rcs.readConfigFile(filename, Config)
    
    numCameras = Config['numCameras']
    numFrames = Config['numFrames']
    no_of_trials = Config['no_of_trials']
    cam_suffix = Config['cam_suffix']
    
    isPlugin = 0
    isJaaba = 0
    islogging = 0
    isCamOnly = 0
    
    isnidaq = 0
    isfram2frame = 0
    isqueue = 0 
    
    ## timing configurations
    nidaq_prefix = Config['nidaq_prefix']
    f2f_prefix = Config['f2f_prefix']
    queue_prefix  = Config['queue_prefix'] 
    
    ## set metrics flag
    isnidaq = rcs.setFlags(nidaq_prefix)
    isframetoframe = rcs.setFlags(f2f_prefix)
    isqueue = rcs.setFlags(queue_prefix)
    latency_metric = rcs.LatencyMetric(isnidaq, isframetoframe, isqueue)

    ## mode to run in BIAS
    plugin_prefix = Config['plugin_prefix']
    logging_prefix = Config['logging_prefix']
    framegrab_prefix = Config['framegrab_prefix'] 
    
    islogging = rcs.setFlags(logging_prefix)
    isCamOnly = rcs.setFlags(framegrab_prefix)
    isPlugin = rcs.setFlags(plugin_prefix)
    if plugin_prefix == 'jaaba_plugin':
        isJaaba = 1
    biasConig_mode = rcs.BiasConfigMode(isCamOnly, islogging, isPlugin, isJaaba)
    
    if plugin_prefix:
        bias_config = Config['plugin_prefix']
        
    if logging_prefix:    
        bias_config = Config['logging_prefix']
    
    if framegrab_prefix:
        bias_config = Config['framegrab_prefix'] 
 
    
    latency_imagegrab_data_cam1 = rcs.LatencyData([],[],[],[],[],[])
    latency_imagegrab_data_cam2 = rcs.LatencyData([],[],[],[],[],[])
    latency_plugin_data_cam1 = rcs.LatencyData([],[],[],[],[],[])
    latency_plugin_data_cam2 = rcs.LatencyData([],[],[],[],[],[])
    
    if framegrab_prefix:
        latency_imagegrab_data_cam1 = rcs.LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]))
        
            
        if numCameras == 2:
            latency_imagegrab_data_cam2 = rcs.LatencyData(np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]))
        rcs.readData(Config, latency_metric, latency_imagegrab_data_cam1, latency_imagegrab_data_cam2, Config['framegrab_prefix']) 
       
    if plugin_prefix:
        
        latency_plugin_data_cam1 = rcs.LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]))
        
        if numCameras == 2:
            latency_plugin_data_cam2 = rcs.LatencyData(np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]))    
      
        rcs.readData(Config, latency_metric, latency_plugin_data_cam1, latency_plugin_data_cam2, Config['plugin_prefix'])
    
    
    match_skipped_frames(Config, latency_metric, latency_imagegrab_data_cam1, latency_imagegrab_data_cam2, \
                         latency_plugin_data_cam1, latency_plugin_data_cam2, \
                         biasConig_mode)        

        
    
if __name__ == "__main__":
    main()