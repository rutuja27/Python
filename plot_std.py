# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 17:36:20 2021

@author: 27rut
"""
import numpy as np 
import csv
import read_csvconfigfile as rcs
import matplotlib.pyplot as plt

def set_plot_var():
     
     fig, axes = plt.subplots(1, 1, figsize=(12,10))
     ax = axes.get_gridspec() 
     plt.xlabel('Days', fontsize=12)
     plt.ylabel('Spikes per sec', fontsize=12)
     plt.ylim((0 , 0.150))
     plt.title('Error bar plot of Spikes/sec BIAS jaaba plugin ', fontsize=15)
     
     return fig, axes
 
def plot_errorbar(Config, numCameras, filepath, fileList, alpha)    :
    
    shape= ['+', '*', 'x', '.', '^', 'o']
    color = ['r', 'b','g', 'm', 'c', 'k']
    fig, axes = set_plot_var()
    
    for cam_id in range(0, numCameras):
    
        std_arr = np.array(len(fileList) * [0.0])
        mean_arr = np.array(len(fileList) * [0.0])
        
        for i in range (0,len(fileList)):
        
            filename = filepath + fileList[i] + '.csv'      
            rcs.readConfigFile(filename, Config)         
            mean_arr[i] = float(Config['mean_spikes_f2f'][cam_id]) * Config['framerate']
            std_arr[i] = float(Config['std_spikes_f2f'][cam_id]) * Config['framerate']
        
        
        plt.errorbar(x=np.arange(0,len(fileList)),y=mean_arr, yerr=std_arr,  marker=shape[cam_id], color=color[cam_id],alpha=alpha)

    plt.xticks(np.arange(0,len(fileList)))
    axes.set_xticklabels(fileList)
    plt.legend(['Short Trial', 'Long Trial']) 
    
    
def plot_queue(Config, filepath, fileList, alpha):
    
    shape= ['+', '*', 'x', '.', '^', 'o']
    color = ['r', 'b','g', 'm', 'c', 'k']
            
    ## plotting variables
    filename = filepath + fileList[0] + '.csv'      
    rcs.readConfigFile(filename, Config)
    queue_prefix = Config['queue_prefix']
    no_of_trials = Config['no_of_trials']
    fig, axes = rcs.set_plot_var(queue_prefix, no_of_trials)
   
    for i in range (0,len(fileList)):
        
        filename = filepath + fileList[i] + '.csv'      
        rcs.readConfigFile(filename, Config)
        numCameras = Config['numCameras']
        numFrames = Config['numFrames']
        no_of_trials = Config['no_of_trials']
        framerate  = Config['framerate'] 
        latency_threshold = Config['latency_threshold']
        cam_suffix = Config['cam_suffix']
        queue_prefix  = Config['queue_prefix']
        plugin_prefix = Config['plugin_prefix']
              
        # timing configurations
        nidaq_prefix = Config['nidaq_prefix']
        f2f_prefix = Config['f2f_prefix']
        queue_prefix  = Config['queue_prefix'] 
        
        isnidaq = rcs.setFlags(nidaq_prefix)
        isframetoframe = rcs.setFlags(f2f_prefix)
        isqueue = rcs.setFlags(queue_prefix)
        latency_metric = rcs.LatencyMetric(isnidaq, isframetoframe, isqueue)
        
        if queue_prefix:
              
            title = Config['queue_prefix'] + ' length'
            
            latency_data_cam1 = rcs.LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]))
        
            
            if Config['numCameras'] == 2:
                latency_data_cam2 = rcs.LatencyData(np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                                np.array(no_of_trials*[numFrames * [0.0]]))
                    
                    
            rcs.readData(Config, latency_metric, latency_data_cam1, latency_data_cam2)   
             
            
            rcs.plot_queue_length(latency_data_cam1, latency_data_cam2, shape, color, alpha, \
                                  axes, no_of_trials, \
                                  numCameras, numFrames,\
                                  title,plugin_prefix)
           


def main():
    
     
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]    
    
    Config_long = {
        
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
    
    Config_short = {
        
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
    
    filepath = dir_path + trial_type + '/' +  configfile_prefix + '_' + cam_dir  + 'camera' + '_' + trial_type + 'trial_run_'
    
    fileList = ['b8633_11_8_2021']
    
    #fileList = ['41e45_10_25_2021','f312d_10_24_2021' , 'f312d_10_22_2021', 'f312d_10_21_2021']
    
    
    if cam_dir == 'multi' and configfile_prefix != 'jaaba_plugin':
        numCameras = 2
    else:
        numCameras = 1
    
    #plot_errorbar(Config_short, numCameras, filepath, fileList, 0.3)
    
    plot_queue(Config_short, filepath, fileList, 0.5)
    
    trial_type = 'long'
    
    filepath = dir_path + trial_type + '/' +  configfile_prefix + '_' + cam_dir  + 'camera' + '_' + trial_type + 'trial_run_'
    
    #plot_errorbar(Config_long, numCameras, filepath, fileList, 1)
    
    plot_queue(Config_long, filepath, fileList, 1)
    

    
    
    
    #fig.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/errorplot_spikes_per_sec_f2f_trials_400fps_jaaba_day_trials.png')
              
    plt.show()     
            
        
        
    
if __name__ == "__main__":
    main()