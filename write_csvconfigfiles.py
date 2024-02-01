# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 16:17:14 2021

@author: 27rut
"""

import numpy as np 
import csv
import git_hash as ghs
import read_csvconfigfile as rcs
import pandas as pd


def writeConfigFile(config):
    
    filename = config['filename']
    fieldnames = list(config.keys())   
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
      
        for key, value in config.items():
        
            if type(value) == np.ndarray:
                 tmp_list = []
                 for val in value:
                     tmp_list.append(val)
                 print(tmp_list)
                 writer.writerow((key,tmp_list))
            else:
                 writer.writerow((key,value))
            
    f.close()
       
    
def appendWriteConfigFile(filename, config, key, val):
    
    # print(filename)
    # field_flag = 0
    with open(filename, 'r', newline='') as f:
        
        config_reader = csv.reader(f, delimiter=',')
        mydict = {row[0]:row[1] for row in config_reader}
        
        f.close()
        
        mydict[key] = val
        writeConfigFile(mydict) 
           

def main():

    Config = {
        
        'filename': '',
        'numCameras': 2,
        'cam_suffix': [0,1],
        'dir_len': 1,
        'dir_list': ['C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/'],
        'numFrames':100000,
        'no_of_trials': 5,
        'framerate': 400,
        'latency_threshold' : 6.0,
        'cam_dir' : 'multi',
        #'detectSpike': '0',
        'nidaq_prefix': 'nidaq',
        'f2f_prefix': '',
        'queue_prefix': '',
        'plugin_prefix': 'jaaba_plugin',
        'logging_prefix': '',
        'framegrab_prefix': 'imagegrab',
        'git_commit': '',
        'date':''
        '''count_latencyspikes_nidaq':[],
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
        'fracIntwspikes_f2f': [],'''

    }
       
    ## read csv files
    filename = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/' + \
               'config_files/short/jaaba_plugin_multicamera_shorttrial_run_'         
    Config['git_commit'] = str(ghs.get_sha()) 
    Config['date'] = str(ghs.get_current_date())
    Config['filename'] = filename + Config['git_commit'] + '_' + Config['date'] + '.csv'
    Config['git_commit'] = ',' + str(ghs.get_sha())
    writeConfigFile(Config)  
    
    ## timing configurations
    nidaq_prefix = Config['nidaq_prefix']
    f2f_prefix = Config['f2f_prefix']
    queue_prefix  = Config['queue_prefix'] 
    
    #c set metrics flag
    isnidaq = rcs.setFlags(nidaq_prefix)
    isframetoframe = rcs.setFlags(f2f_prefix)
    isqueue = rcs.setFlags(queue_prefix)
    
    latency_metric = rcs.LatencyMetric(isnidaq, isframetoframe, isqueue)
    
    # make/set experiment directory
    if latency_metric.isnidaq:
        path_dir = Config['dir_list'][0] + Config['nidaq_prefix'] \
                              + '/' + Config['cam_dir']
        
        ghs.makemydir(path_dir)                      
    
    if latency_metric.isframetoframe:
        path_dir = Config['dir_list'][0] + Config['f2f_prefix'] \
                               + '/' + Config['cam_dir']
        ghs.makemydir(path_dir)                        
                               
    if latency_metric.isqueue:
        path_dir = Config['dir_list'][0] + Config['queue_prefix'] \
                               + '/' + Config['cam_dir']     
        ghs.makemydir(path_dir) 
    
    
if __name__ == "__main__":
    main()