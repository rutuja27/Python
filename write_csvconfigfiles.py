# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 16:17:14 2021

@author: 27rut
"""

import numpy as np 
import csv


def writeConfigFile(filename, config):
    
    fieldnames = list(config.keys())   
    with open(filename, 'w', newline='') as f:
      writer = csv.writer(f)
      
      for key, value in config.items():
        
        if type(value) is list:
            tmp_list = []
            for val in value:
                tmp_list.append(val)
            print(tmp_list)
            writer.writerow((key,value))
        else:
            writer.writerow((key,value))
    
def main():
    
    ## read csv files
    Config = {
        'numCameras': 2,
        'cam_suffix': [0,1], 
        'dir_len': 1,
        'dir_list': ['C:/Users/27rut/BIAS/misc/imagegrab_day_trials/two_camera_lat/'],
        'numFrames': 500000,
        'no_of_trials': 1,
        'framerate': 500,
        'latency_threshold' : 5.0,
        'cam_dir' : '',
        'nidaq_prefix': 'nidaq',
        'f2f_prefix': 'f2f',
        'queue_prefix': 'queue',
        'plugin_prefix': '',
        'logging_prefix': '',
        'framegrab_prefix': 'framegrab'
            
    }
                   
    filename = 'C:/Users/27rut/BIAS/scripts/python/config_files/cameragrab_multicamera_run.csv'    
    writeConfigFile(filename,Config)  
    
    
if __name__ == "__main__":
    main()