# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:51:32 2020

@author: patilr
"""


import csv
import numpy as np
import matplotlib
#matplotlib.use('Tkagg')
#matplotlib.interactive(True)
import matplotlib.pyplot as plt

 
def readcsvFile(filename, arr, inter_latency):
  
    data_grab = csv.reader(filename, delimiter=',')
    prev=0;   
    for idx,row in enumerate(data_grab):
        if(not inter_latency):
            arr.append(np.float(row[1])/1000)
        else: 
            if(idx == 0):
              prev =  np.float(row[1])/1000
            else:
                arr.append((np.float(row[1])/1000) - prev)
            prev =  (np.float(row[1])/1000)      
        

def plot_data(arr, inter_latency, shape, color):
    
    sz = len(arr)
    if inter_latency:
        plt.plot(arr[0:sz-2],'.',color=color, marker=shape)
    else:
        plt.plot(arr[0:],'.',color=color, marker=shape)
   

def main():

    inter_latency = 1
    no_of_frames = 100000
    cam = ['0', '1']
    no_of_cam = 1
    
    dir_list =['C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera/21_12_2020/',
               'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera/22_12_2020/',
               'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/two_camera/23_12_2020/']
   
    cam0_list = []
    cam1_list = []

    fig = plt.figure();
    shapes= ['o', 'x', '^', 'P', '*']
    color = ['r', 'b']

    for frame_dir in dir_list:
       
       for i in range(1,6):
           
           cam0_list = []
           cam1_list = []
           try:
               if(no_of_cam == 2):
                   cam0_handle = open(frame_dir + 'jaaba_f2f' + cam[0] + '_trial'+ str(i)  + '.csv', 'r+');
                   cam1_handle = open(frame_dir + 'jaaba_f2f' + cam[1] + '_trial'+ str(i)  + '.csv', 'r+');
               else:
                   cam0_handle = open(frame_dir + 'jaaba_f2f' + cam[0] + '_trial'+ str(i)  + '.csv', 'r+');
                   
            
           except IOError:
               print (' File could not be open. Check file location',i)
               return -1;
           if(no_of_cam==2):
               readcsvFile(cam0_handle, cam0_list, inter_latency)
               readcsvFile(cam1_handle, cam1_list, inter_latency)
               plot_data(cam0_list, inter_latency, shapes[i-1], color[0])
               plot_data(cam1_list, inter_latency, shapes[i-1], color[1])
           else:
               readcsvFile(cam0_handle, cam0_list, inter_latency)
               plot_data(cam0_list, inter_latency, shapes[i-1], color[0])
              
          
           
        
    plt.plot(5*np.ones((no_of_frames)))
    plt.xlabel('Frames')
    plt.ylabel('Milliseconds')
    plt.title('Jaaba Plugin - 2 Cameras')
    labels = ['Cam 0']
    plt.legend(labels, fontsize=7)
    plt.show()     
    #
    
    
    #fig.savefig('C:/Users/27rut/BIAS/misc/signal_slot_day_trials/figs/_twocameras.png')


if __name__ == "__main__":
    main()