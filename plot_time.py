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

 
def readcsvFile(fhandle, arr, inter_latency):
  
    data_grab = csv.reader(fhandle, delimiter=',')
    prev=0; 
    prev1=0;   
    count=0;
    for idx,row in enumerate(data_grab):
        if(not inter_latency):
            arr.append((np.float(row[0])/1000))
            if(np.float((row[0])) > 5):
                count+=1;
        else: 
            if(idx == 0):
              prev =  np.float(row[1])
              #prev1 = np.float(row[2])   
            else:
                if(np.float((np.float(row[1]) - prev)) > 5):
                    count+=1;
                arr.append((np.float(row[1]) - prev)/1000)
                #arr1.append((np.float(row[2]) - prev1)/1000)
            prev =  (np.float(row[1]))
            #prev1 = (np.float(row[2]))
    fhandle.close()
    print(count)        

def plot_data(arr, inter_latency, shape, color):
    
    sz = len(arr)
    if inter_latency:
        plt.plot(arr[0:sz-2],'.',color=color, marker=shape,markersize=5)
    else:
        plt.plot(arr[0:],'.',color=color, marker=shape,markersize=5)
   

def main():

    inter_latency = 0
    no_of_frames = 100000
    cam = ['0', '1']
    no_of_cam = 2

    dir_list = ['C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/b0821_9_27_2022/']
               #['C:/Users/27rut/BIAS/misc/spinnaker_toy_example/bias_demo_example/cam2sys_lat/single_camera/']
               #
               #['C:/Users/27rut/BIAS/build/Release/']
               #['C:/Users/27rut/spinnaker_examples/bin64/vs2015/']
               #['C:/Users/27rut/BIAS/build/Release/']
               #['C:/Users/27rut/spinnaker_examples/bin64/vs2015/misc/camtrig_camgrab_latency_external/']
               #['C:/Users/27rut/spinnaker_examples/bin64/vs2015/']
               #['C:/Users/27rut/spinnaker_examples/bin64/vs2015/misc/camtrig_camgrab_latency/']
               #['C:/Users/27rut/BIAS/misc/imagegrab_day_trials/two_camera_1000000/7_12_2020/']
               #['C:/Users/27rut/BIAS/misc/imagegrab_day_trials/two_camera_reducedmemleak/9_12_2020/']
               #['C:/Users/27rut/BIAS/misc/imagegrab_day_trials/two_camera_reducedmemleak/6_12_2020/',
               #'C:/Users/27rut/BIAS/misc/imagegrab_day_trials/two_camera_reducedmemleak/7_12_2020/',
               #'C:/Users/27rut/BIAS/misc/imagegrab_day_trials/two_camera_reducedmemleak/8_12_2020/']
             
              
   
    cam0_list = []
    cam1_list = []

    fig = plt.figure();
    shapes= ['.', 'x', '^', '*', 'P', '.', '+']
    color = ['r', 'b', 'g', 'm', 'c', 'k']


    count = 0
    for frame_dir in dir_list:
       for i in range(1,4):
           cam0_list = []
           cam1_list = []
           try:
               if(no_of_cam == 2):
                   filename0 = frame_dir + 'imagegrab_start_timecam' + cam[0] + '_short_trial'+ str(i)  + '.csv'
                   filename1 = frame_dir + 'imagegrab_start_timecam' + cam[1] + '_short_trial'+ str(i)  + '.csv'
                   cam0_handle = open(filename0, 'r+');
                   cam1_handle = open(filename1, 'r+');
               else:
                   filename = frame_dir + 'cam2sys_latency_vidread' + '_trialshort' + str(i) + '.csv.';
                   cam0_handle = open(filename, 'r+');

           except IOError:
               print (' File could not be open. Check file location',i)
               print(filename0)
               return -1;
           if(no_of_cam==2):
               readcsvFile(cam0_handle, cam0_list, inter_latency)
               readcsvFile(cam1_handle, cam1_list, inter_latency)
               plot_data(cam0_list, inter_latency, shapes[i-1%7], color[(i-1)%5])
               #plot_data(cam1_list, inter_latency, shapes[i-1%7], color[(i-1)%5])
           else:
               readcsvFile(cam0_handle, cam0_list, inter_latency)
               plot_data(cam0_list ,inter_latency, shapes[i-1%7], color[(i-1)%5])
               #plot_data(cam1_list,inter_latency, shapes[i-1], color[1])
               count += len(cam0_list)
          
           
          
    #plt.plot(5*np.ones((no_of_frames)))
    plt.xlabel('Frames')
    plt.ylabel('Milliseconds')
    plt.title('Video Read Time BIAS JAABA with file open wait')
    labels = ['Trial 1', 'Trial 2', 'Trial 3']
    plt.legend(labels, fontsize=7)
    plt.show()
    
    #   
    #
    
    #fig.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/b0821_9_27_2022/vidread_biasjaaba_wfopenwait.jpg')

if __name__ == "__main__":
    main()