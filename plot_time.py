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

 
def readcsvFile(filename, arr, arr1, inter_latency):
  
    data_grab = csv.reader(filename, delimiter=',')
    prev=0; 
    prev1=0;   
    count=0;
    for idx,row in enumerate(data_grab):
        if(not inter_latency):
            arr.append((np.float(row[0]))) 
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
    print(count)        

def plot_data(arr, inter_latency, shape, color):
    
    sz = len(arr)
    if inter_latency:
        plt.plot(arr[0:sz-2],'.',color=color, marker=shape)
    else:
        plt.plot(arr[0:],'.',color=color, marker=shape)
   

def main():

    inter_latency = 0
    no_of_frames = 100000
    cam = ['0', '1']
    no_of_cam = 1
     
    

    dir_list = ['C:/Users/27rut/BIAS/build/Release/']
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
    color = ['r', 'b']


    count = 0
    for frame_dir in dir_list:
       
       for i in range(1,2):
           
           cam0_list = []
           cam1_list = []
           try:
               if(no_of_cam == 2):
                   print(frame_dir + 'imagegrab_latency' + cam[0] + '_trial'+ str(i)  + '.csv', 'r+')
                   cam0_handle = open(frame_dir + 'imagegrab_f2f' + cam[0] + '_trial'+ str(i)  + '.csv', 'r+');
                   cam1_handle = open(frame_dir + 'imagegrab_f2f' + cam[1] + '_trial'+ str(i)  + '.csv', 'r+');
               else:
                   print(frame_dir + 'imagegrab_f2f' + cam[0] + '_trial'+ str(i)  + '.csv', 'r+')
                   cam0_handle = open(frame_dir + 'imagegrab_f2f' + cam[0] + '_trial'+ str(i)  + '.csv', 'r+');
                   
            
           except IOError:
               print (' File could not be open. Check file location',i)
               return -1;
           if(no_of_cam==2):
               readcsvFile(cam0_handle, cam0_list, inter_latency)
               readcsvFile(cam1_handle, cam1_list, inter_latency)
               plot_data(cam0_list, inter_latency, shapes[i-1], color[0])
               plot_data(cam1_list, inter_latency, shapes[i-1], color[1])
           else:
               readcsvFile(cam0_handle, cam0_list, cam1_list, inter_latency)
               plot_data(cam0_list ,inter_latency, shapes[i-1], color[0])
               #plot_data(cam1_list,inter_latency, shapes[i-1], color[1])
               count += len(cam0_list)
          
           
          
    #plt.plot(5*np.ones((no_of_frames)))
    plt.xlabel('Frames')
    plt.ylabel('Milliseconds')
    plt.title('Single Camera Trigger-Signal Slot Frame Process Latency')
    labels = ['Cam 0']
    plt.legend(labels, fontsize=7)
    plt.show() 
    cam0_handle.close()
    
    #   
    #
    
    #fig.savefig('C:/Users/27rut/BIAS/misc/signal_slot_day_trials/figs/signalslot_singlecam__camtrig_process_latency.png')


if __name__ == "__main__":
    main()