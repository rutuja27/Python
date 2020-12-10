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

 
def readcsvFile(filename, arr):
    data_grab = csv.reader(filename, delimiter=',')
    for idx,row in enumerate(data_grab):
        arr[idx] = np.float(row[1])/1000
        

def plot_data(arr, inter_latency):
    
    sz = arr.size;
    if inter_latency:
        lat_val=np.diff(arr)
        plt.plot(lat_val[0:sz-2],'.')
    else:
        plt.plot(arr[0:],'.')
   

def main():

    inter_latency = 1;
    no_of_frames = 100000
    
    dir_list =['C:/Users/27rut/BIAS/misc/signal_slot_day_trials/two_camera/10_12_2020/']
               #'C:/Users/27rut/BIAS/misc/imagegrab_day_trials/7_12_2020/',
               #'C:/Users/27rut/BIAS/misc/imagegrab_day_trials/6_12_2020/']
    lat_val = np.zeros((no_of_frames),dtype = np.float)


    fig = plt.figure();

    for frame_dir in dir_list:
     
       for i in range(1,4):
            
           lat_val.fill(0.0)
           try:
               file_handle = open(frame_dir + 'signal_slot_f2f1_trial'+ str(i)  + '.csv', 'r+');
           except IOError:
               print (' File could not be open. Check file location',i)
               return -1;
          
           readcsvFile(file_handle, lat_val)
           plot_data(lat_val, inter_latency)
    file_handle.close 
    plt.plot(5*np.ones((no_of_frames)))
    plt.xlabel('Frames')
    plt.ylabel('Milliseconds')
    plt.title('ImageGrab f2f Latency')
    plt.show()     
    #fig.savefig('C:/Users/27rut/BIAS/misc/signal_slot_day_trials/figs/imagegrab_f2f_latencies.pdf')


if __name__ == "__main__":
    main()