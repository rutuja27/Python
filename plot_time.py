# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 12:51:32 2020

@author: patilr
"""


import csv
import numpy as np
import matplotlib.pyplot as plt
 
inter_latency = 1;
plot_file1 = open('C:/build-Release/Release/imagegrab_0.csv', 'r')
plot_file2 = open('C:/build-Release/Release/imagedispatch_0.csv', 'r')
#plot_file3 = open('C:/Users/patilr/BIAS/misc/grasshopper/csv_data/imagegrab_grasshopper_mono8.csv','r')
#plot_file2 = open('C:/Users/patilr/BIAS/test_code/bin64/vs2015/delay.csv', 'r')

image_plot1 = csv.reader(plot_file1, delimiter=',')
image_plot2 = csv.reader(plot_file2, delimiter=',')
#image_plot3 = csv.reader(plot_file3, delimiter=',')

line_count = 0
image_ts1 = np.array([0],dtype = np.int64)
image_ts2 = np.array([0],dtype = np.int64)
#image_ts3 = np.array([0],dtype = np.int64)
output = np.array([],dtype=np.ulonglong)

for im1,im2  in zip(image_plot1,image_plot2):
    image_ts1 = np.append(image_ts1, np.int64(im1[1]))
    image_ts2 = np.append(image_ts2, np.int64(im2[1]))
#    image_ts3 = np.append(image_ts3, np.int64(im3[1]))
    line_count += 1

if inter_latency:
    diff = [j-i for i, j in zip(image_ts2[:-1], image_ts2[1:])]
    diff = np.array(diff)

print(image_ts1.size)
#plt.plot(image_ts1[10:50000])
#plt.plot(image_ts3[7100:7200],'.')
plt.plot(image_ts1)
plt.plot(diff[1:])
#plt.plot(image_ts3)
plt.xlabel('Frames')
plt.ylabel('MicroSeconds')
plt.title('Image Grab Latency')
plt.show()

plot_file1.close
plot_file2.close
#plot_file3.close
