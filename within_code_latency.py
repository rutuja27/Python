# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
 

imagegrab_file = open('C:/build-Release/Release/grabImageCommon_1.csv', 'r')
imagedispatch_file = open('C:/build-Release/Release/imagegrab_0.csv', 'r')

image_grab = csv.reader(imagegrab_file, delimiter=',')
image_dispatch = csv.reader(imagedispatch_file, delimiter=',')
line_count = 0
imagegrab_ts = np.array([0],dtype = np.int64)
imagedispatch_ts = np.array([0] , dtype=np.int64)
output = np.array([],dtype=np.ulonglong)

for imagegrab_row , imagedispatch_row in zip(image_grab, image_dispatch):
    if(imagegrab_row[0] == imagedispatch_row[0]):
        imagegrab_ts = np.append(imagegrab_ts, np.int64(imagegrab_row[1]))
        imagedispatch_ts = np.append(imagedispatch_ts, np.int64(imagedispatch_row[1]))
    else:
        print('MisMatch at Frame : ' ,line_count)
    '''print(line_count)'''
    line_count += 1

print(imagegrab_ts.size)
latency = (imagedispatch_ts- imagegrab_ts)   
latency = latency[1:imagegrab_ts.size]     
fig1 = plt.plot(latency)
plt.title('Execution time Difference between GrabImageCommon and ImageGrab')
plt.show()
    
imagegrab_file.close
imagedispatch_file.close