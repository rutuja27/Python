# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
 
frames = 500000
imagegrab_file = open('C:/Users/27rut/BIAS/misc/logging_trials/two_camera_f2f/imagegrab_f2f_0_triallong1.csv', 'r')
imagedispatch_file = open('C:/Users/27rut/BIAS/misc/logging_trials/two_camera_f2f/imagedispatch_f2f_0_triallong1.csv', 'r')
imagelogger_file = open('C:/Users/27rut/BIAS/misc/logging_trials/two_camera_f2f/imagelogger_f2f_0_triallong1.csv', 'r')

image_grab = csv.reader(imagegrab_file, delimiter=',')
image_dispatch = csv.reader(imagedispatch_file, delimiter=',')
image_logger = csv.reader(imagedispatch_file, delimiter=',')

line_count = 0
imagegrab_ts = []
imagedispatch_ts = []
imagelogger_ts = []


#for imagegrab_row , imagedispatch_row, imagelogger_row in zip(image_grab, image_dispatch, image_logger):
    #if(imagegrab_row[0] == imagedispatch_row[0]):
    
  
#        imagedispatch_ts.append(np.float(np.float(imagedispatch_row[0]) - np.float(imagegrab_row[0]))/1000)
        
        #imagelogger_ts[line_count] = np.int64(imagelogger_row[0]) - np.int64(imagegrab_row[0])
    #else:
    #    print('MisMatch at Frame : ' ,line_count)
    #print(line_count)
#        line_count += 1

zipped_data = zip(image_grab, image_dispatch)
print(len(tuple(zipped_data)))
#for i in range(frames)
    

'''for imagegrab_row , imagedispatch_row, imagelogger_row in zip(image_grab, image_dispatch, image_logger):
    
    if(line_count < 10):
        print(np.float(imagedispatch_row[0]))
        print(np.float(imagegrab_row[0]))
        print(np.float(np.float(imagedispatch_row[0]) - np.float(imagegrab_row[0]))/1000 )
    
    line_count+=1'''
    

    
fig1 = plt.plot(imagedispatch_ts)
plt.title('Execution time Difference between GrabImageCommon and ImageGrab')
plt.show()
    
imagegrab_file.close
imagedispatch_file.close
imagelogger_file.close