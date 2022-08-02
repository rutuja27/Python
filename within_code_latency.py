# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import csv
import numpy as np
import matplotlib.pyplot as plt

 
frames = 100000

                      
imagegrab_file =  open('C:/Users/27rut/BIAS/misc/imagegrab_day_trials/two_camera_lat/f2f/single/24d1a_5_5_2022/imagegrab_f2fcam0_short_trial1.csv')
imagedispatch_file = open('C:/Users/27rut/BIAS/misc/imagegrab_day_trials/two_camera_lat/f2f/single/24d1a_5_5_2022/imagedispatch_f2fcam0_short_trial1.csv')

image_grab = csv.reader(imagegrab_file, delimiter=',')
image_dispatch = csv.reader(imagedispatch_file, delimiter=',')


line_count = 0
imagegrab_ts = []
imagedispatch_ts = []
imagelogger_ts = []

for imagegrab, imagedispatch in zip(image_grab,image_dispatch):
     imagedispatch_ts.append(np.float(np.float(imagedispatch[0])-np.float(imagegrab[0]))/1000)


    
fig1 = plt.plot(imagedispatch_ts)
plt.title('Execution time Difference between ImageDispatch and ImageGrab')
plt.show()
    
imagegrab_file.close()
imagedispatch_file.close()
