# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:11:24 2021

@author: 27rut
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import csv


## gndtruth data
scores = loadmat('C:/Users/27rut/BIAS/misc/classifier_trials/gnd_truth/scores_Handopenm134w.mat')  
gnd_scores = np.array(scores['allScores']['scores'][0][0])
print(len(gnd_scores[0][0]))
plt.plot(gnd_scores[0][0][1:500])
#plt.show()


## predicted from jaaba
pred_file = 'C:/Users/27rut/BIAS/misc/classifier_trials/classifierscr.csv';
pred_handle = open(pred_file, 'r+');
pred = csv.reader(pred_handle, delimiter=',')
pred_scores = []
for idx,row in enumerate(pred):
    pred_scores.append(float(row[1]))
    
pred_scores = np.array(pred_scores) 
pred_scores = np.reshape(pred_scores, (2497,1))    
print(pred_scores)
plt.plot(pred_scores[1:500])
plt.show()    