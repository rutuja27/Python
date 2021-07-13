# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 15:11:24 2021

@author: 27rut
"""
import h5py
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import csv


## gndtruth data
scores = loadmat('C:/Users/27rut/BIAS/misc/classifier_trials/gnd_truth/scores_Atmouthm134w.mat')  
gnd_scores = np.array(scores['allScores']['scores'][0][0])
print(len(gnd_scores[0][0]))
plt.plot(gnd_scores[0][0])
#plt.show()


#mulclass = h5py.File('C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/multiclassifier.mat','r+')
#class_Lift = mulclass['Lift']
#print(class_Lift['dir'])
#lift_mat = h5py.File('C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/classifier_Lift.mat','r+')
#plt.plot(class_Lift['dim'],lift_mat['dim'])


## predicted from jaaba
filename = 'C:/Users/27rut/BIAS/build/Release/test_Lift.h5';
with h5py.File(filename, "r") as f:
    # List all groups
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
data = data[0][0:2497]
print(data)

## predicted from biasjaaba
pred_file = 'C:/Users/27rut/BIAS/build/Release/classifierscr.csv';
pred_handle = open(pred_file, 'r+');
pred = csv.reader(pred_handle, delimiter=',')
pred_scores = []
for idx,row in enumerate(pred):
    pred_scores.append(float(row[1]))
    
pred_scores = np.array(pred_scores) 
#pred_scores = np.reshape(pred_scores, (2498,1))    
print(pred_scores)
#plt.plot(data)
plt.plot( pred_scores)
plt.show()    