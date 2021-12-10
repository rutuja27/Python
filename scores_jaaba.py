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

#3 this does not match the demo classifier because if read funtion diff betwn opencv and matlab
## gndtruth data
scores = loadmat('C:/Users/27rut/BIAS/misc/classifier_trials/gnd_truth/scores_Liftm134w.mat')  
gnd_scores = np.array(scores['allScores']['scores'][0][0])
#plt.plot(gnd_scores[0][0])


#mulclass = h5py.File('C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/multiclassifier.mat','r+')
#class_Lift = mulclass['Lift']
#print(class_Lift['dir'])
#lift_mat = h5py.File('C:/Users/27rut/BIAS/BIASJAABA/src/plugin/jaaba_plugin/json_files/classifier_Lift.mat','r+')
#plt.plot(class_Lift['dim'],lift_mat['dim'])


# # predicted from jaaba demo code
# filename = 'C:/Users/27rut/BIAS/build/Release/test_Lift.h5';
# with h5py.File(filename, "r") as f:
#     # List all groups
#     a_group_key = list(f.keys())[0]

#     # Get the data
#     data = list(f[a_group_key])
# data = data[0][0:2497]

demo_file = 'C:/Users/27rut/BIAS/build/Release/lift_classifier.csv';
demo_handle = open(demo_file, 'r+');
demo_pred = csv.reader(demo_handle, delimiter=',')
demo_pred_scores = []
for idx,row in enumerate(demo_pred):
    demo_pred_scores.append(float(row[1]))
demo_handle.close()
    
demo_pred_scores = np.array(demo_pred_scores) 


## predicted from bias jaaba
bias_pred_file = 'C:/Users/27rut/BIAS/build/Release/classifierscr.csv';
bias_pred_handle = open(bias_pred_file, 'r+');
bias_pred = csv.reader(bias_pred_handle, delimiter=',')
bias_pred_scores = []
for idx,row in enumerate(bias_pred):
    bias_pred_scores.append(float(row[1]))
bias_pred_handle.close()
    
bias_pred_scores = np.array(bias_pred_scores) 
# pred_scores = np.reshape(pred_scores, (2497,1))   
print(len(bias_pred_scores)) 
print(len(demo_pred_scores))

print(demo_pred_scores)
print(bias_pred_scores)
plt.plot(demo_pred_scores,alpha=0.2)
plt.plot(bias_pred_scores, alpha=0.2)
plt.show()    