# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:04:56 2021

@author: 27rut
"""

import h5py
import csv
import matplotlib.pyplot as plt
import numpy as np

fn1 = "C:/Users/27rut/BIAS/build/Release/hoghof_1.h5"
fn2 = "C:/Users/27rut/BIAS/build/Release/hoghof_org.h5"



with h5py.File(fn1, "r") as f1,  h5py.File(fn2, "r") as f2:
    # List all groups
    hof_front_l1 = f1[list(f1.keys())[0]]
    hof_side_l1 = f1[list(f1.keys())[1]]
    hog_front_l1 = f1[list(f1.keys())[2]]
    hog_side_l1 = f1[list(f1.keys())[3]]

    # List all groups
    hof_front_l2 = f2[list(f2.keys())[0]]
    hof_side_l2 = f2[list(f2.keys())[1]]
    hog_front_l2 = f2[list(f2.keys())[2]]
    hog_side_l2 = f2[list(f2.keys())[3]]
    
    #plt.plot( hof_front_l1[0], hof_front_l2[2])
    

fn3 = "C:/Users/27rut/BIAS/misc/classifier_trials/scores_demo/test_Lift.h5"
with h5py.File(fn3, "r") as f3:
    demo_scores = f3['scores'][0]
    #plt.plot(demo_scores)
    print(demo_scores)

lift_class_gnd = "C:/Users/27rut/BIAS/misc/classifier_trials/classifierscr_Lift.csv"
lift_class_test = "C:/Users/27rut/BIAS/build/Release/lift_classifier.csv"

with open(lift_class_gnd, 'r', newline='') as f1, open(lift_class_test, 'r', newline='') as f2:
    lift_scores_gnd = csv.reader(f1, delimiter=',')
    lift_scores_pred = csv.reader(f2, delimiter=',')
    lift_scr_gnd =  [np.float(row[1]) for idx, row in enumerate(lift_scores_gnd)]
    lift_scr_pred =  [np.float(row[1]) for idx, row in enumerate(lift_scores_pred)]
    print(np.array(lift_scr_gnd))
    print(np.array(lift_scr_pred))
    plt.plot(np.array(lift_scr_pred), np.array(lift_scr_gnd))
   
    