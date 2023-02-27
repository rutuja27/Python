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
import itertools
import pandas as pd

def match_scores(arr1, arr2):

    count=0
    idx=0
    cnt_mismatch=0
    for (scr1, scr2) in zip(arr1,arr2):
        if round(scr1,3) == round(scr2,3):
            count +=1
        else:
            if(round(scr1,3)==0 or round(scr2,3) ==0):
                continue
            else:
                cnt_mismatch += 1
                print(scr1,scr2)
        idx+=1
    print("Number of matching scores" ,count, cnt_mismatch)

def readScore_ts(arr1):

    prev=0
    scr_lat = []
    for idx, scr_ts1 in enumerate(arr1):
        if(idx == 0):
            prev = scr_ts1
        else:
            scr_lat.append( abs(scr_ts1 - prev)/1000)
            prev = scr_ts1
    plt.plot(scr_lat, '.', )
    plt.show()

def main():

    #3 this does not match the demo classifier because if read funtion diff betwn opencv and matlab
    ## gndtruth data
    scores = loadmat('C:/Users/27rut/BIAS/misc/classifier_trials/gnd_truth/scores_Liftm134w.mat')
    gnd_scores = np.array(scores['allScores']['scores'][0][0])
    print(gnd_scores[0][0])
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

    demo_file = 'C:/Users/27rut/BIAS/misc/classifier_trials/classifier_scores_new/lift_classifier.csv';
    demo_handle = open(demo_file, 'r+');
    demo_pred = csv.reader(demo_handle, delimiter=',')
    demo_pred_scores = []
    for idx,row in enumerate(demo_pred):
        demo_pred_scores.append(float(row[1]))
    demo_handle.close()


    ## predicted from bias jaaba
    bias_pred_file = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/c1a39_2_21_2023/classifier_trial2.csv';

    bias_pred = pd.read_csv(bias_pred_file)
    bias_pred_ts = bias_pred['Score ts']

    bias_pred_scores = []
    bias_pred_handle = open(bias_pred_file, 'r+');
    bias_pred_scrhd = csv.reader(bias_pred_handle, delimiter=',')
    for idx,row in enumerate(bias_pred_scrhd):
        if(idx==0):
           continue
        bias_pred_scores.append(float(row[3]))
    bias_pred_handle.close()

    match_scores(bias_pred_scores, demo_pred_scores)
    #readScore_ts(bias_pred_ts[140:-1])

    #plt.plot(demo_pred_scores[:],color='b', alpha=0.3)
    #plt.plot(bias_pred_scores[:-1], color='r', alpha=0.3)
    plt.plot(demo_pred_scores[:], bias_pred_scores[:],'.')

    plt.title('Lift Classifier Prediction')
    plt.xlabel('JAABA Offline Classifier Prediction Score')
    plt.ylabel('BIAS JAABA Classifier Prediction Score')
    #plt.legend(['JAABA Offline Classifier Prediction', 'BIAS JAABA Classifier Prediction'], fontsize=8)
    #plt.savefig('C:/Users/27rut/OneDrive/Pictures/cvml 2022/lift_classifier_scores.pdf')
    #plt.savefig('C:/Users/27rut/OneDrive/Pictures/cvml 2022/correlation_scores.jpg')
    plt.show()

if __name__ == "__main__":
    main()