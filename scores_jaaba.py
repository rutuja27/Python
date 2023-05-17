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
import utils as ut

def match_scores(arr1, arr2, view):

    count=0
    idx=0
    cnt_mismatch=0
    cnt_skipped=0
    for (scr1, scr2) in zip(arr1,arr2):

        if round(scr1,4) == round(scr2,4):
            count +=1
        else:
            if view[idx] == 1 or view[idx]==2:
                cnt_skipped += 1
            else:
                cnt_mismatch+=1
        idx+=1
    print("Number of matching scores" ,count,cnt_skipped, cnt_mismatch)

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

    # this does not match the demo classifier because if read funtion diff betwn opencv and matlab
    ## gndtruth data
    scores = loadmat('Y:\hantman_data\jab_experiments\STA14\STA14\\20230503\STA14_20230503_142341\scores_Handopen.mat')
    gnd_scores = np.array(scores['allScores']['scores'][0][0])
    numFrames = len(gnd_scores[0][0][0][:-1])
    print(gnd_scores[0][0][0])

    demo_file = 'C:/Users/27rut/BIAS/misc/classifier_trials/classifier_scores/lift_classifier.csv';
    demo_pred_scores = np.array(numFrames * [0.0], dtype=np.double)
    ut.read_score(demo_file,demo_pred_scores,1,1)

    ## predicted from bias jaaba
    bias_pred_file = 'Y:/hantman_data/jab_experiments/STA14/STA14/20230503/STA14_20230503_142341/classifier_score.csv';
    bias_pred_scores = np.array(numFrames*[0.0],dtype=np.double)
    bias_pred_view = np.array(numFrames*[0.0], dtype=np.double)
    ut.read_score(bias_pred_file,bias_pred_scores,0,4)
    #ut.read_score(bias_pred_file,bias_pred_view,0,10)

    scr_diff = (abs(gnd_scores[0][0][0][:-1] - bias_pred_scores[:]))
    sort_diff = np.sort(scr_diff)
    reverse_sort_diff = sort_diff[::-1]  ##numpy does not sort in descending order
    sort_indexes = np.argsort(scr_diff)
    reverse_sort_indexes = sort_indexes[::-1];

    print('Score', gnd_scores[0][0][0][1113], bias_pred_scores[1113])
    print('Score diff',reverse_sort_diff[0:10])
    print('Scores indexes',reverse_sort_indexes[0:10])

    match_scores(bias_pred_scores[:], gnd_scores[0][0][0][:-1],bias_pred_view)

    '''file = 'C:/Users/Public/Documents/National Instruments/NI-DAQ/Examples/DAQmx ANSI C/Counter/Count Digital Events/' \
           'Cnt-Buf-Cont-ExtClk/x64/Release/latency.csv'
    file_handle = open(file, 'r+');
    lat_read = csv.reader(file_handle, delimiter=',')
    delay_test_nidaq = []

    for idx,row in enumerate(lat_read):
        delay_test_nidaq.append(float(row[0]))
    file_handle.close()

    plt.plot(delay_test_nidaq[0:40], '.')
    plt.show()'''

    #plt.plot(demo_pred_scores[:-1],'.',color='b', alpha=0.3)
    #plt.plot(gnd_scores[0][0][0][1:-1], '.', color='b')
    #plt.plot(bias_pred_scores[1:], '.', color='r', alpha=0.3)
    #plt.figure(figsize=(10, 10))
    #plt.plot(bias_pred_scores[1:], demo_pred_scores[:-1],'.')
    plt.plot(bias_pred_scores[1:], gnd_scores[0][0][0][1:-1], '.')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Lift Classifier Prediction',fontsize=20)
    plt.xlabel('Matlab JAABA Classifier Prediction Score', fontsize=18)
    plt.ylabel('BIAS JAABA Classifier Prediction Score', fontsize=18)
    #plt.legend(['JAABA Offline Classifier Prediction', 'BIAS JAABA Classifier Prediction'], fontsize=8)
    #plt.savefig('C:/Users/27rut/OneDrive/Pictures/cvml 2022/lift_classifier_scores.pdf')
    #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/correlation_scores.jpg')
    plt.show()

if __name__ == "__main__":
    main()