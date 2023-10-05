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
import sys

def match_scores(gndtruth_scores, bias_pred_scores, bias_pred_view ,beh_names, isofflineScores):

    numbehs = len(beh_names)

    for i in range (0,numbehs):
        arr1 = gndtruth_scores[i]
        arr2 = bias_pred_scores[i]
        count = 0
        idx = 0
        cnt_mismatch = 0
        cnt_skipped =0
        cnt_both_skipped =0
        for (scr1, scr2) in zip(arr1,arr2):
            if round(scr1,4) == round(scr2,4):
                count +=1
            elif( bias_pred_view[i][idx] == 1 or bias_pred_view[i][idx] == 2):
                cnt_skipped+=1
            elif(bias_pred_view[i][idx] == 0):
                cnt_both_skipped+=1
            else:
                cnt_mismatch+=1
            idx+=1
        print("Number of matching scores for " ,beh_names[i], count, cnt_skipped, cnt_both_skipped, cnt_mismatch)

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

def plotScore(gndtruth_scores, bias_pred_scores, offline_pred_scores, beh_names, isofflineScores):

    numbehs = len(gndtruth_scores)
    if(isofflineScores):
        fig, ax = plt.subplots(numbehs, 3, figsize=(15, 15))
    else:
        fig, ax = plt.subplots(numbehs, 1, figsize=(15,15))

    rows = ['{}'.format(beh) for beh in beh_names]
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.8,
                        hspace=0.4)

    for i in range(0, numbehs):

        if(isofflineScores):
            ax[i][0].plot(gndtruth_scores[i][:], bias_pred_scores[i][:], '.')
            ax[i][0].set_xlabel('Ground truth Scores')
            ax[i][0].set_ylabel('Biasjaaba pred scores')

            ax[i][1].plot(gndtruth_scores[i][:], offline_pred_scores[i][:], '.')
            ax[i][1].set_xlabel('Ground truth Scores')
            ax[i][1].set_ylabel('offline jaaba pred scores')

            ax[i][2].plot(offline_pred_scores[i][:], bias_pred_scores[i][:], '.')
            ax[i][2].set_ylabel('Biasjaaba scores')
            ax[i][2].set_xlabel('offline jaaba pred scores')
        else:
            ax[i].plot(gndtruth_scores[i][:], bias_pred_scores[i][:], '.')
            ax[i].set_xlabel('Ground truth Scores')
            ax[i].set_ylabel('Biasjaaba pred scores')

    if isofflineScores :
        for ax_id, row in zip(ax[:,0], rows):
            ax_id.set_ylabel(row, rotation=90, size='large')
    else:
        for ax_id, row in zip(ax[:], rows):
            ax_id.set_ylabel(row, rotation=90, size='large')

def plot_diff_scores(gndtruth_scores, bias_pred_scores, offline_pred_scores, beh_names, isofflineScores):

    numbehs = len(gndtruth_scores)
    if(isofflineScores):
       fig, ax = plt.subplots(numbehs, 3, figsize=(15, 15))
    else:
       fig, ax = plt.subplots(numbehs, 1, figsize=(15, 15))

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.8,
                        hspace=0.4)

    for i in range(0, numbehs):
        diff = abs(gndtruth_scores[i] - bias_pred_scores[i])
        sorted_diff = np.sort(diff)
        max_diff = max(diff)
        print(sorted_diff[::-1])


        if(isofflineScores):
            ax[i][0].plot(gndtruth_scores[i] - bias_pred_scores[i], '.')
            ax[i][0].set_ylabel(beh_names[i])
            ax[i][0].set_yticks(np.arange((max_diff) * -1, max_diff, 10.0))
            if (i == 0):
                ax[i][0].set_title('Difference between gndtruth and bias scores')

            ax[i][1].plot(gndtruth_scores[i] - offline_pred_scores[i], '.')
            ax[i][1].set_yticks(np.arange((max_diff)*-1, max_diff, 10.0))
            if (i == 0):
                ax[i][1].set_title('Difference between gndtruth and offline scores')

            ax[i][2].plot(offline_pred_scores[i] - bias_pred_scores[i], '.')
            ax[i][2].set_yticks(np.arange((max_diff)*-1, max_diff, 10.0))
            if(i==0):
                ax[i][2].set_title('Difference between offline and bias_scores ')
        else:
            ax[i].plot(gndtruth_scores[i] - bias_pred_scores[i], '.')
            ax[i].set_ylabel(beh_names[i])
            ax[i].set_yticks(np.arange((max_diff) * -1, max_diff, 10.0))

def loadScoreData(gndtruth_score_file_path, pred_score_file_path, pred_score_offline_file_path,
                  pred_score_filename, numFrames, isofflineScores):

    behavior_scores = ["scores_Lift", "scores_Handopen", "scores_Grab", "scores_Supinate", "scores_Chew",
                       "scores_Atmouth"]
    numbehs = len(behavior_scores)
    print(numbehs)
    print(behavior_scores[1])

    gnd_scores = np.zeros((numbehs, numFrames), dtype=np.double)
    bias_pred_scores = np.zeros((numbehs ,numFrames) ,dtype=np.double)
    bias_pred_scores_view = np.zeros((numbehs ,numFrames) ,dtype=np.int)
    bias_scores_ts = np.zeros((numbehs, numFrames), dtype=np.int64)

    offline_pred_scores = np.zeros((numbehs,numFrames), dtype=np.double)

    for i in range(0, numbehs):

        ## predicted from matlab
        gndtruth_scores_struct = loadmat(gndtruth_score_file_path + behavior_scores[i] + '.mat')
        gnd_scores[i][:] = np.array(gndtruth_scores_struct['allScores']['scores'][0][0][0][0][0][:])

        ## predicted from bias jaaba
        bias_pred_file = pred_score_file_path + pred_score_filename + '.csv';
        ut.read_score(bias_pred_file, bias_pred_scores[i], 0, i+3)
        ut.read_score(bias_pred_file, bias_pred_scores_view[i], 0, 10)

        ## predicted scores from offline jaaba
        if(isofflineScores):
            pred_offline_file = pred_score_offline_file_path + pred_score_filename + '.csv';
            ut.read_score(pred_offline_file, offline_pred_scores[i],0, i+3);

    plotScore(gnd_scores, bias_pred_scores, offline_pred_scores, behavior_scores, isofflineScores)
    plot_diff_scores(gnd_scores, bias_pred_scores, offline_pred_scores, behavior_scores, isofflineScores)
    match_scores(gnd_scores, bias_pred_scores, bias_pred_scores_view, behavior_scores, isofflineScores)

    plt.show()

def main():

    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if(len(sys.argv) < 7):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-path to groundtruth score file\n' +
              '-path to predicted score file\n' +
              '-path to predicted score file offline\n'
              '-predicted  score file name\n' +
              '-number of frames for which score is predicted\n'
              '-is offline scores'
              )

    else:
        gndtruth_score_file_path = sys.argv[1]
        pred_score_file_path = sys.argv[2]
        pred_score_offline_file_path = sys.argv[3]
        pred_score_filename = sys.argv[4]
        numFrames = np.int(sys.argv[5])
        isofflineScores = np.int(sys.argv[6])
        loadScoreData(gndtruth_score_file_path, pred_score_file_path, pred_score_offline_file_path,
                    pred_score_filename, numFrames, isofflineScores)

    '''scr_diff = (abs(gnd_scores[0][0][0][:-1] - bias_pred_scores[:]))
    sort_diff = np.sort(scr_diff)
    reverse_sort_diff = sort_diff[::-1]  ##numpy does not sort in descending order
    sort_indexes = np.argsort(scr_diff)
    reverse_sort_indexes = sort_indexes[::-1];

    print('Score', gnd_scores[0][0][0][1113], bias_pred_scores[1113])
    print('Score diff',reverse_sort_diff[0:10])
    print('Scores indexes',reverse_sort_indexes[0:10])

    match_scores(bias_pred_scores[:], gnd_scores[0][0][0][:-1],bias_pred_view)'''

    '''demo_file = 'C:/Users/27rut/BIAS/misc/classifier_trials/classifier_scores/lift_classifier.csv';
    demo_pred_scores = np.array(numFrames * [0.0], dtype=np.double)
    ut.read_score(demo_file, demo_pred_scores, 1, 1)'''

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
    '''plt.plot(bias_pred_scores[1:], gnd_scores[0][0][0][1:-1], '.')

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Handopen Classifier Prediction',fontsize=20)
    plt.xlabel('Matlab JAABA Classifier Prediction Score', fontsize=18)
    plt.ylabel('BIAS JAABA Classifier Prediction Score', fontsize=18)'''
    #plt.legend(['JAABA Offline Classifier Prediction', 'BIAS JAABA Classifier Prediction'], fontsize=8)
    #plt.savefig('C:/Users/27rut/OneDrive/Pictures/cvml 2022/lift_classifier_scores.pdf')
    #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/correlation_scores.jpg')
    plt.show()

if __name__ == "__main__":
    main()