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
        sorted_diff = np.sort(diff)[::-1]
        diff_ind = np.argsort(diff)[::-1][:]
        max_diff = max(diff)
        print(sorted_diff[1:10])
        print([diff_ind[1:10]])


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
                  pred_score_filename, numFrames, isofflineScores, behavior_names):

    numbehs = len(behavior_names)
    print(numbehs)
    print(behavior_names)

    gnd_scores = np.zeros((numbehs, numFrames), dtype=np.float)
    bias_pred_scores = np.zeros((numbehs ,numFrames) ,dtype=np.float)
    bias_pred_scores_view = np.zeros((numbehs ,numFrames) ,dtype=np.int)
    bias_scores_ts = np.zeros((numbehs, numFrames), dtype=np.int64)

    offline_pred_scores = np.zeros((numbehs,numFrames), dtype=np.float)

    for i in range(0, numbehs):

        ## predicted from matlab
        gndtruth_scores_struct = loadmat(gndtruth_score_file_path + behavior_names[i] + '.mat')
        gnd_scores[i][:] = np.array(gndtruth_scores_struct['allScores']['scores'][0][0][0][0][0][:])

        ## predicted from bias jaaba
        bias_pred_file = pred_score_file_path + pred_score_filename + '.csv';
        ut.read_score(bias_pred_file, bias_pred_scores[i], numFrames, i+3)
        ut.read_score(bias_pred_file, bias_pred_scores_view[i], numFrames, 8)

        ## predicted scores from offline jaaba
        if(isofflineScores):
            pred_offline_file = pred_score_offline_file_path + '/scores_offline.csv';
            ut.read_score(pred_offline_file, offline_pred_scores[i], numFrames, i+3);

    plotScore(gnd_scores, bias_pred_scores, offline_pred_scores, behavior_names, isofflineScores)
    plot_diff_scores(gnd_scores, bias_pred_scores, offline_pred_scores, behavior_names, isofflineScores)
    match_scores(gnd_scores, bias_pred_scores, bias_pred_scores_view, behavior_names, isofflineScores)

def main():

    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if(len(sys.argv) < 8):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-path to groundtruth score file\n' +
              '-path to predicted score file\n' +
              '-path to predicted score file offline\n'
              '-predicted  score file name\n' +
              '-number of frames for which score is predicted\n'
              '-is offline scores\n'
              '-behavior names'
              )

    else:
        gndtruth_score_file_path = sys.argv[1]
        pred_score_file_path = sys.argv[2]
        pred_score_offline_file_path = sys.argv[3]
        pred_score_filename = sys.argv[4]
        numFrames = np.int(sys.argv[5])
        isofflineScores = np.int(sys.argv[6])
        behavior_names = sys.argv[7]

        behavior_names = [ 'scores_' + x for x in behavior_names.split(',') ]
        print(behavior_names)
        loadScoreData(gndtruth_score_file_path, pred_score_file_path, pred_score_offline_file_path,
                    pred_score_filename, numFrames, isofflineScores, behavior_names)
    plt.show()

if __name__ == "__main__":
    main()