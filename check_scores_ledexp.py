import h5py
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import csv
import itertools
import pandas as pd
import utils as ut
import sys

def check_beh_starts_online(score_file_dir, score_filename, numFrames, numbehs, num_seqFrames):


    print('Number of behaviors is ', numbehs)
    print('Number of frames in a led sequence', num_seqFrames) ## number of frames is single led sequence

    num_ledOnFrames = np.int(num_seqFrames / 2)
    print('Number of on frames', num_ledOnFrames)

    num_ledRepeat = num_seqFrames  ## number of frames after which led sequence repeats
    print('Number of frames after which led sequence will repeat', num_ledRepeat)

    number_of_correct_starts = []
    number_of_wrong_starts = []

    biasjaaba_scores = np.zeros((numbehs, numFrames), dtype=np.double)
    filename = score_file_dir + score_filename + '.csv'
    print('Reading bias jaaba score file', filename)

    for beh_id in range(0,numbehs):

        ut.read_score(filename , biasjaaba_scores[beh_id], numFrames, 0, beh_id + 3)


    for beh_id in range(0,numbehs):
        frm_id=0
        count_correct_starts = 0
        count_wrong_starts = 0
        while(frm_id < numFrames):
            if(biasjaaba_scores[beh_id][frm_id] > 0):
                positive_start_index = frm_id
                #print('Positive start index', positive_start_index)
                count_correct_onframes = 0
                for pos_id in range(positive_start_index, positive_start_index + num_ledOnFrames):
                    if(biasjaaba_scores[beh_id][pos_id] > 0):
                        count_correct_onframes = count_correct_onframes + 1
                        frm_id = frm_id + 1
                        continue
                    else:
                        #print('Wrong on index', pos_id, count_correct_onframes)
                        frm_id = frm_id + 1
                        count_wrong_starts = count_wrong_starts + 1

                        break
                #print(count_correct_onframes)
                if(count_correct_onframes == num_ledOnFrames):
                    count_correct_starts = count_correct_starts + 1
            else:
                frm_id = frm_id + 1

        print('Number of correct led on starts for ', beh_id,  'is', count_correct_starts)
        print('Number of wrong led on starts for ', beh_id, 'is', count_wrong_starts)



def check_beh_starts(score_file_dir, beh_names_arr, num_seqFrames):

    numbehs = len(beh_names_arr) ## total number of leds
    print('Number of behaviors is ', numbehs)
    print('Number of frames in a led sequence', num_seqFrames) ## number of frames is single led sequence

    num_ledOnFrames = np.int(num_seqFrames / 2)
    print('Number of on frames', num_ledOnFrames)

    num_ledRepeat = num_seqFrames * numbehs ## number of frames after which led sequence repeats
    print('Number of frames after which led eequence will repeat', num_ledRepeat)

    score_file_names = ['scores_' + x + '.mat' for x in beh_names_arr]
    print('Score files to read ', score_file_names)

    allScores = []
    number_of_correct_starts = []
    number_of_wrong_starts = []

    ## read score mat files
    for beh_id in range(0,numbehs):
        jaaba_scores_struct = loadmat(score_file_dir + score_file_names[beh_id])
        allScores.append(np.array(jaaba_scores_struct['allScores']['scores'][0][0][0][0][0][:]))

    for beh_id in range(0,numbehs):
        numFrames = len(allScores[beh_id])
        positive_start_index = np.where(allScores[beh_id] > 0)[0][0]

        ## check if the first led sequqence is a complete sequence to count correct repetitions
        fst_off_frame = 0
        if(allScores[beh_id][positive_start_index + (num_ledOnFrames-1)] < 0):
            ## get next positive sequence
            for pos_frm in range(positive_start_index, num_ledRepeat):
                if(allScores[beh_id][pos_frm] < 0):
                    fst_off_frame = pos_frm
                    break
            print('First off frame ', fst_off_frame)
            if(allScores[beh_id][num_ledRepeat - (fst_off_frame+1)] > 0): ## adding one to index because zero indexed
                positive_start_index = ((num_ledRepeat-1) - (fst_off_frame+1))

        count_correct_starts = 0
        count_wrong_starts = 0
        print('First on index for beh', beh_names_arr[beh_id], 'is', positive_start_index)
        for frm_id in range(positive_start_index, numFrames, num_ledRepeat):
            if(frm_id > 0 and frm_id < numFrames):
                if(allScores[beh_id][frm_id] > 0 ## check if frm is first on frame for this bout
                        and  allScores[beh_id][frm_id-1] < 0): ## check if frm-1 led is off
                    count_correct_starts = count_correct_starts  + 1
                else:
                    print('Led is wrong for ', beh_names_arr[beh_id], ' at index ', frm_id)
                    print(allScores[beh_id][frm_id], ' ', allScores[beh_id][frm_id-1])
                    count_wrong_starts = count_wrong_starts  + 1
            else:
                count_correct_starts = 1
                if (allScores[beh_id][frm_id] > 0  ## check if frm is first on frame for this bout
                        and allScores[beh_id][frm_id - 1] < 0):  ## check if frm-1 led is off
                    count_correct_starts = count_correct_starts + 1
                else:
                    count_wrong_starts = count_wrong_starts + 1
            number_of_correct_starts.append(count_correct_starts)
            number_of_wrong_starts.append(number_of_wrong_starts)


        print('Number of positive starts for ', beh_names_arr[beh_id], 'is', count_correct_starts)
        print('Number of wrong starts for ', beh_names_arr[beh_id], 'is', count_wrong_starts)
        print('\n')




def main():
    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if (len(sys.argv) < 8):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-score files directory\n' +
              '-online jaaba score file name\n' +
              '-beh names\n' +
              '-framerate\n' +
              '-time per sequence\n'  + ## time for which a single led in on and off
              '-if matlab or not\n'
              )

    else:
        score_file_dir = sys.argv[1]
        online_scr_filename = sys.argv[2]
        beh_names = sys.argv[3]
        framerate = np.float(sys.argv[4])
        time_per_seq = np.float(sys.argv[5])
        ismatlab = np.int(sys.argv[6])
        numFrames = np.int(sys.argv[7])

        beh_names_lst = beh_names.split(',')
        num_seqFrames = np.int(((time_per_seq)) * framerate)
        numbehs = len(beh_names_lst)

        if ismatlab:
            check_beh_starts(score_file_dir, beh_names_lst, num_seqFrames) ## this is the function to check matlab scores
        else:
            check_beh_starts_online(score_file_dir, online_scr_filename, numFrames, numbehs, num_seqFrames)


if __name__ == "__main__":
    main()