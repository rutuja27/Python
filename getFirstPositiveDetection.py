import sys

import numpy as np
import utils as ut
from scipy.io import loadmat


def getFirstPositiveScore(arr, numFrames):

    for frm in range(0,numFrames):
        if(arr[frm] > 0):
            print(arr[frm])
            return frm
    print('No Positive frm')
    return 0


def getFirstFrameDetection(gndtruth_score_file_path, pred_score_file_path, score_filename,
                           numFrames, numbehs, isgndtruth):

   # refactor behavior_names in future put this in arguments.
   behavior_names = ['Lift', 'Handopen', 'Grab', 'Supinate', 'Chew', 'Atmouth']

   gnd_scores = np.zeros((numbehs, numFrames), dtype=np.double)
   bias_pred_scores = np.zeros((numbehs, numFrames), dtype=np.double)

   if isgndtruth:

        for i in range(0, numbehs):
            ## predicted from matlab
            gndtruth_scores_struct = loadmat(gndtruth_score_file_path + 'scores_' + behavior_names[i] + '.mat')
            gnd_scores[i][:] = np.array(gndtruth_scores_struct['allScores']['scores'][0][0][0][0][0][:])

            fst_positive_frm = getFirstPositiveScore(gnd_scores[i][:], numFrames)
            print('First positive frame for ', behavior_names[i], fst_positive_frm)

   else:
        bias_pred_file = pred_score_file_path + score_filename + '.csv'
        print(bias_pred_file)
        for beh_id in range(0,numbehs):
            ut.read_score(bias_pred_file, bias_pred_scores[beh_id], numFrames, 0, beh_id+3)
            #print(bias_pred_scores[beh_id])
            fst_positive_frm = getFirstPositiveScore(bias_pred_scores[beh_id], numFrames)
            print('First positive frame for ', behavior_names[beh_id], fst_positive_frm)


def main():

    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if(len(sys.argv) < 7):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-path to groundtruth score file\n' +
              '-path to predicted score file\n' +
              '-score file name\n' +
              '-number of frames for which score is predicted\n'
              '-number of behaviors\n'
              '-is groundtruth'
              )

    else:
        gndtruth_score_file_path = sys.argv[1]
        pred_score_file_path = sys.argv[2]
        score_filename = sys.argv[3]
        numFrames = np.int(sys.argv[4])
        numbehs = np.int(sys.argv[5])
        isgndtruth = np.int(sys.argv[6])

        getFirstFrameDetection(gndtruth_score_file_path, pred_score_file_path, score_filename,
                     numFrames, numbehs, isgndtruth)

if __name__ == "__main__":
    main()