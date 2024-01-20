import numpy as np
import sys
from scipy.io import loadmat
import utils as ut
import math

def matchScore_perframe(gndtruth_scorepath, biasjaaba_scorepath, biasjaaba_scorefile_prefix,
                        biasjaaba_scorepath_suffix, behavior_name, frameNumber, numClasssifierparams,
                        numViews):

    scr_gndtruth_struct = loadmat(gndtruth_scorepath + '/scores_' + behavior_name  + '.mat')
    scr_gndtruth = np.array(scr_gndtruth_struct['allScores']['scores'][0][0][0][0][0][:])

    scr_cumulation_matlab_strct = loadmat(gndtruth_scorepath + '/scores_' + str(frameNumber+1) + '.mat')
    scr_cumulation_matlab = scr_cumulation_matlab_strct['scores'][0]


    if(scr_gndtruth[frameNumber+1] == scr_cumulation_matlab[numClasssifierparams-1]):
        print('Matlab Computed', scr_cumulation_matlab[numClasssifierparams - 1])
        print('Matlab gndtruth score', scr_gndtruth[frameNumber + 1])
    else:
        print('Something is wrong')
        print('Matlab Computed', scr_cumulation_matlab[numClasssifierparams - 1])
        print('Matlab gndtruth score', scr_gndtruth[frameNumber + 1])
        return


    scr_cumulation_biasjaaba = np.zeros((numViews,numClasssifierparams), dtype=np.float)
    scr_cumulation_biasjaaba_joint = np.zeros(numClasssifierparams, dtype=np.float)

    for view_id in range(0,numViews):

       scr_cumulation_biasjaaba[view_id] = np.array(numClasssifierparams * [0.0])
       scr_biasjaaba_filename = biasjaaba_scorepath +  '/' + biasjaaba_scorefile_prefix + '_' \
                             + str(frameNumber) + '_' + biasjaaba_scorepath_suffix[view_id] + '.csv'
       ut.read_score(scr_biasjaaba_filename, scr_cumulation_biasjaaba[view_id], numClasssifierparams, 6)
       scr_cumulation_biasjaaba_joint += scr_cumulation_biasjaaba[view_id]

    for cls_id in range(0,numClasssifierparams):

        if(abs(scr_cumulation_biasjaaba_joint[cls_id] - scr_cumulation_matlab[cls_id]) > 0.05):

            print('Not matching cls id: , Matlab: , Cuda: ',
                  cls_id, round(scr_cumulation_matlab[cls_id],4),
                  round(scr_cumulation_biasjaaba_joint[cls_id],4))
        else:
            continue



def main():


    print('Number of input arguments ', len(sys.argv))
    print('Argument list:' , sys.argv)

    if(len(sys.argv) < 7):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-path to gnd truth score matlab file\n' +
              '-path to score file from biasjaaba\n' +
              '-prefix ºfor biasjaaba score data\n' +
              '-suffix for biasjaaba score data\n' +
              '-behavior_name\n' +
              '-frameNumber\n' +
              '-number of classifier params'
              )
    else:
        gndtruth_scrpath = sys.argv[1]
        biasjaaba_scrpath = sys.argv[2]
        biasjaaba_scrfile_prefix = sys.argv[3]
        biasjaaba_scrpath_suffix = sys.argv[4]
        behavior_name = sys.argv[5]
        frameNumber = np.int(sys.argv[6])
        numClsparams = np.int(sys.argv[7])

        biasjaaba_scrpath_suffix_arr = biasjaaba_scrpath_suffix.split((','))
        print('Number of views', biasjaaba_scrpath_suffix_arr)
        numViews = len(biasjaaba_scrpath_suffix_arr)

        matchScore_perframe(gndtruth_scrpath, biasjaaba_scrpath, biasjaaba_scrfile_prefix,
                        biasjaaba_scrpath_suffix_arr, behavior_name, frameNumber, numClsparams,
                        numViews)


if __name__ == '__main__':
    main()
