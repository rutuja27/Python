import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut
import sys

def match_timestamps(filepath, file_suffix1, file_suffix2, trial_num, numFrames):

     imagegrab_file_ts0 = filepath + file_suffix1 + "_cam0_short_trial" + trial_num + ".csv"
     imagegrab_file_ts1 = filepath + file_suffix1 + "_cam1_short_trial" + trial_num + ".csv"
     jaaba_file_ts0 = filepath + file_suffix2 + "_cam0_short_trial" + trial_num + ".csv"
     jaaba_file_ts1 = filepath + file_suffix2 + "_cam1_short_trial" + trial_num + ".csv"
     print('Files to compare', imagegrab_file_ts0, jaaba_file_ts0)
     print('Files to compare', imagegrab_file_ts1, jaaba_file_ts1)

     imagegrab_ts_cam0 = np.array(numFrames * [0.0])
     imagegrab_ts_cam1 = np.array(numFrames * [0.0])
     jaaba_ts_cam0 = np.array(numFrames * [0.0])
     jaaba_ts_cam1 = np.array(numFrames * [0.0])

     ut.readcsvFile_float(imagegrab_file_ts0, imagegrab_ts_cam0)
     ut.readcsvFile_float(imagegrab_file_ts1, imagegrab_ts_cam1)
     ut.readcsvFile_float(jaaba_file_ts0, jaaba_ts_cam0)
     ut.readcsvFile_float(jaaba_file_ts1, jaaba_ts_cam1)

     print(len(jaaba_ts_cam0), len(imagegrab_ts_cam0))
     print(len(jaaba_ts_cam1), len(imagegrab_ts_cam1))
     assert(len(jaaba_ts_cam0) == len(imagegrab_ts_cam0))
     assert(len(jaaba_ts_cam1) == len(imagegrab_ts_cam1))

     frames=len(jaaba_ts_cam0)
     not_matching_indexes_cam0 = []
     not_matching_indexes_cam1 = []
     imagegrab_sync_diff = []
     print('Frames',frames)

     for i in range(0, frames):
         if(imagegrab_ts_cam0[i] != jaaba_ts_cam0[i]):
             not_matching_indexes_cam0.append(i)
             print('Val of timestamp at index cam0', i, imagegrab_ts_cam0[i], jaaba_ts_cam0[i])
         if(imagegrab_ts_cam1[i] != jaaba_ts_cam1[i]):
             not_matching_indexes_cam1.append(i)
             print('Val of timestamp at index cam1 ', i, imagegrab_ts_cam1[i], jaaba_ts_cam1[i])
         imagegrab_sync_diff.append(abs(imagegrab_ts_cam1[i] - imagegrab_ts_cam0[i]))

     print('Not matching indexes in cam 0',not_matching_indexes_cam0)
     print('Not matching indexes in cam1', not_matching_indexes_cam1)
     plt.plot(imagegrab_sync_diff,'.')
     plt.show()



def main():

    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if(len(sys.argv) < 6):
        print('Insufficient arguments')
    else:
        match_timestamps(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], np.int(sys.argv[5]))

if __name__ == "__main__":
    main()