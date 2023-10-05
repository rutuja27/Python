import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut
import sys


def match_raw_timestamps(filepath, file_prefix, trial_num, numFrames):

    imagegrab_file_raw_ts_cam0 = filepath + file_prefix + "_cam0_short_trial" + trial_num + ".csv"
    imagegrab_file_raw_ts_cam1 = filepath + file_prefix + "_cam1_short_trial" + trial_num + ".csv"

    imagegrab_raw_ts_cam0 = np.array(numFrames * [0], dtype=float)
    imagegrab_raw_ts_cam1 = np.array(numFrames * [0], dtype=float)

    ut.readcsvFile_float(imagegrab_file_raw_ts_cam0, imagegrab_raw_ts_cam0)
    ut.readcsvFile_float(imagegrab_file_raw_ts_cam1, imagegrab_raw_ts_cam1)

    imagegrab_rawts_diff_cam0 = (imagegrab_raw_ts_cam0[1:] - imagegrab_raw_ts_cam0[0:-1])*0.001
    imagegrab_rawts_diff_cam1 = (imagegrab_raw_ts_cam1[1:] - imagegrab_raw_ts_cam1[0:-1])*0.001
    imagegrab_rawts_sync = []

    for i in range(0,numFrames):
        imagegrab_rawts_sync.append(abs(imagegrab_raw_ts_cam0[i] - imagegrab_raw_ts_cam1[i])*0.001)
    max_rawts_sync = max(imagegrab_rawts_sync)
    print(max_rawts_sync)

    print('raw ts diff between cams', abs(imagegrab_raw_ts_cam0[0] - imagegrab_raw_ts_cam1[0]))
    fig3 = plt.figure(figsize=(10, 10))
    fig3.tight_layout()
    fontsize = 15
    plt.plot(imagegrab_rawts_diff_cam0,'.', color='red', alpha=0.2)
    plt.plot(imagegrab_rawts_diff_cam1,'.', color='blue', alpha=0.5)
    plt.title('Difference in raw timeStamps between frames', fontsize=fontsize)
    plt.xlabel('Frames', fontsize=fontsize - 3)
    plt.ylabel('Time in ms', fontsize=fontsize - 3)
    plt.legend(['Cam 0', 'Cam 1'])

    fig4 = plt.figure(figsize=(10, 10))
    fig4.tight_layout()
    fontsize = 15
    plt.plot(imagegrab_rawts_sync, '.')
    plt.yticks(np.arange(max_rawts_sync-0.1, max_rawts_sync+0.1,0.01))
    plt.title('Difference in raw timestamps between cameras ', fontsize=fontsize)
    plt.ylabel('Time in ms', fontsize=fontsize - 3)
    plt.xlabel('Frames', fontsize=fontsize - 3)



def match_imagegrab2jaaba_timestamps(filepath, file_suffix1, file_suffix2, trial_num, numFrames):

     multiplying_factor=1000
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
         imagegrab_sync_diff.append((abs(imagegrab_ts_cam1[i] - imagegrab_ts_cam0[i])))

     imagegrab_ts_diff_cam0 = (imagegrab_ts_cam0[1:] - imagegrab_ts_cam0[0:-1]) #* multiplying_factor
     imagegrab_ts_diff_cam1 = (imagegrab_ts_cam1[1:] - imagegrab_ts_cam1[0:-1]) #* multiplying_factor

     print('Not matching indexes in cam 0', not_matching_indexes_cam0)
     print('Not matching indexes in cam 1', not_matching_indexes_cam1)

     fig1 = plt.figure(figsize=(10, 10))
     fig1.tight_layout()
     fontsize = 15
     plt.plot(imagegrab_sync_diff,'.')
     plt.title('Difference in timestamps between cameras ', fontsize=fontsize)
     plt.ylabel('Time in ms', fontsize=fontsize-3)
     plt.xlabel('Frames', fontsize=fontsize-3)

     fig2 = plt.figure(figsize=(10, 10))
     fig2.tight_layout()
     fontsize = 15
     plt.plot(imagegrab_ts_diff_cam0, '.', color='red')
     plt.plot(imagegrab_ts_diff_cam1, '.', color='blue')
     plt.title('Difference in timeStamps between frames', fontsize=fontsize)
     plt.xlabel('Frames', fontsize=fontsize-3)
     plt.ylabel('Time in ms', fontsize=fontsize-3)
     plt.legend(['Cam 0', 'Cam 1'])


def main():

    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if(len(sys.argv) < 7):
        print('Insufficient arguments')
        print('Argument options\n' +
              '-filepath\n' +
              '-image grab convert ts prefix\n' +
              '-image grab raw ts prefix\n' +
              '-jaaba convert ts prefix\n' +
              '-trial num\n' +
              '-numFrames'
              )
    else:
        filepath = sys.argv[1]
        imagegrab_convert_ts_prefix = sys.argv[2]
        imagegrab_raw_ts_prefix = sys.argv[3]
        jaaba_convert_ts_prefix = sys.argv[4]
        trial_num = sys.argv[5]
        numFrames = np.int(sys.argv[6])
        match_imagegrab2jaaba_timestamps(filepath,imagegrab_convert_ts_prefix, jaaba_convert_ts_prefix, trial_num, numFrames)
        match_raw_timestamps(filepath, imagegrab_raw_ts_prefix, trial_num, numFrames)
        plt.show()
if __name__ == "__main__":
    main()