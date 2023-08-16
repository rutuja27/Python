import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut
import sys

def match_nidaq_timestamps(filepath, file_suffix, trial_num, numFrames, fast_clock_framerate):

    imagegrab_file_nidaq_ts0 = filepath + file_suffix + 'cam0_short_trial' + trial_num + '.csv'
    imagegrab_file_nidaq_ts1 = filepath + file_suffix + 'cam1_short_trial' + trial_num + '.csv'
    print(imagegrab_file_nidaq_ts0)
    print(imagegrab_file_nidaq_ts1)

    imagegrab_nidaqcamtrig_ts_cam0 = np.array(numFrames * [0])
    imagegrab_nidaqcamtrig_ts_cam1 = np.array(numFrames * [0])

    ut.readcsvFile_int(imagegrab_file_nidaq_ts0, imagegrab_nidaqcamtrig_ts_cam0,1, 0)
    ut.readcsvFile_int(imagegrab_file_nidaq_ts1, imagegrab_nidaqcamtrig_ts_cam1,1, 0)

    nidaq_mismatch_ind = []
    period = (1/fast_clock_framerate)*1000

    for i in range(0,numFrames):
        if(imagegrab_nidaqcamtrig_ts_cam0[i] == imagegrab_nidaqcamtrig_ts_cam1[i]):
            continue
        else:
            nidaq_mismatch_ind.append(i)

    print('Mismatch nidaq indexes', nidaq_mismatch_ind)

    fontsize=15
    plt.plot((imagegrab_nidaqcamtrig_ts_cam0[1:] - imagegrab_nidaqcamtrig_ts_cam0[0:-1])*period, '.', color='red', alpha=0.8)
    plt.plot((imagegrab_nidaqcamtrig_ts_cam1[1:] - imagegrab_nidaqcamtrig_ts_cam1[0:-1])*period, '.', color='blue',alpha=0.2)
    plt.title('Camera Trigger Differences', fontsize=fontsize)
    plt.ylabel('Time in ms')
    plt.xlabel('Frames')
    plt.yticks(np.arange(-2, 4, 0.5))

    plt.show()

def main():

    if (len(sys.argv) < 5):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-filepath\n' +
              '-file prefix to nidaq file\n' +
              '-trial number\n' +
              '-numFrames\n' +
              '-fast_clock_framrate\n')
    else:
        filepath = sys.argv[1]
        file_suffix = sys.argv[2]
        trial_num = sys.argv[3]
        numFrames = np.int(sys.argv[4])
        fast_clock_framerate = np.int(sys.argv[5])

        match_nidaq_timestamps(filepath, file_suffix, trial_num, numFrames, fast_clock_framerate)

if __name__ == "__main__":
    main()