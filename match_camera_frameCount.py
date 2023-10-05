import numpy as np
import utils as ut
import sys
import matplotlib.pyplot as plt

def match_camera_frameCount(filepath, file_prefix, trial_num, numFrames):

    imagegrab_file_frameCount_cam0 = filepath + file_prefix + '_cam0_short_trial' + trial_num + '.csv'
    imagegrab_file_frameCount_cam1 = filepath + file_prefix + '_cam1_short_trial' + trial_num + '.csv'

    imagegrab_camFrameCount_cam0 = np.array(numFrames * [0])
    imagegrab_camFrameCount_cam1 = np.array(numFrames * [0])

    ut.readcsvFile_int(imagegrab_file_frameCount_cam0, imagegrab_camFrameCount_cam0, 1, 0)
    ut.readcsvFile_int(imagegrab_file_frameCount_cam1, imagegrab_camFrameCount_cam1, 1 ,0)

    ## check if frameCounts increment by 1 in both cameras
    imagegrab_frame_diff_cam0 = imagegrab_camFrameCount_cam0[1:] - imagegrab_camFrameCount_cam0[0:-1]
    imagegrab_frame_diff_cam1 = imagegrab_camFrameCount_cam1[1:] - imagegrab_camFrameCount_cam1[0:-1]

    #get indices that increment by more/less than 1 in both cameras
    imagegrab_camFrameCount_ind_cam0 = np.zeros(numFrames)
    imagegrab_camFrameCount_ind_cam1 = np.zeros(numFrames)

    if(imagegrab_frame_diff_cam0.all() == 1):
        print('FrameCounts increment by 1 in camera 0')
    else:
        ind_cam0 = np.argwhere(imagegrab_frame_diff_cam0 != 0)
        imagegrab_camFrameCount_ind_cam0[ind_cam0] = 1
        plt.plot(ind_cam0)
        plt.yticks(np.arange(-2,2,1))
        plt.xlabel('Frames','.',color='red')
        plt.ylabel('Indexes that have non zero increment')

    if(imagegrab_frame_diff_cam1.all() == 1):
        print('FrameCounts increment by 1 in camera 1')
    else:
        ind_cam1 = np.argwhere(imagegrab_frame_diff_cam1 != 0)
        imagegrab_camFrameCount_ind_cam1[ind_cam1] = 1
        plt.plot(ind_cam1,'.',color='blue')
        plt.yticks(np.arange(-2,2,1))
        plt.xlabel('Frames')
        plt.ylabel('Indexes that have non zero increment')

    plt.show()

    ## match frameCount starts
    if(imagegrab_camFrameCount_cam0[0] == imagegrab_camFrameCount_cam1[0]):
        print('FrameCount starts match for this trial between cameras')
    else:
        print('FrameCount counting does not match for this trial between cameras ',
              abs(imagegrab_camFrameCount_cam1[0] - imagegrab_camFrameCount_cam0[0]))


def main():

    if (len(sys.argv) < 5):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-filepath\n' +
              '-file prefix to frameCount file\n' +
              '-trial number\n' +
              '-numFrames\n')
    else:
        filepath = sys.argv[1]
        file_prefix = sys.argv[2]
        trial_num = sys.argv[3]
        numFrames = np.int(sys.argv[4])

        match_camera_frameCount(filepath, file_prefix, trial_num, numFrames)

if __name__ == "__main__":
    main()