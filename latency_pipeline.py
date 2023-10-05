import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut
import sys
import matplotlib

def plot_pipeline_latency(filepath, file_prefix_imagegrab, file_prefix_jaaba, numFrames, trial_num, period_ms, range):

    imagegrab_file_nidaq_camtrig_cam0 = filepath + file_prefix_imagegrab + 'cam0_short_trial' + trial_num + '.csv'
    imagegrab_file_nidaq_camtrig_cam1 = filepath + file_prefix_imagegrab + 'cam1_short_trial' + trial_num + '.csv'

    jaaba_file_nidaq_camtrig_cam0 = filepath + file_prefix_jaaba + 'cam0_short_trial' + trial_num + '.csv'
    jaaba_file_nidaq_camtrig_cam1 = filepath + file_prefix_jaaba + 'cam1_short_trial' + trial_num + '.csv'

    imagegrab_nidaq_camtrig_cam0 = np.array(numFrames * [0.0])
    imagegrab_nidaq_camtrig_cam1 = np.array(numFrames * [0.0])
    imagegrab_nidaq_cam0 = np.array(numFrames * [0.0])
    imagegrab_nidaq_cam1 = np.array(numFrames * [0.0])
    jaaba_nidaq_cam0 = np.array(numFrames * [0.0])
    jaaba_nidaq_cam1 = np.array(numFrames * [0.0])

    ut.readcsvFile_nidaq(imagegrab_file_nidaq_camtrig_cam0, imagegrab_nidaq_camtrig_cam0, imagegrab_nidaq_cam0, period_ms)
    ut.readcsvFile_nidaq(imagegrab_file_nidaq_camtrig_cam1, imagegrab_nidaq_camtrig_cam1, imagegrab_nidaq_cam1, period_ms)
    ut.readcsvFile_nidaq(jaaba_file_nidaq_camtrig_cam0, imagegrab_nidaq_camtrig_cam0, jaaba_nidaq_cam0, period_ms)
    ut.readcsvFile_nidaq(jaaba_file_nidaq_camtrig_cam1, imagegrab_nidaq_camtrig_cam1, jaaba_nidaq_cam1, period_ms)

    print(imagegrab_nidaq_cam0[range])
    print(jaaba_nidaq_cam0[range])

    stages = ['Jaaba', 'Imagegrab']

    fig,ax= plt.subplots(1,2 , figsize=(20,10), gridspec_kw={
             'hspace':0.1, 'wspace':0.1
    })
    fig.tight_layout()
    height=0.3
    alpha1=0.2
    fontsize=15

    #ax.barh(['Camera Trigger'],imagegrab_nidaq_camtrig_cam0[0], height=height,color='blue', alpha=alpha1)
    ax[0].barh(range, jaaba_nidaq_cam0[range]-imagegrab_nidaq_cam0[range], height=height,
            left=imagegrab_nidaq_cam0[range], color='blue', alpha=alpha1,align='edge')
    ax[0].barh(range, imagegrab_nidaq_cam0[range]-imagegrab_nidaq_camtrig_cam0[range],height=height,
            left=imagegrab_nidaq_camtrig_cam0[range], color='red', alpha=alpha1)
    ax[0].set_xticks(np.arange(0,max(jaaba_nidaq_cam0[range]),2.5))
    ax[0].invert_yaxis()
    ax[0].set_xlabel('Time in ms', fontsize=fontsize-3)
    ax[0].set_ylabel('Frames', fontsize=fontsize-3)
    ax[0].set_title('Camera 0 latency pipeline', fontsize=fontsize)
    ax[0].set_yticks(range)
    ax[0].grid()

    ax[1].barh(range, jaaba_nidaq_cam1[range]-imagegrab_nidaq_cam1[range], height=height,
           left=imagegrab_nidaq_cam1[range], color='blue', alpha=alpha1,align='edge')
    ax[1].barh(range, imagegrab_nidaq_cam1[range]-imagegrab_nidaq_camtrig_cam1[range],height=height,
            left=imagegrab_nidaq_camtrig_cam1[range], color='red', alpha=alpha1)
    ax[1].set_xticks(np.arange(0,max(jaaba_nidaq_cam1[range]),2.5))
    ax[1].invert_yaxis()
    ax[1].set_xlabel('Time in ms', fontsize=fontsize-3)
    ax[1].set_ylabel('Frames', fontsize=fontsize-3)
    ax[1].set_title('Camera 1 latency pipeline', fontsize=fontsize)
    ax[1].set_yticks(range)
    ax[1].grid()
    plt.legend(stages, loc='upper right')

    plt.show()

    print(jaaba_nidaq_cam1)

def main():
    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if (len(sys.argv) < 6):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-filepath to latency data dir\n' +
              '-file prefix imagegrab file \n' +
              '-file prefix to jaaba file\n ' +
              '-number of frames\n' +
              '-trial number\n' +
              '-nidaq fast clock rate')
    else:
        filepath = sys.argv[1]
        file_prefix_imagegrab =  sys.argv[2]
        file_prefix_jaaba = sys.argv[3]
        numFrames = np.int(sys.argv[4])
        trial_num = sys.argv[5]
        fast_clock_rate = np.int(sys.argv[6])

        period_ms = (1/fast_clock_rate)*1000  ##convert clock period to ms
        range = np.arange(0,10,1)
        print(range)
        plot_pipeline_latency(filepath, file_prefix_imagegrab, file_prefix_jaaba, numFrames, trial_num, period_ms, range)

if __name__ == "__main__":
    main()