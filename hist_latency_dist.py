
import csv
import matplotlib.pyplot as plt
import numpy as np
import skipped_frames_correlation as sfc
import utils as ut
import plotting_code as pc
import lat_defs as ld

def get_peaks(arr, thres):
    return arr > thres

def main():

    file_dir = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/3ed1c_11_29_2022/'
    file_names = ['imagegrab_nidaqcam0_short_', 'imagegrab_nidaqcam1_short_']
    numFrames = 100000
    no_of_trials = 5;
    thres =4.0

    imagegrab_lat_nidaq = ld.LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials * [numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]))
    jaaba_lat_nidaq = ld.LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]), \
                                        np.array(no_of_trials * [numFrames * [0.0]]), \
                                        np.array(no_of_trials*[numFrames * [0.0]]))

    max_peak_img=0
    max_peak_jaaba=0
    jaaba_num_peaks=0
    imagegrab_num_peaks=0

    for trial_id in range(0,no_of_trials):

        trial_suffix = 'trial'+ str(trial_id+1) + '.csv'
        imagegrab_file = file_dir + file_names[0] + trial_suffix
        jaaba_file = file_dir + file_names[1] + trial_suffix
        ut.readcsvFile_nidaq(imagegrab_file,imagegrab_lat_nidaq.lat_camtrig[trial_id], imagegrab_lat_nidaq.lat_nidaq[trial_id])
        ut.readcsvFile_nidaq(jaaba_file, jaaba_lat_nidaq.lat_camtrig[trial_id], jaaba_lat_nidaq.lat_nidaq[trial_id])

        # find peak
        peaks = get_peaks(imagegrab_lat_nidaq.lat_nidaq[trial_id], thres)
        # count peaks
        imagegrab_num_peaks += sum(bool(val) for val in peaks)

        peaks = get_peaks(jaaba_lat_nidaq.lat_nidaq[trial_id], thres)
        # count peaks
        jaaba_num_peaks += sum(bool(val) for val in peaks)

    max_peak_img =np.max(imagegrab_lat_nidaq.lat_nidaq)
    max_peak_jaaba = np.max(jaaba_lat_nidaq.lat_nidaq)
    max_peak = max(max_peak_img, max_peak_jaaba)
    print('Imagegrab number of peaks', imagegrab_num_peaks)
    print('Jaaba number of peaks', jaaba_num_peaks)
    print('Maximum Latency Spike', max_peak)

    #histogram of magnitude of peaks
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.tight_layout(pad=3.0)
    bins = np.arange(0,max_peak)

    img_lat = imagegrab_lat_nidaq.lat_nidaq.flatten()
    jaaba_lat = jaaba_lat_nidaq.lat_nidaq.flatten()

    counts1, bins1, patches1 = ax1.hist(img_lat, bins=bins ,color='red',alpha=0.5, rwidth=1,
                                        align = 'mid', log=True, label='Imagegrab')
    counts2, bins2, patches2 = ax1.hist(jaaba_lat, bins=bins, color='blue', alpha=0.5, rwidth=1,
                                        align='mid',log=True, label='Jaaba Grab')

    ax1.set_xticks(bins[:-1])
    #ax1.set_yticks(np.arange(0,1.1,0.1))
    plt.legend()
    plt.xlabel('Latency in ms')
    fig.suptitle("Distribution of differences syncing JAABA- between Cam 0 and Cam 1 ")
    #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/jaabagrab_diff_cam0Vscam1_wcomputeon.jpg')
    plt.show()

if __name__ == "__main__":
    main()