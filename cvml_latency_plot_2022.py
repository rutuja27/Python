import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut

def main():

    data_dir = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/b67a7_6_13_2022/'
    data_dir_jaaba = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/33cc6_8_1_2022/'
    trial_type='1'
    numFrames = 100000
    imagegrab_file = data_dir +'imagegrab_nidaqcam0_short_trial' + trial_type + '.csv'
    imagedispatch_file = data_dir + 'imagedispatch_nidaqcam0_short_trial' + trial_type + '.csv';
    jaaba_file= data_dir_jaaba + 'jaaba_plugin_process_timecam0_short_trial' + trial_type + '.csv'

    imagegrab_lat = np.array(numFrames*[0.0])
    imagegrab_cam_lat = np.array(numFrames*[0.0])
    imagedispatch_lat = np.array(numFrames*[0.0])

    jaaba_process_lat = np.array(numFrames*[0.0])

    ut.readcsvFile_nidaq(imagegrab_file, imagegrab_lat, imagegrab_cam_lat, 0, '')
    ut.readcsvFile_nidaq(imagedispatch_file, imagedispatch_lat,imagegrab_cam_lat, 0,'')
    ut.readcsvFile_f2f(jaaba_file, jaaba_process_lat,0,0, '')

    imagedispatch_proc_lat = imagedispatch_lat - imagegrab_lat
    end_to_end_lat = imagegrab_lat + imagedispatch_proc_lat + jaaba_process_lat
    # sort the data:
    data_sorted = np.sort(end_to_end_lat[:])
    print(data_sorted)
    binwidth = 0.1
    min_bin = 2
    max_bin=7
    # calculate the proportional values of samples
    p = 1. * np.arange(len(end_to_end_lat[:])) / (len(end_to_end_lat[:]) - 1)
    print(p)
    plt.figure()
    ax1 = plt.gca()
    ax1.axes.xaxis.set_ticklabels([])
    ax1.plot(imagegrab_lat[0:6000], color='brown')
    ax1.plot(imagedispatch_proc_lat[0:6000], color='orange')
    ax1.plot(jaaba_process_lat[0:6000], color='pink')
    plt.legend(['Frame grab Time', 'Imagedispatch Time', 'Jaaba Compute Time'], fontsize='8',loc='upper left')
    plt.ylabel('Time in Milliseconds ms')
    plt.title('BIAS Latency')
    #plt.savefig('C:/Users/27rut/OneDrive/Pictures/cvml 2022/BIAS_latency.jpg')
    plt.show()


    plt.figure()
    ax2 = plt.gca()
    plt.plot(data_sorted, p)
    plt.xlim([3,8])
    plt.xticks(np.arange(3,8,0.5))
    plt.ylabel('Cumulative Fraction of frames')
    plt.xlabel('Time in ms')
    plt.title('CDF of end to end latency')
    #plt.savefig('C:/Users/27rut/OneDrive/Pictures/cvml 2022/cdf_latency.jpg')
    plt.show()






if __name__ == "__main__":
        main()
