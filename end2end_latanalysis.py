import numpy as np
import utils as ut
import sys
import matplotlib.pyplot as plt

def lat_analysis(end2end_latency, numFrames, bout_len_sec, lat_thres):

    num_intervals = numFrames// bout_len_sec
    count_spikes_per_bout = np.array(num_intervals * [0])
    #print('Number of intervals ',num_intervals)
    #print('latency threshold ', lat_thres)

    for interval_id in range(0,num_intervals):
        for frame_id in range(0, bout_len_sec):
            if(end2end_latency[(interval_id*bout_len_sec) + frame_id] > lat_thres):
                count_spikes_per_bout[interval_id] += 1
    #print(count_spikes_per_bout)
    return sum(count_spikes_per_bout)/num_intervals

def read_end2endData(filepath, classifier_filename, imagegrab_file_prefix,
                     numFrames, no_of_trials, conversion_factor, framerate,
                     latency_threshold, isnidaq):

    lat_spikes = np.array(no_of_trials * [0.0])
    classifier_lat = np.array(numFrames * [0.0])
    imagegrab_startlat_cam0 = np.array(numFrames * [0.0])
    imagegrab_startlat_cam1 = np.array(numFrames * [0.0])
    imagegrab_trig_cam0 = np.array(numFrames * [0.0])
    imagegrab_trig_cam1 = np.array(numFrames * [0.0])

    for trial_id in range(1, no_of_trials+1):
        classifier_score_file = filepath + classifier_filename + str(trial_id) + '.csv'
        imagegrab_filename_cam0 = filepath + imagegrab_file_prefix + 'cam0_short_trial' + str(trial_id) + '.csv'
        imagegrab_filename_cam1 = filepath + imagegrab_file_prefix + 'cam1_short_trial' + str(trial_id) + '.csv'

        if isnidaq:
            ut.readcsvFile_nidaq(imagegrab_filename_cam0,imagegrab_trig_cam0, imagegrab_startlat_cam0, conversion_factor)
            ut.readcsvFile_nidaq(imagegrab_filename_cam1, imagegrab_trig_cam1, imagegrab_startlat_cam1, conversion_factor)
        else:
            ut.readcsvFile_int(imagegrab_filename_cam0, imagegrab_startlat_cam0, conversion_factor, 0)
            ut.readcsvFile_int(imagegrab_filename_cam1, imagegrab_startlat_cam1, conversion_factor, 0)

        ut.read_score(classifier_score_file, classifier_lat, 0, 0)
        #print(classifier_lat)

        if isnidaq:
            classifier_lat = classifier_lat * conversion_factor
        else:
            classifier_lat = classifier_lat/conversion_factor

        # minimum of the imagegrab start times
        if isnidaq:
            imagegrab_min = np.minimum(imagegrab_trig_cam0, imagegrab_trig_cam1)
        else:
            imagegrab_min = np.minimum(imagegrab_startlat_cam0, imagegrab_startlat_cam1)

        #print(imagegrab_min)

        # classifier output score time
        end2end_latency = classifier_lat - imagegrab_min
        print(end2end_latency)

        bout_length = framerate
        lat_spikes[trial_id-1] = lat_analysis(end2end_latency, numFrames, bout_length, latency_threshold)
    average_spikes_per_sec = sum(lat_spikes)/no_of_trials
    print(lat_spikes)
    print(average_spikes_per_sec)

def main():

    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if(len(sys.argv) < 10):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-path to latency data dir\n' +
              '-predicted  score file name\n' +
              '-prefix to imagegrab lat data\n' +
              '-number of frames\n' +
              '-no of trials\n' +
              '-conversion factor for latency\n' +
              'framerate\n' +
              'threshold\n' +
              'isnidaq\n'
              )
        return -1
    else:
        filepath = sys.argv[1]
        classifier_filename = sys.argv[2]
        imagegrab_file_prefix = sys.argv[3]
        numFrames = np.int(sys.argv[4])
        no_of_trials = np.int(sys.argv[5])
        conversion_factor = np.float(sys.argv[6])
        framerate = np.int(sys.argv[7])
        latency_threshold = np.int(sys.argv[8])
        isnidaq = np.int(sys.argv[9])


    read_end2endData(filepath, classifier_filename, imagegrab_file_prefix, numFrames,
                     no_of_trials, conversion_factor, framerate, latency_threshold, isnidaq)

if __name__ == "__main__":
    main()