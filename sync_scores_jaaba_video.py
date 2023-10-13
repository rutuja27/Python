import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut
import sys
import matplotlib

def readcsvFile(filename, arr):

    with open(filename, 'r', newline='') as f:
        frm_id=0
        config_reader = csv.reader(f, delimiter=',')
        for row in config_reader:
            arr[frm_id] = float(row[0])/1000
            frm_id += 1


def readcsvFile_int(filename, arr):

    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')
    datatype = arr.dtype
    print(datatype)

    for idx, row in enumerate(data_grab):
        arr[idx] = np.cast[datatype](row[0])
    fhandle.close()

def read_score(filename,arr_scr, arr_scr_ts, flag_gt,index):

    with open(filename, 'r', newline='') as f:
        config_reader = csv.reader(f, delimiter=',')
        for idx,row in enumerate(config_reader):
            if (idx == 0 and not flag_gt):
                continue
            arr_scr[idx-1] = np.float(row[3])
            if not flag_gt:
                arr_scr_ts[idx-1] = np.float(row[index])/1000.0
    f.close()

def read_score_gt(filename,arr_scr, arr_scr_ts, flag_gt,index):

    with open(filename, 'r', newline='') as f:
        config_reader = csv.reader(f, delimiter=',')
        for idx,row in enumerate(config_reader):
            if (idx == 0 and not flag_gt):
                continue
            arr_scr[idx-1] = np.float(row[1])
            if not flag_gt:
                arr_scr_ts[idx-1] = np.float(row[index])/1000.0

    f.close()

def readcsvFile_float(filename, arr, f2f_flag):

    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')

    for idx, row in enumerate(data_grab):

        if f2f_flag:
            if idx == 0:
                prev = np.float(row[0])
            else:
                arr[idx] = (np.float(row[0]) - prev) / 1000
                prev = np.float(row[0])
        else:
            arr[idx] = (np.float(row[0]) / 1000)

    fhandle.close()


def plot_raw_latencydata(isImagegrabflag, isJaabaflag, isClassifierFlag,
                         image_proc_cam0, image_proc_cam1,
                         image_nidaq_cam0, image_nidaq_cam1,
                         jaaba_proc_cam0, jaaba_proc_cam1,
                         jaaba_nidaq_cam0, jaaba_nidaq_cam1,
                         image_nidaq_camtrig_cam0, image_nidaq_camtrig_cam1,
                         start_range, end_range):

    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    fig, ax = plt.subplots(3, 2)
    fig.set_figwidth(10)
    fig.set_figheight(8)
    fig.tight_layout(pad=5.0)
    fontsize=12
    range = np.arange(start_range,end_range)

    max_yrange_imag_proc = max(max(image_proc_cam0), max(image_proc_cam1))
    ax[0, 0].plot(image_proc_cam0[range], '.', color='red', alpha=0.7)
    ax[0, 0].plot(image_proc_cam1[range], '.', color='blue', alpha=0.4)
    ax[0, 0].set_title('Imagegrab Processing Time', fontsize=fontsize)
    #ax[0, 0].legend(['Cam 0', 'Cam 1'], fontsize=fontsize-2)
    ax[0,0].set_yticks(np.arange(0,max_yrange_imag_proc+1,1.0))
    ax[0,0].set_xlabel('Frames',fontsize=fontsize)
    ax[0,0].set_ylabel('Latency in ms',fontsize=fontsize)

    max_yrange_imag_nidaq = max(max(image_nidaq_cam0), max(image_nidaq_cam1))
    ax[1, 0].plot(image_nidaq_cam0[range],'.', color='green', alpha=0.7)
    ax[1, 0].plot(image_nidaq_cam1[range],'.', color='orange', alpha=0.4)
    ax[1, 0].set_title('Imagegrab Nidaq processing Time', fontsize=fontsize)
    #ax[1, 0].legend(['Cam 0', 'Cam 1'], fontsize=fontsize-2)
    ax[1, 0].set_yticks(np.arange(0, max_yrange_imag_nidaq+1, 1.0))
    ax[1, 0].set_xlabel('Frames', fontsize=fontsize)
    ax[1, 0].set_ylabel('Latency in ms',fontsize=fontsize)

    max_yrange_jaaba_proc = max(max(jaaba_proc_cam0), max(jaaba_proc_cam1))
    ax[0, 1].plot(jaaba_proc_cam0[range], '.', color='pink', alpha=0.7)
    ax[0, 1].plot(jaaba_proc_cam1[range], '.', color='darkblue', alpha=0.4)
    ax[0, 1].set_title('Jaaba Processing Time', fontsize=fontsize)
    #ax[0, 1].legend(['Cam 0', 'Cam 1'], fontsize=fontsize-2)
    ax[0, 1].set_yticks(np.arange(0, max_yrange_jaaba_proc+1, 0.5))
    ax[0, 1].set_xlabel('Frames',fontsize=fontsize)
    ax[0, 1].set_ylabel('Latency in ms',fontsize=fontsize)

    max_yrange_jaaba_nidaq = max(max(jaaba_nidaq_cam0), max(jaaba_nidaq_cam1))
    ax[1, 1].plot(jaaba_nidaq_cam0[range],'.', color='lightgreen', alpha=0.7)
    ax[1, 1].plot(jaaba_nidaq_cam1[range],'.', color='brown', alpha=0.4)
    ax[1, 1].set_title('Jaaba Nidaq processing Time', fontsize=fontsize)
    ax[1, 1].legend(['Cam 0', 'Cam 1'], fontsize=fontsize-2)
    #ax[1, 1].set_yticks(np.arange(0, max_yrange_jaaba_nidaq+1, 0.5))
    ax[1, 1].set_xlabel('Frames',fontsize=fontsize)
    ax[1, 1].set_ylabel('Latency in ms',fontsize=fontsize)

    ax[2, 0].plot(image_nidaq_camtrig_cam0[start_range+1:end_range]-image_nidaq_camtrig_cam0[start_range:end_range-1], '.', alpha=0.7)
    ax[2, 0].plot(image_nidaq_camtrig_cam1[start_range+1:end_range]-image_nidaq_camtrig_cam0[start_range:end_range-1], '.', alpha=0.4)
    #print("Camera trigger differences",image_nidaq_camtrig_cam1[start_range+1:end_range]
    #                                                                 -image_nidaq_camtrig_cam0[start_range:end_range-1])
    ax[2, 0].set_title('Camera Trigger Differences', fontsize=fontsize)
    ax[2, 0].set_yticks(np.arange(-2, 4, 0.5))

    ax[2,1].set_visible(False)

def plot_sync_btwn_cams(isImagegrabflag, isJaabaflag, isClassifierFlag,
                        imagegrab_start_cam0, imagegrab_start_cam1,
                        imagegrab_end_cam0, imagegrab_end_cam1,
                        jaaba_start_cam0, jaaba_start_cam1,
                        jaaba_end_cam0, jaaba_end_cam1, start_range, end_range):

    diff_betwn_imagegrab_start = abs(imagegrab_start_cam1 - imagegrab_start_cam0)
    diff_betwn_jaaba_start = abs(jaaba_start_cam1 - jaaba_start_cam0)
    diff_betwn_imagegrab_end = abs(imagegrab_end_cam1-imagegrab_end_cam0)
    diff_betwn_jaaba_end = abs(jaaba_end_cam1-jaaba_end_cam0)

    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    fig1, ax1 = plt.subplots(2, 2)
    fig1.set_figwidth(10)
    fig1.set_figheight(8)
    fig1.tight_layout(pad=5.0)
    fontsize=12

    range = np.arange(start_range, end_range)

    ax1[0, 0].plot(diff_betwn_imagegrab_start[range], '.', color='red', alpha=0.2)
    ax1[0, 0].set_title('Difference between imagegrab start time', fontsize=fontsize)
    ax1[0, 0].set_xlabel('Frames', fontsize=fontsize)
    ax1[0, 0].set_ylabel('Latency in ms', fontsize=fontsize)

    ax1[1, 0].plot(diff_betwn_imagegrab_end[range], '.', color='blue', alpha=0.2)
    ax1[1, 0].set_title('Difference between imagegrab end time', fontsize=fontsize)
    ax1[1, 0].set_xlabel('Frames', fontsize=fontsize)
    ax1[1, 0].set_ylabel('Latency in ms', fontsize=fontsize)

    ax1[0, 1].plot(diff_betwn_jaaba_start[range],'.', color='green', alpha=0.2)
    ax1[0, 1].set_title('Difference between jaaba start time', fontsize=fontsize)
    ax1[0, 1].set_xlabel('Frames', fontsize=fontsize)
    ax1[0, 1].set_ylabel('Latency in ms', fontsize=fontsize)

    ax1[1, 1].plot(diff_betwn_jaaba_end[range],'.', color='orange', alpha=0.2)
    ax1[1, 1].set_title('Difference between jaaba end time', fontsize=fontsize)
    ax1[1, 1].set_xlabel('Frames', fontsize=fontsize)
    ax1[1, 1].set_ylabel('Latency in ms', fontsize=fontsize)

def plot_skipped_frames_latencyplot(isClassifier, total_lat,
                                    classifier_scr_view, start_range, end_range,
                                    latency_threshold):

    range = np.arange(start_range,end_range)

    if isClassifier:
        numFrames = len(total_lat)
        classifier_skips = np.zeros(numFrames) #frames that get scores from either view in the classification
        classifier_skips_nonmatch = np.zeros(numFrames)
        ind_skips = np.argwhere((classifier_scr_view == 2) | (classifier_scr_view == 1))
        classifier_skips[ind_skips] = 3
        #print(classifier_scr_view)
        #print(ind_skips_viewB)
        #print(ind_skips_viewA)
        higher_latency_ind = np.argwhere(total_lat > latency_threshold) # all frames where latency is greater than threshold
        non_match_indexes = match_skips2latency(higher_latency_ind, ind_skips) # all frames that are either
                                                        # high latency and not skipped
                                                        # or frames that are skipped but not above the latency threshold
        classifier_skips_nonmatch[non_match_indexes] = 3.5
        print('Number of frames that are skipped/not skipped wrongly are: ', len(non_match_indexes))

        fig2 = plt.figure(figsize=(10,10))
        fig2.tight_layout()
        fontsize=15
        max_total_lat = max(total_lat)
        min_total_lat = min(total_lat)
        print(max_total_lat)
        print(min_total_lat)

        plt.plot(total_lat[range],'.',color='black')
        plt.plot(classifier_skips[range],'.',color='red')
        plt.plot(classifier_skips_nonmatch[range], '.', color='blue')
        plt.yticks(np.arange(0,max_total_lat,(max_total_lat-min_total_lat)/10) ,fontsize=fontsize-2)
        plt.xticks(np.arange(0, end_range-start_range, (end_range-start_range)/10), fontsize=fontsize-2)
        plt.xlabel('Frames', fontsize=fontsize)
        plt.ylabel('Latency in ms', fontsize=fontsize)
        plt.legend(['Latency', 'skipped', 'latency and skip non match'])
        plt.title('Classifier end to end latency with skipped frames', fontsize=fontsize)


def plot_process_scores_latency(isClassifier, isVideo,
                                jaaba_end_time_cam0, jaaba_end_time_cam1,
                                total_lat, start_range, end_range):

    range = np.arange(start_range,end_range)
    if isClassifier:
        lat_processScores = total_lat - np.maximum(jaaba_end_time_cam1,jaaba_end_time_cam0)

    fig3 = plt.figure(figsize=(10, 10))
    fig3.tight_layout()
    fontsize = 15
    plt.plot(lat_processScores[range], '.', color='red')


def match_skips2latency(higher_latency_ind, ind_skips):
    return np.setxor1d(higher_latency_ind,ind_skips) # non intersecting frames

def read_latency_rawdata(filepath, cls_filepath, numFrames, trial_type,
                         exp_dir, isVideo, numCameras,
                         isImagegrab, isJaaba, isClassifier, latency_threshold,
                         start_range, end_range):
    if isImagegrab:
        imagegrab_file_cam0 = filepath + 'imagegrab_start_timecam0_short_trial' + trial_type + '.csv'
        img_proc_cam0 = filepath + 'imagegrab_process_timecam0_short_trial' + trial_type + '.csv'
        img_end_time_cam0 = filepath + 'imagegrab_end_timecam0_short_trial' + trial_type + '.csv'
        imagegrab_nidaq_cam0 = filepath + 'imagegrab_nidaqcam0_short_trial' + trial_type + '.csv'
        image_nidaqThres_cam0 = filepath + 'imagegrab_nidaq_threscam0_short_trial' + trial_type + '.csv'
        #img_skipped_cam0 = filepath + 'imagegrab_skipped_framescam0_short_trial' + trial_type + '.csv'

    if isJaaba:
        jaaba_strtfile_cam0 = filepath + 'jaaba_plugin_start_timecam0_short_trial' + trial_type + '.csv'
        jaaba_endfile_cam0 = filepath + 'jaaba_plugin_end_time_cam0_short_trial' + trial_type + '.csv'
        jaaba_proc_cam0 = filepath + 'jaaba_plugin_process_timecam0_short_trial' + trial_type + '.csv'
        jaaba_nidaq_cam0 = filepath + 'jaaba_plugin_nidaqcam0_short_trial' + trial_type + '.csv'
        jaaba_nidaqThres_cam0 = filepath + 'jaaba_plugin_nidaq_threscam0_short_trial' + trial_type + '.csv'

    if numCameras == 2:

        if isImagegrab:
            imagegrab_file_cam1 = filepath + 'imagegrab_start_timecam1_short_trial' + trial_type + '.csv'
            img_proc_cam1 = filepath + 'imagegrab_process_timecam1_short_trial' + trial_type + '.csv'
            img_end_time_cam1 = filepath + 'imagegrab_end_timecam1_short_trial' + trial_type + '.csv'
            imagegrab_nidaq_cam1 = filepath + 'imagegrab_nidaqcam1_short_trial' + trial_type + '.csv'
            image_nidaqThres_cam1 = filepath + 'imagegrab_nidaq_threscam1_short_trial' + trial_type + '.csv'
            # img_skipped_cam1 = filepath + 'imagegrab_skipped_framescam1_short_trial' + trial_type + '.csv'

        if isJaaba:
            jaaba_strtfile_cam1 = filepath + 'jaaba_plugin_start_timecam1_short_trial' + trial_type + '.csv'
            jaaba_endfile_cam1 = filepath + 'jaaba_plugin_end_time_cam1_short_trial' + trial_type + '.csv'
            jaaba_proc_cam1 = filepath + 'jaaba_plugin_process_timecam1_short_trial' + trial_type + '.csv'
            jaaba_nidaq_cam1 = filepath + 'jaaba_plugin_nidaqcam1_short_trial' + trial_type + '.csv'
            jaaba_nidaqThres_cam1 = filepath + 'jaaba_plugin_nidaq_threscam1_short_trial' + trial_type + '.csv'

    if exp_dir:
        classifier_scr_file = classifier_filepath + 'classifier_score.csv'
    else:
        classifier_scr_file = filepath + 'classifier_v' + trial_type.rjust(3,'0') + '.csv'
        print(classifier_scr_file)

    # allocate data
    if isImagegrab:
        imagegrab_process_time_cam0 = np.array(numFrames * [0.0])
        imagegrab_process_time_cam1 = np.array(numFrames * [0.0])
        imagegrab_start_cam0 = np.array(numFrames * [0.0])
        imagegrab_start_cam1 = np.array(numFrames * [0.0])
        image_end_time_cam0 = np.array(numFrames * [0.0])
        image_end_time_cam1 = np.array(numFrames * [0.0])
        image_nidaq_cam0 = np.array(numFrames * [0.0])
        image_nidaq_cam1 = np.array(numFrames * [0.0])
        image_nidaq_camtrig0 = np.array(numFrames * [0.0])
        image_nidaq_camtrig1 = np.array(numFrames * [0.0])
        image_nidaqThres0 = np.array(numFrames * [0.0])
        image_nidaqThres1 = np.array(numFrames * [0.0])
        imagegrab_skipped_cam0 = np.array(numFrames * [0.0])
        imagegrab_skipped_cam1 = np.array(numFrames * [0.0])

    if isJaaba:
        jaaba_process_time_cam0 = np.array(numFrames * [0.0])
        jaaba_process_time_cam1 = np.array(numFrames * [0.0])
        jaaba_start_cam0 = np.array(numFrames * [0.0])
        jaaba_start_cam1 = np.array(numFrames * [0.0])
        jaaba_end_cam0 = np.array(numFrames * [0.0])
        jaaba_end_cam1 = np.array(numFrames * [0.0])
        jaaba_nidaq_camtrig0 = np.array(numFrames * [0.0])
        jaaba_nidaq_camtrig1 = np.array(numFrames * [0.0])
        jaaba_nidaqThres0 = np.array(numFrames * [0.0])
        jaaba_nidaqThres1 = np.array(numFrames * [0.0])
        jaaba_nidaqcam0 = np.array(numFrames * [0.0])
        jaaba_nidaqcam1 = np.array(numFrames * [0.0])

    if isClassifier:
        classifier_scr = np.array((numFrames) * [0.0])
        classifier_scr_gt = np.array((numFrames) * [0.0])
        classifier_scr_ts = np.array((numFrames ) * [0.0])
        classifier_scr_view = np.array((numFrames) * [0])
        classifier_side_scr_ts = np.array((numFrames) * [0.0])
        classifier_front_scr_ts = np.array((numFrames ) * [0.0])

    # Read data from csv
    if isImagegrab:
        ut.readcsvFile_int(imagegrab_file_cam0, imagegrab_start_cam0, numFrames, 1000, 0)
        ut.readcsvFile_int(img_end_time_cam0, image_end_time_cam0, numFrames, 1000, 0)
        ut.readcsvFile_int(img_proc_cam0, imagegrab_process_time_cam0, numFrames, 1000, 0)
        ut.readcsvFile_nidaq(imagegrab_nidaq_cam0, image_nidaq_camtrig0, image_nidaq_cam0, numFrames, 0.02)
        ut.readcsvFile_int(image_nidaqThres_cam0, image_nidaqThres0, numFrames, 1, 0)
        #ut.readcsvFile_int(img_skipped_cam0, imagegrab_skipped_cam0, numFrames, 1, 0)

    if isJaaba:
        ut.readcsvFile_nidaq(jaaba_nidaq_cam0, jaaba_nidaq_camtrig0, jaaba_nidaqcam0, numFrames, 0.02)
        ut.readcsvFile_int(jaaba_nidaqThres_cam0, jaaba_nidaqThres0, numFrames, 1, 0)
        ut.readcsvFile_int(jaaba_proc_cam0, jaaba_process_time_cam0, numFrames, 1000, 0)
        ut.readcsvFile_int(jaaba_strtfile_cam0, jaaba_start_cam0,numFrames, 1000, 0)
        ut.readcsvFile_int(jaaba_endfile_cam0, jaaba_end_cam0, numFrames, 1000,0)

    if numCameras == 2:

        if isImagegrab:
            ut.readcsvFile_int(imagegrab_file_cam1, imagegrab_start_cam1,numFrames,1000, 0)
            ut.readcsvFile_int(img_proc_cam1, imagegrab_process_time_cam1, numFrames, 1000, 0)
            ut.readcsvFile_int(img_end_time_cam1, image_end_time_cam1, numFrames, 1000, 0)
            ut.readcsvFile_nidaq(imagegrab_nidaq_cam1, image_nidaq_camtrig1, image_nidaq_cam1, numFrames, 0.02)
            ut.readcsvFile_int(image_nidaqThres_cam1, image_nidaqThres1, numFrames, 1, 0)
            #ut.readcsvFile_int(img_skipped_cam1, imagegrab_skipped_cam1, numFrames, 1, 0)

        if isJaaba:
            ut.readcsvFile_nidaq(jaaba_nidaq_cam1, jaaba_nidaq_camtrig1, jaaba_nidaqcam1, numFrames,0.02)
            ut.readcsvFile_int(jaaba_nidaqThres_cam1, jaaba_nidaqThres1, numFrames, 1, 0)
            ut.readcsvFile_int(jaaba_proc_cam1, jaaba_process_time_cam1, numFrames, 1000, 0)
            ut.readcsvFile_int(jaaba_strtfile_cam1, jaaba_start_cam1,numFrames, 1000, 0)
            ut.readcsvFile_int(jaaba_endfile_cam1, jaaba_end_cam1, numFrames, 1000,0)

    # read ts from score files
    if isClassifier:
        ut.read_score(classifier_scr_file, classifier_side_scr_ts, numFrames, 0, 1)
        ut.read_score(classifier_scr_file, classifier_front_scr_ts, numFrames, 0, 2)
        ut.read_score(classifier_scr_file, classifier_scr_ts, numFrames,0, 0)
        ut.read_score(classifier_scr_file, classifier_scr_view, numFrames, 0, 10)

    if isImagegrab:
        image_nidaq_cam0 = image_nidaq_cam0 - image_nidaq_camtrig0
        image_nidaq_cam1 = image_nidaq_cam1 - image_nidaq_camtrig0

        dif_imagegrab = abs(imagegrab_start_cam0 - imagegrab_start_cam1)
        img_min = np.minimum(imagegrab_start_cam0, imagegrab_start_cam1)

    if isJaaba:
        jaaba_nidaqcam0 = jaaba_nidaqcam0 - image_nidaq_camtrig0
        jaaba_nidaqcam1 = jaaba_nidaqcam1 - image_nidaq_camtrig0
        jaaba_nidaqcam0[jaaba_nidaqcam0 < 0] = 0
        jaaba_nidaqcam1[jaaba_nidaqcam1 < 0] = 0

        jaaba2imaggrab_cam0_pc_time = jaaba_start_cam0 - imagegrab_start_cam0
        jaaba2imaggrab_cam1_pc_time = jaaba_start_cam1 - imagegrab_start_cam1
        dif_jaaba_start = abs(jaaba_start_cam0 - jaaba_start_cam1)
        dif_jaaba_end = abs(jaaba_end_cam0 - jaaba_end_cam1)
        imagegrab_min = np.minimum(imagegrab_start_cam1[:],imagegrab_start_cam0[:])

        if isVideo and isClassifier:
            classifier_scr_ts = classifier_scr_ts / 1000
            classifier_side_scr_ts = classifier_side_scr_ts / 1000
            classifier_front_scr_ts = classifier_front_scr_ts / 1000
        elif not isVideo and isClassifier:
            classifier_scr_ts = classifier_scr_ts * 0.02
            classifier_side_scr_ts = classifier_side_scr_ts * 0.02
            classifier_front_scr_ts = classifier_front_scr_ts * 0.02

    if isImagegrab and isClassifier:
        if not isVideo:
            total_lat = ((classifier_scr_ts[:]) - image_nidaq_camtrig0[:])
            total_lat[total_lat < 0] = 0
            print(total_lat)
            print(classifier_side_scr_ts)
            print(classifier_front_scr_ts)
        else:
            total_lat = (classifier_scr_ts[:] - imagegrab_min)
            print('Skipped in both views', np.argwhere(total_lat < 0))
            total_lat[total_lat < 0] = 0
            print(total_lat[2])
            print(imagegrab_min[2])
            print(classifier_scr_ts[2])
            print(classifier_scr_ts[2] - imagegrab_min[2])

    img_camtrig_diff_max_cam0 = max(image_nidaq_camtrig0[1:]-image_nidaq_camtrig0[0:-1])
    img_camtrig_diff_max_cam1 = max(image_nidaq_camtrig1[1:]-image_nidaq_camtrig1[0:-1])
    print('Image camera trigger difference Cam 0',img_camtrig_diff_max_cam0)
    print('Image camera trigger difference Cam 1', img_camtrig_diff_max_cam1)

    plot_raw_latencydata(isImagegrab, isJaaba, isClassifier,
                         imagegrab_process_time_cam0, imagegrab_process_time_cam1,
                         image_nidaq_cam0, image_nidaq_cam1,
                         jaaba_process_time_cam0, jaaba_process_time_cam1,
                         jaaba_nidaqcam0, jaaba_nidaqcam1,
                         image_nidaq_camtrig0, image_nidaq_camtrig1, start_range, end_range)

    plot_sync_btwn_cams(isImagegrab, isJaaba, isClassifier,
                        imagegrab_start_cam0, imagegrab_start_cam1,
                        image_end_time_cam0, image_end_time_cam1,
                        jaaba_start_cam0, jaaba_start_cam1,
                        jaaba_end_cam0, jaaba_end_cam1, start_range, end_range)

    if isClassifier:
        plot_skipped_frames_latencyplot(isClassifier, total_lat,
                                        classifier_scr_view, start_range, end_range,
                                        latency_threshold)
        if isVideo:
            plot_process_scores_latency(isClassifier, isVideo,
                                        classifier_side_scr_ts, classifier_front_scr_ts,
                                        classifier_scr_ts, start_range, end_range)
        else:
            plot_process_scores_latency(isClassifier, isVideo,
                                        classifier_side_scr_ts, classifier_front_scr_ts,
                                        classifier_scr_ts,start_range, end_range)
    plt.show()

def main():
    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if (len(sys.argv) < 14):
        print('Insufficient arguments')
        print('Argument Options:\n' +
              '-filepath to latency data\n' +
              '-filepath to jaaba exp dir\n' +
              '-number of frames\n' +
              '-trial number\n' +
              '-isexpDir\n' +
              '-plot Video data latency plot\n' +
              '-number of camera\n' +
              '-plot imagegrab latency data flag\n' +
              '-plot jaaba plugin latency data flag\n' +
              '-plot end to end classifier latency flag\n' +
              -'latency_threshold for each frame to process\n' +
              -'start range\n' +
              -'end_range\n')
    else:
        filepath = sys.argv[1]
        cls_filepath = sys.argv[2]
        numFrames = np.int(sys.argv[3])
        trial_num = sys.argv[4]
        isexpDir=np.int(sys.argv[5])
        isVideo = np.int(sys.argv[6])
        numCameras = np.int(sys.argv[7])
        isImagegrab = np.int(sys.argv[8])
        isJaaba = np.int(sys.argv[9])
        isClassifier=np.int(sys.argv[10])
        latency_threshold = np.int(sys.argv[11])
        start_range = np.int(sys.argv[12])
        end_range = np.int(sys.argv[13])

        read_latency_rawdata(filepath,cls_filepath,numFrames,
                             trial_num, isexpDir, isVideo,
                             numCameras, isImagegrab,
                             isJaaba, isClassifier, latency_threshold,
                             start_range, end_range)



if __name__ == "__main__":
    main()