import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut
import sys
import matplotlib

def check_output_latency(total_latency, lat_wo_processScores, exp_total_latency,
                         classifier_skips,
                         numFrames, latency_thres):

    count_wrong_skips = 0
    unknown_latency_skip = 0
    known_latency_skip = 0 # skips
    known_latency_skip_id = []
    unknown_latency_skip_id = []
    wrong_skip_ids = []
    wait_thres_delta = 0.3 # wait ime in ms

    for frm_id in range(0,numFrames):
        if(total_latency[frm_id] > latency_thres):
               if(classifier_skips[frm_id] == 3):
                   if(lat_wo_processScores[frm_id]  <= (exp_total_latency[frm_id] + wait_thres_delta)):
                       continue
                   else:
                       count_wrong_skips += 1
                       wrong_skip_ids.append(frm_id)
                       print(lat_wo_processScores[frm_id] - exp_total_latency[frm_id])
               elif ((classifier_skips[frm_id] == 1) or (classifier_skips[frm_id] == 2)):
                   process_scores_outputwrite_lat = total_latency[frm_id] - lat_wo_processScores[frm_id]
                   if (lat_wo_processScores[frm_id] < latency_thres):
                       known_latency_skip += 1
                       known_latency_skip_id.append(frm_id)
                   else:
                       unknown_latency_skip += 1
                       unknown_latency_skip_id.append(frm_id)
    print('Wrong skips in classifier' , count_wrong_skips)
    print('Unknown latency skips', unknown_latency_skip)
    print('Known latency skips', known_latency_skip)
    print('Known latency_skips frame ids', known_latency_skip_id)
    print('Unknown latency_skips frame ids', unknown_latency_skip_id)
    print('wrong latency_skips frame ids', wrong_skip_ids)

def readData(filepath, classifier_file_name, isnidaq, trial_num, numFrames,
             jaaba_nidaq_ts_prefix, jaaba_end_ts_perfix,
             imagegrab_nidaq_ts_prefix, imagegrab_start_ts_prefix,
             start_range, end_range, latency_thres):

    classifier_file = filepath + classifier_file_name + trial_num + '.csv'
    print('IsNidaq', isnidaq)

    if isnidaq:
        jaaba_lat_cam0_file = filepath + jaaba_nidaq_ts_prefix + 'cam0_short_trial' + trial_num + '.csv'
        jaaba_lat_cam1_file = filepath + jaaba_nidaq_ts_prefix  + 'cam1_short_trial' + trial_num + '.csv'
        imagegrab_lat_cam0_file = filepath + imagegrab_nidaq_ts_prefix + 'cam0_short_trial' + trial_num + '.csv'
        imagegrab_lat_cam1_file = filepath + imagegrab_nidaq_ts_prefix + 'cam1_short_trial' + trial_num + '.csv'

    else:
        jaaba_lat_cam0_file = filepath + jaaba_end_ts_perfix + 'cam0_short_trial' + trial_num + '.csv'
        jaaba_lat_cam1_file = filepath + jaaba_end_ts_perfix + 'cam1_short_trial' + trial_num + '.csv'
        imagegrab_lat_cam0_file = filepath + imagegrab_start_ts_prefix + 'cam0_short_trial' + trial_num + '.csv'
        imagegrab_lat_cam1_file = filepath + imagegrab_start_ts_prefix + 'cam1_short_trial' + trial_num + '.csv'

    print(jaaba_lat_cam0_file)
    print(jaaba_lat_cam1_file)
    print(imagegrab_lat_cam0_file)
    print(imagegrab_lat_cam1_file)

    jaaba_lat_cam0 = np.array(numFrames * [0.0])
    jaaba_lat_cam1 = np.array(numFrames * [0.0])
    imagegrab_lat_cam0 = np.array(numFrames * [0.0])
    imagegrab_lat_cam1 = np.array(numFrames * [0.0])
    imagegrab_camtrig_cam0 = np.array(numFrames * [0.0])
    imagegrab_camtrig_cam1 = np.array(numFrames * [0.0])
    jaaba_nidaq_camtrig_cam0 = np.array(numFrames * [0.0])
    jaaba_nidaq_camtrig_cam1 = np.array(numFrames * [0.0])

    classifier_scr_ts = np.array(numFrames * [0.0])
    classifier_scr_ts_side = np.array(numFrames * [0.0])
    classifier_scr_ts_front = np.array(numFrames * [0.0])
    classifier_scr_view = np.array(numFrames * [0])
    classifier_frames_skip = np.array(numFrames * [0])

    if isnidaq:
        ut.readcsvFile_nidaq(imagegrab_lat_cam0_file, imagegrab_camtrig_cam0,imagegrab_lat_cam0, 0.02)
        ut.readcsvFile_nidaq(imagegrab_lat_cam1_file, imagegrab_camtrig_cam1, imagegrab_lat_cam1, 0.02)
        ut.readcsvFile_nidaq(jaaba_lat_cam0_file, jaaba_nidaq_camtrig_cam0, jaaba_lat_cam0, 0.02)
        ut.readcsvFile_nidaq(jaaba_lat_cam1_file, jaaba_nidaq_camtrig_cam1, jaaba_lat_cam1, 0.02)

    else:
        ut.readcsvFile_int(jaaba_lat_cam0_file, jaaba_lat_cam0, 1000, 0)
        ut.readcsvFile_int(jaaba_lat_cam1_file, jaaba_lat_cam1, 1000, 0)
        ut.readcsvFile_int(imagegrab_lat_cam0_file, imagegrab_lat_cam0, 1000, 0)
        ut.readcsvFile_int(imagegrab_lat_cam1_file, imagegrab_lat_cam1, 1000, 0)

    ut.read_score(classifier_file, classifier_scr_ts, 0, 0)
    ut.read_score(classifier_file, classifier_scr_ts_side, 0, 1)
    ut.read_score(classifier_file, classifier_scr_ts_front, 0, 2)
    ut.read_score(classifier_file, classifier_scr_view, 0, 10)

    if isnidaq:
        classifier_scr_ts = classifier_scr_ts * 0.02
        classifier_scr_ts_side = classifier_scr_ts_side * 0.02
        classifier_scr_ts_front  = classifier_scr_ts_front * 0.02
    else:
        classifier_scr_ts = classifier_scr_ts / 1000
        classifier_scr_ts_side = classifier_scr_ts_side / 1000
        classifier_scr_ts_front = classifier_scr_ts_front / 1000

    if isnidaq:
        imagegrab_start_min = np.minimum(imagegrab_camtrig_cam0, imagegrab_camtrig_cam1)
        imagegrab_start_min_fstfrm = np.minimum(imagegrab_camtrig_cam0[0], imagegrab_camtrig_cam1[0])
    else:
        imagegrab_start_min = np.minimum(imagegrab_lat_cam0, imagegrab_lat_cam1) # framewise mins of imagegrab starts
        imagegrab_start_min_fstfrm = np.minimum(imagegrab_lat_cam0[0], imagegrab_lat_cam1[0]) # get the start for first frame

    exp_scr_ts = imagegrab_start_min_fstfrm + latency_thres + (2.5 * np.arange(0, numFrames, 1)) # exp scr lat per frame from fst initial
                                                                                # imagegrab start ts
    exp_imagegrab_ts = imagegrab_start_min_fstfrm + (2.5 * np.arange(0,numFrames,1))

    # predicted total latency
    total_lat = classifier_scr_ts - imagegrab_start_min
    total_lat[total_lat < 0] = 0 ## assign zero latency to frames in classifier skipped in both views

    #exp total latency
    total_pred_lat = exp_scr_ts - exp_imagegrab_ts

    #total latency without the process scores thread write to serial
    total_lat_wo_processScores_lat = np.maximum(classifier_scr_ts_side,classifier_scr_ts_front) - imagegrab_start_min
    total_lat_wo_processScores_lat[total_lat_wo_processScores_lat < 0] = 0 # assign zero latency to frames
                                                                           # in classifier skipped in both views

    # skipped frames
    ind_skips = np.argwhere((classifier_scr_view == 2) | (classifier_scr_view == 1))
    classifier_frames_skip[ind_skips] = 3

    # difference in predicted Vs expected total latency
    diff_predVsexp_total_lat = classifier_scr_ts - exp_scr_ts
    diff_predVsexp_total_lat[diff_predVsexp_total_lat < -10] = -1 # assign random negative latency to frames
                                                                   # in classifier skipped in both views

    check_output_latency(total_lat, total_lat_wo_processScores_lat, total_pred_lat,
                         classifier_scr_view, numFrames, latency_thres)

    plot_total_lat(classifier_scr_ts, total_lat, total_pred_lat,
                   total_lat_wo_processScores_lat, classifier_frames_skip,
                   diff_predVsexp_total_lat, numFrames, start_range, end_range, 6)

def plot_total_lat(classifier_scr_ts,total_lat,  total_lat_pred,
                   total_lat_wo_processScores_lat, classifier_frames_skip,
                   diff_predVsexp_totollat, numFrames,
                   start_range, end_range, latency_threshold):

    range = np.arange(start_range, end_range)

    fig4 = plt.figure(figsize=(10, 10))
    fig4.tight_layout()
    fontsize = 15

    plt.plot(total_lat[range],linestyle='--',marker='.',color='brown', alpha=0.2)
    plt.plot(total_lat_pred[range],linestyle='--',marker='*',color='pink', alpha=0.6)
    plt.plot(total_lat_wo_processScores_lat[range], linestyle='--', marker='o', color='orange', alpha=0.4)
    plt.plot(classifier_frames_skip[range], '.', color='red', alpha=0.8)
    plt.plot(diff_predVsexp_totollat[range], linestyle = '--', marker='>', alpha=1.0)
    plt.yticks(np.arange(0,10,1))

def main():

    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if (len(sys.argv) < 13):
        print('Insufficient arguments')
        print('Argument Options:\n' + '')
        return
    else:
        filepath = sys.argv[1]
        classifier_file_name = sys.argv[2]
        trial_num = sys.argv[3]
        isnidaq = np.int(sys.argv[4])
        numFrames = np.int(sys.argv[5])
        jaaba_nidaq_ts_prefix = sys.argv[6]
        jaaba_end_ts_prefix = sys.argv[7]
        imagegrab_nidaq_ts_prefix = sys.argv[8]
        imagegrab_start_ts_prefix = sys.argv[9]
        start_range = np.int(sys.argv[10])
        end_range = np.int(sys.argv[11])
        latency_threshold = np.float(sys.argv[12])

    readData(filepath, classifier_file_name, isnidaq, trial_num, numFrames,
            jaaba_nidaq_ts_prefix, jaaba_end_ts_prefix,
            imagegrab_nidaq_ts_prefix, imagegrab_start_ts_prefix, start_range, end_range,
             latency_threshold)
    plt.show()

if __name__ == "__main__":
    main()