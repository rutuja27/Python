import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut

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
    print(arr)
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


def main():

#files
    filepath = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/2c5ba_9_8_2022/'

    trial_type = '1'
    numCameras=2
    isVideo=0

    if isVideo:
        numFrames = 2498
    else:
        numFrames = 100000
    imagegrab_file_cam0 = filepath + 'imagegrab_start_timecam0_short_trial' + trial_type + '.csv'
    img_proc_cam0 = filepath + 'imagegrab_process_timecam0_short_trial' + trial_type + '.csv'
    img_skipped_cam0 = filepath + 'imagegrab_skipped_framescam0_short_trial' + trial_type + '.csv'
    img_end_time_cam0 = filepath + 'imagegrab_end_timecam0_short_trial' + trial_type + '.csv'
    imagegrab_nidaq_cam0 = filepath + 'imagegrab_nidaqcam0_short_trial' + trial_type + '.csv'
    image_nidaqThres_cam0 = filepath + 'imagegrab_nidaq_threscam0_short_trial' + trial_type + '.csv'

    if numCameras==2:
      imagegrab_file_cam1 = filepath + 'imagegrab_start_timecam1_short_trial' + trial_type + '.csv'
      img_proc_cam1 = filepath + 'imagegrab_process_timecam1_short_trial' + trial_type + '.csv'
      img_skipped_cam1 = filepath + 'imagegrab_skipped_framescam1_short_trial' + trial_type + '.csv'
      img_end_time_cam1 = filepath + 'imagegrab_end_timecam1_short_trial' + trial_type + '.csv'
      imagegrab_nidaq_cam1 = filepath + 'imagegrab_nidaqcam1_short_trial' + trial_type + '.csv'
      image_nidaqThres_cam1 = filepath + 'imagegrab_nidaq_threscam1_short_trial' + trial_type + '.csv'

    jaaba_strtfile_cam0 = filepath + 'jaaba_plugin_start_timecam0_short_trial' + trial_type + '.csv'
    jaaba_strtfile_cam1 = filepath + 'jaaba_plugin_start_timecam1_short_trial' + trial_type + '.csv'
    jaaba_endfile_cam0 = filepath + 'jaaba_plugin_end_time_cam0_short_trial' + trial_type + '.csv'
    jaaba_endfile_cam1 = filepath + 'jaaba_plugin_end_time_cam1_short_trial' + trial_type + '.csv'
    jaaba_proc_cam0 = filepath + 'jaaba_plugin_process_timecam0_short_trial' + trial_type + '.csv'
    jaaba_proc_cam1 = filepath + 'jaaba_plugin_process_timecam1_short_trial' + trial_type + '.csv'

    jaaba_nidaq_cam0 = filepath + 'jaaba_plugin_nidaqcam0_short_trial' + trial_type + '.csv'
    jaaba_nidaq_cam1 = filepath + 'jaaba_plugin_nidaqcam1_short_trial' + trial_type + '.csv'
    jaaba_nidaqThres_cam0 = filepath + 'jaaba_plugin_nidaq_threscam0_short_trial' + trial_type + '.csv'
    jaaba_nidaqThres_cam1 = filepath + 'jaaba_plugin_nidaq_threscam1_short_trial' + trial_type + '.csv'
    jaaba_curTimecam0_file = filepath + 'jaaba_plugin_cur_timecam0_short_trial' + trial_type + '.csv'
    jaaba_curTimecam1_file = filepath + 'jaaba_plugin_cur_timecam1_short_trial' + trial_type + '.csv'

    classifier_scr_file = filepath + 'classifier_trial' + trial_type + '.csv'
    classifier_scr_side_file = filepath + 'classifier_scr_side' + trial_type + '.csv'
    classifier_scr_front_file = filepath + 'classifier_scr_front' + trial_type + '.csv'
    classifier_scr_file_gt = filepath + 'lift_classifier.csv'#'C:/Users/27rut/BIAS/misc/classifier_trials/classifier_scores_new/lift_classifier_side.csv'


    imagegrab_process_time_cam0 = np.array(numFrames*[0.0])
    imagegrab_process_time_cam1 = np.array(numFrames * [0.0])
    imagegrab_start_cam0 = np.array(numFrames * [0.0])
    imagegrab_start_cam1 = np.array(numFrames * [0.0])
    image_nidaq_cam0 = np.array(numFrames * [0.0])
    image_nidaq_cam1 = np.array(numFrames * [0.0])
    image_nidaq_camtrig0 = np.array(numFrames * [0.0])
    image_nidaq_camtrig1 = np.array(numFrames * [0.0])
    image_nidaqThres0 = np.array(numFrames* [0.0])
    image_nidaqThres1 = np.array(numFrames* [0.0])
    imagegrab_skipped_cam0 = np.array(numFrames * [0.0])
    imagegrab_skipped_cam1 = np.array(numFrames * [0.0])
    image_end_time_cam0 = np.array(numFrames * [0.0])
    image_end_time_cam1 = np.array(numFrames * [0.0])

    jaaba_process_time_cam0 = np.array(numFrames*[0.0])
    jaaba_process_time_cam1 = np.array(numFrames * [0.0])
    jaaba_start_cam0 = np.array(numFrames*[0.0])
    jaaba_start_cam1 = np.array(numFrames*[0.0])
    jaaba_end_cam0 = np.array(numFrames*[0.0])
    jaaba_end_cam1 = np.array(numFrames*[0.0])
    jaaba_curTimecam0 = np.array(numFrames*[0.0])
    jaaba_curTimecam1 = np.array(numFrames*[0.0])
    jaaba_expTimecam0 = np.array(numFrames * [0.0])
    jaaba_expTimecam1 = np.array(numFrames * [0.0])
    jaaba_nidaq_camtrig0 = np.array(numFrames * [0.0])
    jaaba_nidaq_camtrig1 = np.array(numFrames * [0.0])
    jaaba_nidaqThres0 = np.array(numFrames * [0.0])
    jaaba_nidaqThres1 = np.array(numFrames * [0.0])
    jaaba_nidaqcam0 = np.array(numFrames * [0.0])
    jaaba_nidaqcam1 = np.array(numFrames * [0.0])

    cam0_total = np.array(numFrames*[0.0])
    cam1_total = np.array(numFrames * [0.0])

    classifier_scr = np.array((numFrames-1)*[0.0])
    classifier_scr_gt = np.array((numFrames-1)*[0.0])
    classifier_scr_ts = np.array((numFrames-1)*[0.0])
    classifier_side_scr_ts = np.array((numFrames-1)*[0.0])
    classifier_front_scr_ts = np.array((numFrames-1)*[0.0])

    # Read data from csv
    ut.readcsvFile_int(imagegrab_file_cam0, imagegrab_start_cam0, 1,0)
    ut.readcsvFile_int(img_proc_cam0, imagegrab_process_time_cam0,1,0)
    ut.readcsvFile_nidaq(imagegrab_nidaq_cam0, image_nidaq_camtrig0, image_nidaq_cam0)
    ut.readcsvFile_int(image_nidaqThres_cam0, image_nidaqThres0,1,0)
    ut.readcsvFile_int(img_end_time_cam0, image_end_time_cam0,1,0)
    ut.readcsvFile_nidaq(jaaba_nidaq_cam0, jaaba_nidaq_camtrig0, jaaba_nidaqcam0)
    #ut.readcsvFile_int(jaaba_nidaqThres_cam0, jaaba_nidaqThres0,1)
    #ut.readcsvFile_int(jaaba_curTimecam0_file, jaaba_curTimecam0,1)

    if numCameras==2:
      ut.readcsvFile_int(imagegrab_file_cam1, imagegrab_start_cam1,1,0)
      ut.readcsvFile_int(img_proc_cam1, imagegrab_process_time_cam1,1,0)
      ut.readcsvFile_int(img_end_time_cam1, image_end_time_cam1, 1,0)
      ut.readcsvFile_nidaq(imagegrab_nidaq_cam1, image_nidaq_camtrig1, image_nidaq_cam1)
      ut.readcsvFile_int(image_nidaqThres_cam1, image_nidaqThres1, 1,0)
      ut.readcsvFile_nidaq(jaaba_nidaq_cam1, jaaba_nidaq_camtrig1, jaaba_nidaqcam1)
      #ut.readcsvFile_int(jaaba_nidaqThres_cam1, jaaba_nidaqThres1 ,1)
      #ut.readcsvFile_int(jaaba_curTimecam1_file, jaaba_curTimecam1, 1)

    #ut.readcsvFile_int(img_skipped_cam0, imagegrab_skipped_cam0,1,0)
    #ut.readcsvFile_int(img_skipped_cam1, imagegrab_skipped_cam1,1,0)

    #ut.readcsvFile_int(jaaba_proc_cam0, jaaba_process_time_cam0, 1000,0)
    #ut.readcsvFile_int(jaaba_proc_cam1, jaaba_process_time_cam1, 1000,0)
    #ut.readcsvFile_int(jaaba_strtfile_cam0, jaaba_start_cam0, 1)
    #ut.readcsvFile_int(jaaba_strtfile_cam1, jaaba_start_cam1, 1)
    #ut.readcsvFile_int(jaaba_endfile_cam0, jaaba_end_cam0, 1000)
    #ut.readcsvFile_int(jaaba_endfile_cam1, jaaba_end_cam1, 1000)

    ## read ts from score files
    #ut.read_score(classifier_scr_file, classifier_side_scr_ts, 0, 1)
    #ut.read_score(classifier_scr_file, classifier_front_scr_ts, 0, 2)
    ut.read_score(classifier_scr_file, classifier_scr_ts, 0, 0)

    image_nidaq_cam0 = image_nidaq_cam0 - image_nidaq_camtrig0
    image_nidaq_cam1 = image_nidaq_cam1 - image_nidaq_camtrig0
    jaaba_nidaqcam0 = jaaba_nidaqcam0 - image_nidaq_camtrig0
    jaaba_nidaqcam1 = jaaba_nidaqcam1 - image_nidaq_camtrig0
    jaaba_nidaqcam0[jaaba_nidaqcam0 < 0] = 0
    jaaba_nidaqcam1[jaaba_nidaqcam1 < 0] = 0
    print('Imagegrab cam trigger' , image_nidaq_camtrig0)

    jaaba2imaggrab_cam0_pc_time = jaaba_start_cam0 - imagegrab_start_cam0
    jaaba2imaggrab_cam1_pc_time = jaaba_start_cam1 - imagegrab_start_cam1
    dif_jaaba_start = abs(jaaba_start_cam0 - jaaba_start_cam1)
    dif_jaaba_end = abs(jaaba_end_cam0 - jaaba_end_cam1)
    dif_imagegrab = abs(imagegrab_start_cam0 - imagegrab_start_cam1);
    img_min = np.minimum(imagegrab_start_cam0,imagegrab_start_cam1);
    img_min_nidaq = np.minimum(image_nidaq_cam0, image_nidaq_cam1);

    jaaba2imagegrab_cam0_nidaq_time = ((jaaba_curTimecam0/1000)-jaaba_nidaq_camtrig0) - image_nidaq_cam0
    jaaba2imagegrab_cam1_nidaq_time = ((jaaba_curTimecam1/1000)-jaaba_nidaq_camtrig1) - image_nidaq_cam1
    jaaba_pred_curTime0 = np.maximum(image_nidaq_cam0[1:],jaaba_process_time_cam0[:-1]);
    jaaba_pred_curTime1 = np.maximum(image_nidaq_cam1[1:],jaaba_process_time_cam1[:-1]);

    cam0_total = (jaaba_end_cam0 - imagegrab_start_cam0)/1000
    cam1_total = (jaaba_end_cam1 - imagegrab_start_cam1)/1000

    if not isVideo:
        total_lat = (classifier_scr_ts[:-1]*0.02 - image_nidaq_camtrig0[1:-1])
        total_lat[total_lat < 0] = 0
    else:
        total_lat = (classifier_scr_ts[1:]/1000) - np.minimum(imagegrab_start_cam1[1:-1]/1000,imagegrab_start_cam0[1:-1]/1000)
        total_lat[total_lat < 0] = 0
        print('Skipped in both views' ,np.argwhere(total_lat < 0))

    print(np.sum(total_lat)/(numFrames-1))
    print(np.sum(total_lat > 6))

    for j in range(0,numFrames):
      jaaba_expTimecam0[j] = imagegrab_start_cam0[0] + (2500*(j+1)) + 2000
      jaaba_expTimecam1[j] = imagegrab_start_cam1[0] + (2500*(j+1)) + 2000

    print(np.argwhere(imagegrab_skipped_cam0 == 1))
    print(np.argwhere(imagegrab_skipped_cam1 == 1))

    plt.figure(1)
    ax1=plt.gca()
    #ax1.plot(dif_jaaba_end[:-1], '.', alpha=0.3, color='magenta')
    #ax1.plot(jaaba_process_time_cam0[:-1], '.',alpha=0.6,color='red')
    #ax1.plot(jaaba_process_time_cam1[:-1], '.',alpha=0.6,color='blue')
    #ax1.plot(jaaba2imagegrab_cam0_nidaq_time[:], '.', color='green', alpha=0.5)
    #ax1.plot(jaaba2imagegrab_cam1_nidaq_time[:], '.', color='orange', alpha=0.5)
    #ax1.plot(jaaba2imaggrab_cam0_pc_time[:]/1000, '.', color='green', alpha=0.2)
    #ax1.plot(jaaba2imaggrab_cam1_pc_time[:]/1000, '.', color='orange', alpha=0.5)
    #ax1.plot((jaaba_curTimecam0[:]-jaaba_nidaqThres0[:])/1000,'.',color='cyan', alpha=0.5)
    #ax1.plot((jaaba_curTimecam1[:]-jaaba_nidaqThres1[:])/1000,'.',color='magenta', alpha=0.2)

    #ax1.plot(jaaba_pred_curTime0[:],'.',color='magenta', alpha=0.5)
    #ax1.plot(jaaba_pred_curTime1[:] ,'.', color='cyan', alpha=0.5)
    #ax1.plot(jaaba_nidaqThres0[:], '.', alpha=0.5, color='blue')
    #ax1.plot(jaaba_nidaqThres1[:], '.', alpha=0.5, color='red')
    ax1.plot(jaaba_nidaqcam0[:-1], '.', alpha=0.1, color='orange')
    ax1.plot(jaaba_nidaqcam1[:-1], '.', alpha=0.2, color='green')
    #ax1.plot(np.ones(numFrames)*3)
    #ax1.plot(np.ones(numFrames)*4.0)
    ax1.set_yticks(np.arange(-1,20,1))
    plt.title('BIAS JAABA processing time from video')
    plt.xlabel('Frames')
    plt.ylabel('Latency in ms')
    plt.legend(['Side View processing time', 'Front View processing time'], fontsize=8)
    #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/jaaba_enetoend_latency_woinitlat.jpg')


    plt.figure(2)
    ax2 = plt.gca()

    #print('model',imagegrab_start_cam0[0:-1]-imagegrab_process_time_cam0[1:])
    #ax2.plot(dif_imagegrab[:], '.', color='magenta',alpha=1)
    #ax2.plot(image_end_time_cam0[1:]-image_end_time_cam0[0:-1] , '.', color='green', alpha=0.8)
    #ax2.plot((image_end_time_cam0[:-1]-imagegrab_start_cam0[:-1])/1000 , '.', color='orange',alpha=0.8)
    #ax2.plot((image_end_time_cam1[:-1]-imagegrab_start_cam1[:-1])/1000, '.', color='green',alpha=0.8)
    #ax2.plot(imagegrab_start_cam0[1:]-imagegrab_start_cam0[0:-1], '.', color='red',alpha=0.8)
    #ax2.plot(imagegrab_start_cam1[1:]-imagegrab_start_cam1[0:-1], '.', color='green',alpha=0.8)
    ax2.plot(abs(imagegrab_process_time_cam0[:]), '.', color='orange',alpha=0.8)
    ax2.plot(abs(imagegrab_process_time_cam1[:] ), '.', color='yellow',alpha=0.8)
    #ax2.plot((imagegrab_process_time_cam1[:-1]-imagegrab_process_time_cam0[:-1])/1000,'.', color='blue', alpha=0.2)
    #ax2.plot(imagegrab_start_cam1[:] - imagegrab_start_cam0[:], '.', color='red', alpha=0.5)
    #ax2.plot(imagegrab_start_cam1[:]/1000, '.', color='blue', alpha=0.3)
    #ax2.plot(image_nidaq_cam0[:-1], '.', color='red', alpha=0.5)
    #ax2.plot(image_nidaq_cam1[:-1], '.', color='blue', alpha=0.5)
    #ax2.plot(np.ones(numFrames)*3)
    #ax2.plot(np.ones(numFrames)*4)
    #ax2.plot(image_nidaqThres0[550:560], '.', color='red', alpha=0.5)
    #ax2.plot(image_nidaqThres1[:], '.', color='blue', alpha=0.5)
    #ax2.plot(imagegrab_skipped_cam0[:], '.', color='red', alpha=0.5)
    #ax2.plot(imagegrab_skipped_cam1[:], '.', color='blue', alpha=0.5)

    #ax2.set_yticks(np.arange(-1,20,1))
    plt.title('Image Grabber latencies video read')
    plt.xlabel('Frames')
    plt.ylabel('Imagegrab Latencies in ms')
    plt.legend(['Cam 0' , 'Cam 1'], fontsize=8)
    #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/biasjaaba_imagegrablat_vid_woskip_no_preview_nojaaba_noimagedispatch_noupdateDisplay_multicamera_thread_affinityOG.jpg')

    '''plt.figure()
    ax3 = plt.gca()
    #ax3.plot(cam0_total[:-1], '.', color='green', alpha=0.3)
    #ax3.plot(cam1_total[:-1], '.', color='orange', alpha=0.3)
    #ax3.set_yticks(np.arange(-1,20,1))
    #ax3.plot(abs(cam1_total-cam0_total), color='pink' )
    #ax3.plot(classifier_scr_gt[:], color='cyan',alpha=0.3)
    #plt.title('Comparison of Lift Classifier Score')
    #plt.xlabel('Frames')
    #plt.ylabel('JAABA Classifier Scores')
    #plt.legend(['BIAS JAABA score predictions', 'JAABA Classifier Demo-GT'], fontsize=8)
    #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/comp_biasjaaba_predVsgt_woskip.pdf')'''

    plt.figure()
    ax4 = plt.gca()
    ax4.plot(total_lat[:],'.')
    ax4.plot(np.ones(numFrames)*5)
    ax4.set_yticks(np.arange(-1,20,1))
    plt.show()

if __name__ == "__main__":
    main()