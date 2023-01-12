import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut
import lat_defs as ld

def main():

    data_dir = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/2c5ba_9_8_2022/'
    no_of_trials = 5
    numFrames = 100000

    imagegrab_data_cam0 = ld.LatencyData(np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]),
                                    np.array(no_of_trials * [numFrames * [0.0]]))

    imagegrab_data_cam1 = ld.LatencyData(np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]),
                                    np.array(no_of_trials * [numFrames * [0.0]]))

    jaaba_data_cam0 = ld.LatencyData(np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]),
                                    np.array(no_of_trials * [numFrames * [0.0]]))

    jaaba_data_cam1 = ld.LatencyData(np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]), \
                                    np.array(no_of_trials * [numFrames * [0.0]]),
                                    np.array(no_of_trials * [numFrames * [0.0]]))

    classifier_scores = np.array(no_of_trials*[ld.Scores(np.array(numFrames * [0.0]), \
                                    np.array(numFrames * [0.0]), \
                                    np.array(numFrames * [0.0]), \
                                    np.array(numFrames * [0.0]), \
                                    np.array(numFrames * [0]), \
                                    np.array(numFrames * [0]))])

    trial_type = ''
    image_grab_cam0_avg=0
    image_grab_cam1_avg=0
    jaaba_cam0_avg=0
    jaaba_cam1_avg=0


    for i in range(0,no_of_trials):

        trial_type = str(i+1)
        cam_id = '0'

        imagegrab_file = data_dir + 'imagegrab_nidaqcam' + cam_id +'_short_trial' + trial_type + '.csv'
        jaaba_file = data_dir + 'jaaba_plugin_nidaqcam' + cam_id + '_short_trial' + trial_type + '.csv'
        scores_file = data_dir + 'classifier_trial' + trial_type + '.csv'

        ut.readcsvFile_nidaq(imagegrab_file, imagegrab_data_cam0.lat_camtrig[i], imagegrab_data_cam0.lat_nidaq[i])
        ut.readcsvFile_nidaq(jaaba_file, jaaba_data_cam0.lat_camtrig[i], jaaba_data_cam0.lat_nidaq[i])

        cam_id='1'
        imagegrab_file = data_dir + 'imagegrab_nidaqcam' + cam_id +'_short_trial' + trial_type + '.csv'
        jaaba_file = data_dir + 'jaaba_plugin_nidaqcam' + cam_id + '_short_trial' + trial_type + '.csv'
        scores_file = data_dir + 'classifier_trial' + trial_type + '.csv'

        ut.readcsvFile_nidaq(imagegrab_file, imagegrab_data_cam1.lat_camtrig[i], imagegrab_data_cam1.lat_nidaq[i])
        ut.readcsvFile_nidaq(jaaba_file, jaaba_data_cam1.lat_camtrig[i],
                             jaaba_data_cam1.lat_nidaq[i])
        ut.readScoreData(scores_file, classifier_scores[i], 0) # last argument if gt or not flag

        # calculating latencies
        jaaba_data_cam0.lat_process_time[i] = jaaba_data_cam0.lat_nidaq[i][:] - imagegrab_data_cam0.lat_nidaq[i][:]
        jaaba_data_cam1.lat_process_time[i] = jaaba_data_cam1.lat_nidaq[i][:] - imagegrab_data_cam1.lat_nidaq[i][:]
        imagegrab_data_cam0.lat_process_time[i] = imagegrab_data_cam0.lat_nidaq[i] - \
                                                    imagegrab_data_cam0.lat_camtrig[i]
        imagegrab_data_cam1.lat_process_time[i] = imagegrab_data_cam1.lat_nidaq[i] - \
                                                      imagegrab_data_cam1.lat_camtrig[i]
        classifier_scores[i].score_ts[:-1] = ((classifier_scores[i-1].score_ts[:-1])*0.02) - \
                                             np.maximum(jaaba_data_cam0.lat_nidaq[i][1:],
                                             jaaba_data_cam1.lat_nidaq[i][1:])

        #compute sum to calculate averages
        image_grab_cam0_avg += np.sum(imagegrab_data_cam0.lat_process_time[i][:-1])
        image_grab_cam1_avg += np.sum(imagegrab_data_cam1.lat_process_time[i][:-1])

    perc = [50,75,90,95,98]
    image_grab_cam0_percentile = np.percentile(imagegrab_data_cam0.lat_process_time, perc)
    image_grab_cam1_percentile = np.percentile(imagegrab_data_cam1.lat_process_time, perc)
    jaaba_cam0_percentile = np.percentile(jaaba_data_cam0.lat_process_time, perc)
    jaaba_cam1_percentile = np.percentile(jaaba_data_cam1.lat_process_time, perc)

    print('Percentile Image grab Cam 0',image_grab_cam0_percentile)
    print('Percentile Image grab Cam 1',image_grab_cam1_percentile)
    print('Percentile Jaaba Cam 0', jaaba_cam0_percentile)
    print('Percentile Jaaba Cam 1', jaaba_cam1_percentile)

    #computing averages
    camera_cap_avg = 0.2
    image_grab_cam0_avg = image_grab_cam0_avg/(numFrames*no_of_trials)
    image_grab_cam1_avg = image_grab_cam1_avg/(numFrames*no_of_trials)
    jaaba_cam0_avg = np.sum(np.sum(jaaba_data_cam0.lat_process_time))/(numFrames*no_of_trials)
    jaaba_cam1_avg = np.sum(np.sum(jaaba_data_cam1.lat_process_time))/(numFrames*no_of_trials)
    print('Average Image grab latency Cam 0',image_grab_cam0_avg)
    print('Average Image grab latency Cam 1',image_grab_cam1_avg)
    print('Average Jaaba latency Cam 0',jaaba_cam0_avg)
    print('Average Jaaba latency Cam 1',jaaba_cam1_avg)

    classifier_scr_ts = []
    #computing classifier score ts averages
    scr_ts_avg=0
    for i in range(0, no_of_trials):
        classifier_scr_ts.append(classifier_scores[i].score_ts[classifier_scores[0].score_ts > 0][:-1])
        scr_ts_avg += np.sum(classifier_scores[i].score_ts[classifier_scores[0].score_ts > 0][:-1])
    scr_ts_avg = scr_ts_avg/(numFrames*no_of_trials)
    print('Scr ts avg', scr_ts_avg)
    ord_classifier_scr_ts = [val for x in classifier_scr_ts for val in x]
    scr_ts_percentle = np.percentile(ord_classifier_scr_ts, perc)
    print('Percentile scores collection', scr_ts_percentle)

    plt.figure(figsize=(10,4))
    ax1=plt.gca()
    height=0.5
    alpha1=0.3
    alpha2=0.6
    ax1.barh(['Cam 0'], camera_cap_avg, height=height, color='blue', alpha=alpha1)
    ax1.barh(['Cam 0'], image_grab_cam0_avg, left=camera_cap_avg,height=height, color='green', alpha=alpha1)
    ax1.barh(['Cam 0'], jaaba_cam0_avg, left=image_grab_cam0_avg+camera_cap_avg, height=height,color='brown',alpha=alpha1)
    ax1.barh(['Cam 0'], scr_ts_avg, left=image_grab_cam0_avg+jaaba_cam0_avg+camera_cap_avg,height=height, color='orange',\
             alpha=alpha1)
    ax1.barh(['Cam 1'], camera_cap_avg, height=height, color='blue', alpha=alpha2)
    ax1.barh(['Cam 1'], image_grab_cam1_avg, left=camera_cap_avg, height=height, color='green', alpha=alpha2)
    ax1.barh(['Cam 1'], jaaba_cam1_avg, left=image_grab_cam1_avg + camera_cap_avg, height=height, color='brown',
             alpha=alpha2)
    ax1.barh(['Cam 1'], scr_ts_avg, left=image_grab_cam1_avg + jaaba_cam1_avg + camera_cap_avg, height=height,
             color='orange', alpha=alpha2)
    plt.xticks(np.arange(0,6.5,0.5))
    plt.xlabel('Time ms', fontsize=10)
    plt.title('Average Latency of the BIAS-JAABA pipeline')
    plt.savefig('C:/Users/27rut/BIAS/misc/janelia_symposium_figures/avg_latency_plot.jpg')

    plt.figure(figsize=(10, 4))
    ax2=plt.gca()
    height = 0.5
    alpha1 = 0.3
    alpha2 = 0.6
    for i in range(len(perc)-1,-1,-1):

        ylabel = str(perc[i])
        ax2.barh([ylabel], camera_cap_avg, height=height, color='blue', alpha=alpha1)
        ax2.barh([ylabel], image_grab_cam0_percentile[i], left=camera_cap_avg, height=height, color='green', alpha=alpha1)
        ax2.barh([ylabel], jaaba_cam0_percentile[i], left=image_grab_cam0_percentile[i] + camera_cap_avg, height=height, color='brown',
                 alpha=alpha1)
        ax2.barh([ylabel], ord_classifier_scr_ts[i], left=image_grab_cam0_percentile[i] + jaaba_cam0_percentile[i] + camera_cap_avg, height=height,
                 color='orange', alpha=alpha1)

    plt.xticks(np.arange(0, 10, 0.5))

    plt.show()



if __name__ == "__main__":
    main()