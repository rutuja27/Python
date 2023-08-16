import numpy as np
import matplotlib.pyplot as plt
import utils as ut

def assign_ramp(arr):

    len_arr = len(arr)
    scaling_factor = 1/len_arr

    for i in range(0,len_arr):
        arr[i] = ((i+1)*scaling_factor)

def not_intersection(lst1, lst2):
    return list(set(lst1) ^ set(lst2))

def main():

    output_dir = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/ce521_3_20_2023/'
    output_dir_gt = 'C:/Users/27rut/BIAS/misc/classifier_trials/classifier_scores/'
    trial_type = '1'
    no_of_trials = 1
    numFrames = 2498

    imagegrab_skipped_frames_cam0 = 'imagegrab_skipped_framescam0_short_trial'
    imagegrab_skipped_frames_cam1 = 'imagegrab_skipped_framescam1_short_trial'
    scores_prefix = 'classifier_trial'
    scores_prefix_gt = 'lift_classifier.csv'
    scores_side_prefix_gt = 'lift_classifier_side.csv'
    scores_front_prefix_gt = 'lift_classifier_front.csv'

    skips_side_gt = np.array(no_of_trials * [numFrames * [0.0]]) ## frames skipped in imagegrab side by the user
    skips_front_gt = np.array(no_of_trials * [numFrames * [0.0]]) ## frames skipped in imagegrab front by the user
    scores_jaaba = np.array(no_of_trials * [numFrames * [0.0]])      ## scores predicted
    scores_jaaba_view = np.array(no_of_trials * [numFrames * [0.0]]) ## view contributing to the score
    scores_gt = np.array(no_of_trials * [numFrames * [0.0]])         ## ground truth score
    scores_side_gt = np.array(no_of_trials * [numFrames * [0.0]])    ## gt for only side view
    scores_front_gt = np.array(no_of_trials * [numFrames * [0.0]])   ##  gt for only front view

    ## processScores thread
    front_skips = []
    side_skips = []
    both_skips = []
    correct_frames = []

    ## imagegrab thread
    side_skip_imagegrab_gt = []
    front_skip_imagegrab_gt = []

    for i in range(0,no_of_trials):

        trial_type = str(i+1)
        classifier_score_file = output_dir + scores_prefix + trial_type + '.csv.'
        classifier_score_file_gt = output_dir_gt + scores_prefix_gt
        classifier_score_side_file_gt = output_dir_gt + scores_side_prefix_gt
        classifier_score_front_file_gt = output_dir_gt + scores_front_prefix_gt
        imagegrab_skipped_frames_cam0_file = output_dir  + imagegrab_skipped_frames_cam0 + trial_type + '.csv'
        imagegrab_skipped_frames_cam1_file = output_dir + imagegrab_skipped_frames_cam1 + trial_type + '.csv'
        #nidaq_thres_jaaba_cam0_file = output_dir + nidaq_prefix_jaabathres_cam0 + trial_type + '.csv'
        #nidaq_thres_jaaba_cam1_file = output_dir + nidaq_prefix_jaabathres_cam1 + trial_type + '.csv'

        #ut.readcsvFile_int(nidaq_thres_jaaba_cam0_file, nidaq_thres_jaaba_cam0[i], 1)
        #ut.readcsvFile_int(nidaq_thres_jaaba_cam1_file, nidaq_thres_jaaba_cam1[i], 1)
        ut.readcsvFile_int(imagegrab_skipped_frames_cam0_file, skips_side_gt[i], 1,0)
        ut.readcsvFile_int(imagegrab_skipped_frames_cam1_file, skips_front_gt[i], 1,0)
        ut.read_score(classifier_score_file,scores_jaaba[i], 0, 3)
        ut.read_score(classifier_score_file, scores_jaaba_view[i], 0, 5)
        ut.read_score(classifier_score_file_gt, scores_gt[i], 1, 1)
        ut.read_score(classifier_score_side_file_gt, scores_side_gt[i], 1, 1)
        ut.read_score(classifier_score_front_file_gt, scores_front_gt[i], 1, 1)

        front_skips.append(np.argwhere(scores_jaaba_view[i] == 1).flatten()) # 1 meaning front has been skipped
                                                                                  # reporting score from side
        side_skips.append(np.argwhere(scores_jaaba_view[i] == 2).flatten()) # 2 meaning side has been skipped
                                                                                 # and reporting score from front
        both_skips.append(np.argwhere(scores_jaaba_view[i] == -1).flatten()) # skipped in both views
        correct_frames.append(np.argwhere(scores_jaaba_view[i] == 3).flatten()) # both views correct

        side_skip_imagegrab_gt.append(np.argwhere(skips_side_gt[i]==1).flatten())
        front_skip_imagegrab_gt.append(np.argwhere(skips_front_gt[i]==1).flatten())

        indexes = np.arange(0,numFrames-1,1)
        front_scores = np.zeros(numFrames)
        side_scores = np.zeros(numFrames)
        combined_scores  = np.zeros(numFrames)
        assign_ramp(combined_scores)
        assign_ramp(front_scores)
        assign_ramp(side_scores)

        not_front_skips = not_intersection(np.arange(0,numFrames-1), front_skips[i])
        not_side_skips = not_intersection(np.arange(0,numFrames-1), side_skips[i])
        not_correct_frames = not_intersection(np.arange(0,numFrames-1), correct_frames[i])

        front_scores[not_front_skips] = 0
        front_scores = front_scores[:-1]
        side_scores[not_side_skips] = 0
        side_scores = side_scores[:-1]
        combined_scores[not_correct_frames] = 0
        combined_scores = combined_scores[:-1]

        side_imagegrab = np.zeros(numFrames)
        front_imagegrab = np.zeros(numFrames)
        combined_imagegrab = np.zeros(numFrames)
        assign_ramp(combined_imagegrab)
        assign_ramp(front_imagegrab)
        assign_ramp(side_imagegrab)

        not_front_skip_imagegrab = not_intersection(np.arange(0,numFrames-1), front_skip_imagegrab_gt[i])
        not_side_skip_imagegrab = not_intersection(np.arange(0,numFrames-1), side_skip_imagegrab_gt[i])

        combined_imagegrab[front_skip_imagegrab_gt[i]] = 0.0
        combined_imagegrab[side_skip_imagegrab_gt[i]] = 0.0
        front_imagegrab[not_front_skip_imagegrab] = 0.0
        side_imagegrab[not_side_skip_imagegrab] = 0.0
        side_imagegrab = side_imagegrab[:-1]
        front_imagegrab = front_imagegrab[:-1]
        combined_imagegrab = combined_imagegrab[:-1]
        print(front_skip_imagegrab_gt[i])
        print(side_skip_imagegrab_gt[i])
        print(side_imagegrab)

        '''plt.figure(figsize=(30, 10))
        ax1 = plt.gca()
        height = 0.8
        alpha = 1
        ax1.bar(list(indexes),list(front_scores), color='blue', alpha=alpha, width=3)
        ax1.bar(list(indexes),list(side_scores),  color='red', alpha=alpha, width=3)
        ax1.bar(list(indexes), list(combined_scores), color='green', alpha=0.4, width=0.8)

        ax1.plot(indexes,front_imagegrab[1:],marker='o',color='blue')
        ax1.axes.get_yaxis().set_ticks([])
        plt.xticks(np.arange(1,numFrames,100), fontsize=16)
        plt.ylim((0,1.5))

        prev_txt=0
        for i, txt in enumerate(front_skip_imagegrab_gt[i]):
            if((txt - prev_txt) < 50):
                scalar = 0.005
            else:
                scalar=0.0
            prev_txt = txt
            ax1.annotate(txt, xy=(indexes[txt]+20, front_imagegrab[txt]+(i*scalar)),fontsize=16) #xytext=(indexes[txt]+1, front_imagegrab[txt]+1))
        plt.savefig('C:/Users/27rut/BIAS/misc/jab_plugin_day_trials/figs/random_intervals_front_skip_video_delay1800.jpg')'''

        plt.figure(figsize=(30,30))
        ax2 = plt.gca()
        ax2.bar(list(indexes), list(combined_imagegrab), width=0.5 ,color='green',alpha=1, align='center')
        ax2.bar(list(indexes), list(side_imagegrab), width=0.5, color='red', alpha=1, align='center')
        ax2.bar(list(indexes), list(front_imagegrab), width=0.5, color='blue', alpha=1, align='center')

        ax2_new = ax2.twiny().twinx()
        ax2_new.barh(list(indexes), list(combined_scores), height=0.5, color='green', alpha=1, align='center')
        ax2_new.barh(list(indexes), list(front_scores), height=0.5, color='blue', alpha=1, align='center')
        ax2_new.barh(list(indexes), list(side_scores), height=0.5, color='red', alpha=1, align='center')
        plt.title('Match skipped frames at imagegrab to frames skipped at scores scollection', fontsize=40, pad=20)
        ax2.tick_params(axis='x', labelsize=40)
        ax2_new.tick_params(axis='y', labelsize=40)
        plt.show()
        #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/random_intervals_side-front_skip_video_delay1800_scr.jpg')

if __name__ == "__main__":
    main()