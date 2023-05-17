import numpy as np
import matplotlib.pyplot as plt
import utils as ut

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
    imagegrab_nidaqthres_cam0 = 'imagegrab_nidaq_threscam0_short_trial'
    imagegrab_nidaqthres_cam1 = 'imagegrab_nidaq_threscam1_short_trial'
    scores_prefix = 'classifier_trial'
    scores_prefix_gt = 'lift_classifier.csv'
    scores_side_prefix_gt = 'lift_classifier_side.csv'
    scores_front_prefix_gt = 'lift_classifier_front.csv'

    skips_side_gt = np.array(no_of_trials * [numFrames * [0.0]]) ## frames skipped in imagegrab side by the user
    skips_front_gt = np.array(no_of_trials * [numFrames * [0.0]]) ## frames skipped in imagegrab front by the user
    scores_jaaba = np.array(no_of_trials * [numFrames * [0.0]])       ## scores predicted
    scores_jaaba_view = np.array(no_of_trials * [numFrames * [0.0]])  ## view contributing to the score
    scores_gt = np.array(no_of_trials * [numFrames * [0.0]])          ## scores gt
    scores_side_gt = np.array(no_of_trials * [numFrames * [0.0]])     ## gt for only side view
    scores_front_gt = np.array(no_of_trials * [numFrames * [0.0]])    ##  gt for only front view
    image_thres_cam0 = np.array(no_of_trials * [numFrames * [0.0]])
    image_thres_cam1 = np.array(no_of_trials * [numFrames * [0.0]])

    ## processScores thread
    front_skips = []
    side_skips = []
    both_skips = []
    correct_frames = []

    ## imagegrab thread
    side_skip_imagegrab_gt = []
    front_skip_imagegrab_gt = []
    side_skip_imagegrab = []
    front_skip_imagegrab = []

    for i in range(0,no_of_trials):

        trial_type = str(i+1)
        classifier_score_file = output_dir + scores_prefix + trial_type + '.csv.'
        classifier_score_file_gt = output_dir_gt + scores_prefix_gt
        classifier_score_side_file_gt = output_dir_gt + scores_side_prefix_gt
        classifier_score_front_file_gt = output_dir_gt + scores_front_prefix_gt
        imagegrab_skipped_frames_cam0_file = output_dir  + imagegrab_skipped_frames_cam0 + trial_type + '.csv'
        imagegrab_skipped_frames_cam1_file = output_dir + imagegrab_skipped_frames_cam1 + trial_type + '.csv'
        imagegrab_nidaq_cam0_file = output_dir + imagegrab_nidaqthres_cam0 + trial_type + '.csv'
        imagegrab_nidaq_cam1_file = output_dir + imagegrab_nidaqthres_cam1 + trial_type + '.csv'

        ut.readcsvFile_int(imagegrab_nidaq_cam0_file, image_thres_cam0[i], 1, 0)
        ut.readcsvFile_int(imagegrab_nidaq_cam1_file, image_thres_cam1[i], 1, 0)
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
        side_skip_imagegrab.append(np.argwhere(image_thres_cam0[i]==1).flatten())
        front_skip_imagegrab.append(np.argwhere(image_thres_cam1[i]==1).flatten())

        ## plotting skip data in process scores thread
        indexes = np.arange(0,numFrames-1,1)
        front_scores = np.ones(numFrames-1)*2
        side_scores = np.ones(numFrames-1)*2

        not_front_skips = not_intersection(np.arange(0,numFrames-1), front_skips[i])
        not_side_skips = not_intersection(np.arange(0,numFrames-1), side_skips[i])
        not_correct_frames = not_intersection(np.arange(0,numFrames-1), correct_frames[i])
        front_scores[not_front_skips] = 0
        side_scores[not_side_skips] = 0

        ## plotting skip data in imagegrab thread
        side_imagegrab = np.ones(numFrames-1)
        front_imagegrab = np.ones(numFrames-1)

        not_front_skip_imagegrab = not_intersection(np.arange(0,numFrames-1), front_skip_imagegrab[i])
        not_side_skip_imagegrab = not_intersection(np.arange(0,numFrames-1), side_skip_imagegrab[i])

        front_imagegrab[not_front_skip_imagegrab] = 0.0
        side_imagegrab[not_side_skip_imagegrab] = 0.0

        plt.figure(figsize=(20,20))
        ax3=plt.gca()
        ax3.bar(list(indexes),list(front_scores), color='mediumblue', alpha=1, width=0.5)
        ax3.bar(list(indexes),list(front_imagegrab), color='mediumpurple', alpha=1, width=0.5)
        ax3.bar(list(indexes),list(side_scores), color='firebrick', alpha=1, width=0.5)
        ax3.bar(list(indexes),list(side_imagegrab), color='lightcoral', alpha=1, width=0.5)
        ax3.tick_params(axis='x', which='major',labelsize=20)
        ax3.set_yticks([])
        plt.xticks(np.arange(0,numFrames-1,10))
        plt.legend(['Scores front skips','Imagegrab front skips','Scores side skips', 'Imagegrab side skips'], fontsize=15)
        plt.title('Matching skips at image grab with scores', fontsize=30)
        #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/random_intervals_random_delay_bothview_skips.jpg')
        plt.show()

if __name__ == "__main__":
    main()