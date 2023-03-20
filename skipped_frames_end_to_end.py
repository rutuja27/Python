import numpy as np
import matplotlib.pyplot as plt
import utils as ut
import itertools as it
import plotting_code  as pc

def not_intersection(lst1, lst2):
    return list(set(lst1) ^ set(lst2))

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def matching_frames(arr1, arr2):

    if(len(arr1) != len(arr2)):
        print('Skipped frames do not match with gt')

    if(len(arr1) == len(arr2)):
        print('Indexes match same length')

    if(len(arr1) !=0 and len(arr2) != 0):
        not_in_test = [ind for ind in arr2 if ind not in arr1]
        not_in_gt = [ind for ind in arr1 if ind not in arr2 ]

    return not_in_test, not_in_gt

def match_scores(scr_ind_front, scr_ind_side, scr, scr_side_gt, scr_front_gt, scr_gt):

    mismatch_ind1=[]
    mismatch_ind2=[]
    mismatch_ind3=[]

    cnt = 0
    numScores_match = len(scr_ind_front)

    if(len(scr_ind_front) != 0):
        ## score at index 0 corresponds to frame 1 for side scr gt
        for ind in scr_ind_front:
            if( round(scr_side_gt[ind-1],3) == round(scr[ind],3)):
                cnt += 1
                continue
            else:
                mismatch_ind1.append(ind)

        if(numScores_match == cnt):
            print('Front scores match')

    cnt=0
    numScores_match = len(scr_ind_side)

    if(len(scr_ind_side) != 0):
        ## score at index 0 corresponds to frame 1 for frnt scr gt
        for ind in scr_ind_side:
            if(round(scr_front_gt[ind-1],3) == round(scr[ind],3)):
                cnt += 1
                continue
            else:
                mismatch_ind2.append(ind)

        if(cnt == numScores_match):
            print('Side scores match')

    cnt=0
    numScores_match = len(scr)-2

    if(len(scr) != 0):
        ## score at index 0 corresponds to frame 1 for scr gt
        for ind in range(1,len(scr)-1):
            if(round(scr[ind],3) == round(scr_gt[ind-1],3)):
                cnt += 1
                continue
            else:
                mismatch_ind3.append(ind)

    print('Scores matching gt', cnt)
    print('Scores mismatching the gt', numScores_match - cnt)

    return (mismatch_ind1 , mismatch_ind2, mismatch_ind3)

def main():

    output_dir = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/2c5ba_9_8_2022/'
    output_dir_gt = 'C:/Users/27rut/BIAS/misc/classifier_trials/classifier_scores/'
    trial_type = '1'
    no_of_trials = 5
    numFrames = 2498
    video = True
    no_of_skips_view1 = 5
    no_of_skips_view2 = 0

    if video:
        classifier_trial = 'classifier_trial'
        nidaq_prefix_jaabathres_cam0 = 'jaaba_plugin_nidaq_threscam0_short_trial'
        nidaq_prefix_jaabathres_cam1 = 'jaaba_plugin_nidaq_threscam1_short_trial'
        nidaq_prefix_jaaba_cam0 = 'jaaba_plugin_nidaqcam0_short_trial'
        nidaq_prefix_jaaba_cam1 = 'jaaba_plugin_nidaqcam1_short_trial'
        nidaq_prefix_imagegrabthres_cam0 = 'imagegrab_nidaq_threscam0_short_trial'
        nidaq_prefix_imagegrabthres_cam1 = 'imagegrab_nidaq_threscam1_short_trial'
        imagegrab_skipped_frames_cam0 = 'imagegrab_skipped_framescam0_short_trial'
        imagegrab_skipped_frames_cam1 = 'imagegrab_skipped_framescam1_short_trial'
        scores_prefix_gt = 'lift_classifier.csv'
        scores_side_prefix_gt = 'lift_classifier_side.csv'
        scores_front_prefix_gt = 'lift_classifier_front.csv'

    nidaq_thres_imagegrab_cam0 = np.array(no_of_trials * [numFrames * [0.0]])
    nidaq_thres_imagegrab_cam1 = np.array(no_of_trials * [numFrames * [0.0]])
    nidaq_thres_jaaba_cam0 = np.array(no_of_trials * [numFrames * [0.0]])
    nidaq_thres_jaaba_cam1 = np.array(no_of_trials * [numFrames * [0.0]])
    nidaq_jaaba_cam0 = np.array(no_of_trials * [numFrames * [0.0]])
    nidaq_jaaba_cam1 = np.array(no_of_trials * [numFrames * [0.0]])
    nidaq_camtrig = np.array(no_of_trials * [numFrames * [0.0]])

    scores_jaaba = np.array(no_of_trials * [numFrames * [0.0]])
    scores_jaaba_view = np.array(no_of_trials * [numFrames * [0.0]])

    skips_side_gt = np.array(no_of_trials * [numFrames * [0.0]]) ## frames skipped in imagegrab side by the user
    skips_front_gt = np.array(no_of_trials * [numFrames * [0.0]]) ## frames skipped in imagegrab front by the user
    mag_skips_side_gt = np.array(no_of_trials * [numFrames * [0.0]]);# magnitude of the latency introduced by the user
    mag_skips_front_gt = np.array(no_of_trials * [numFrames * [0.0]]);# magnitude of the latency introduced by the user

    scores_gt = np.array(no_of_trials * [numFrames * [0.0]])
    scores_side_gt = np.array(no_of_trials * [numFrames * [0.0]])
    scores_front_gt = np.array(no_of_trials * [numFrames * [0.0]])

    front_skips = []
    side_skips = []
    imagegrab_side_skips_gt = []
    imagegrab_front_skips_gt = []
    imagegrab_side_skips = []
    imagegrab_front_skips = []
    jaaba_side_skips = []
    jaaba_front_skips = []
    both_skips = []
    skips_process_front = []
    skips_process_side = []
    correct_frames = []
    zero_ind = []
    only_imagegrab_side = []
    only_imagegrab_front = []
    both_skips_grab = []
    both_skips_compute = []

    for i in range(0,no_of_trials):
        trial_type = str(i+1)
        classifier_score_file = output_dir + classifier_trial + trial_type + '.csv.'
        classifier_score_file_gt = output_dir_gt + scores_prefix_gt
        classifier_score_side_file_gt = output_dir_gt + scores_side_prefix_gt
        classifier_score_front_file_gt = output_dir_gt + scores_front_prefix_gt
        nidaq_thres_jaaba_cam0_file = output_dir + nidaq_prefix_jaabathres_cam0 + trial_type + '.csv'
        nidaq_thres_jaaba_cam1_file = output_dir + nidaq_prefix_jaabathres_cam1 + trial_type + '.csv'
        nidaq_jaaba_cam0_file = output_dir + nidaq_prefix_jaaba_cam0 + trial_type + '.csv'
        nidaq_jaaba_cam1_file = output_dir + nidaq_prefix_jaaba_cam1 + trial_type + '.csv'
        nidaq_thres_imagegrab_cam0_file = output_dir + nidaq_prefix_imagegrabthres_cam0 + trial_type + '.csv'
        nidaq_thres_imagegrab_cam1_file = output_dir + nidaq_prefix_imagegrabthres_cam1 + trial_type + '.csv'
        imagegrab_skipped_frames_cam0_file = output_dir  + imagegrab_skipped_frames_cam0 + trial_type + '.csv'
        imagegrab_skipped_frames_cam1_file = output_dir + imagegrab_skipped_frames_cam1 + trial_type + '.csv'

        ut.readcsvFile_nidaq(nidaq_jaaba_cam0_file, nidaq_camtrig[i], nidaq_jaaba_cam0[i])
        ut.readcsvFile_nidaq(nidaq_jaaba_cam1_file, nidaq_camtrig[i], nidaq_jaaba_cam1[i])

        ut.readcsvFile_int(nidaq_thres_jaaba_cam0_file, nidaq_thres_jaaba_cam0[i], 1, 0)
        ut.readcsvFile_int(nidaq_thres_jaaba_cam1_file, nidaq_thres_jaaba_cam1[i], 1,0)
        ut.readcsvFile_int(nidaq_thres_imagegrab_cam0_file, nidaq_thres_imagegrab_cam0[i], 1,0)
        ut.readcsvFile_int(nidaq_thres_imagegrab_cam1_file, nidaq_thres_imagegrab_cam1[i], 1,0)
        ut.read_score(classifier_score_file,scores_jaaba[i],0 ,3)
        ut.read_score(classifier_score_file, scores_jaaba_view[i],0, 5)

        # gt readings
        ut.readcsvFile_int(imagegrab_skipped_frames_cam0_file, skips_side_gt[i], 1,0)
        ut.readcsvFile_int(imagegrab_skipped_frames_cam1_file, skips_front_gt[i], 1,0)
        ut.readcsvFile_int(imagegrab_skipped_frames_cam0_file, mag_skips_side_gt[i], 1,1)
        ut.readcsvFile_int(imagegrab_skipped_frames_cam1_file, mag_skips_front_gt[i], 1,1)

        ut.read_score(classifier_score_file_gt, scores_gt[i],1,1)
        ut.read_score(classifier_score_side_file_gt, scores_side_gt[i],1,1)
        ut.read_score(classifier_score_front_file_gt, scores_front_gt[i],1,1)

        #plt.plot(nidaq_jaaba_cam0[i][:-1] - nidaq_camtrig[i][:-1],'.')
        #plt.plot(nidaq_jaaba_cam1[i][:-1] - nidaq_camtrig[i][:-1],'.')

        jaaba_side_skips.append(np.argwhere(nidaq_thres_jaaba_cam0[i]==1).flatten())
        jaaba_front_skips.append(np.argwhere(nidaq_thres_jaaba_cam1[i]==1).flatten())
        imagegrab_side_skips.append(np.argwhere(nidaq_thres_imagegrab_cam0[i]==1).flatten())
        imagegrab_front_skips.append(np.argwhere(nidaq_thres_imagegrab_cam1[i]==1).flatten())
        '''if(len(imagegrab_front_skips[i]) != 0):
            imagegrab_front_skips[i] = imagegrab_front_skips[i][:-1]
        if(len(imagegrab_side_skips[i]) != 0):
            imagegrab_side_skips[i] = imagegrab_side_skips[i][:-1]'''
        imagegrab_side_skips_gt.append(np.argwhere(skips_side_gt[i]==1).flatten())
        imagegrab_front_skips_gt.append(np.argwhere(skips_front_gt[i]==1).flatten())

        front_skips.append(np.argwhere(scores_jaaba_view[i] == 1).flatten()) # 1 meaning front has been skipped
                                                                                  # reporting score from side
        side_skips.append(np.argwhere(scores_jaaba_view[i] == 2).flatten()) # 2 meaning side has been skipped
                                                                                 # and reporting score from front
        both_skips.append(np.argwhere(scores_jaaba_view[i] == -1).flatten()) # skipped in both views
        correct_frames.append(np.argwhere(scores_jaaba_view[i] == 3).flatten()) # both views correct

        both_skips_grab.append((intersection(imagegrab_side_skips[i],imagegrab_front_skips[i]))) # skipped at grab in both views
        both_skips_compute.append(not_intersection(both_skips[i],both_skips_grab[i])) #skipped in both views at jaaba compute

        only_imagegrab_side.append((not_intersection(both_skips_grab[i],imagegrab_side_skips[i]))) # only frames skipped at
                                                                                                   # imagergab with no frames skipped in both views
        only_imagegrab_front.append((not_intersection(both_skips_grab[i],imagegrab_front_skips[i])))

        skips_process_front.append(not_intersection(front_skips[i],only_imagegrab_front[i])) # frames skipped only
                                                                                            # because of compute
        skips_process_side.append(not_intersection(side_skips[i],only_imagegrab_side[i]))


        for val in both_skips_compute[i]:
            if val in skips_process_front[i]:
                skips_process_front[i].remove(val)
            if val in skips_process_side[i]:
                skips_process_side[i].remove(val)
            if val in only_imagegrab_side[i]:
                only_imagegrab_side[i].remove(val)
            if val in only_imagegrab_front[i]:
                only_imagegrab_front[i].remove(val)

                # matching skips in imagegrab against gt
            if (len(imagegrab_side_skips[i]) != 0 and len(imagegrab_side_skips_gt[i]) != 0):
                nomatch_in_test_sde, nomatch_in_gt_sde = matching_frames(imagegrab_side_skips[i],
                                                                         imagegrab_side_skips_gt[i])
                print('Present in gt but not in test for side', nomatch_in_test_sde)
                print('Present in test but not in gt for side', nomatch_in_gt_sde)

                # matching_frames(np.sort(not_intersection(intersection(side_skips[i], \
                #                                             imagegrab_side_skips[i]), both_skips_grab[i])),imagegrab_side_skips[i])

                if ((len(side_skips[i]) - len(only_imagegrab_side[i])) == len(skips_process_side[i])):
                    print('Frames skipped in side at imagegrab match ones in process Scores')

            if (len(imagegrab_front_skips[i]) != 0 and len(imagegrab_front_skips_gt[i]) != 0):
                nomatch_in_test_frt, nomatch_in_gt_frt = matching_frames(imagegrab_front_skips[i],
                                                                         imagegrab_front_skips_gt[i])
                # matching_frames(np.sort(not_intersection(intersection(front_skips[i],imagegrab_front_skips[i])
                #                                                   ,both_skips_grab[i])) ,imagegrab_front_skips[i])
                print('Present in gt but not in test for side', nomatch_in_test_frt)
                print('Present in test but not in gt for side', nomatch_in_gt_frt)

                if ((len(front_skips[i]) - len(only_imagegrab_front[i])) == len(skips_process_front[i])):
                    print('Frames skipped in front at imagegrab match ones in process Scores')

                # match the score of indexes where either side/front
            view1_mismatchind, view2_mismatchind, overall_mismatchind = match_scores(front_skips[i], side_skips[i], \
                                                                                     scores_jaaba[i], scores_side_gt[i],
                                                                                     scores_front_gt[i], scores_gt[i])

            if (view1_mismatchind):
                print(view1_mismatchind)
                print('Check side skipping')

            if (view2_mismatchind):
                print(view2_mismatchind)
                print('Check front skipping')

            if (overall_mismatchind):
                if (imagegrab_side_skips[i].size != 0 and imagegrab_front_skips[i].size != 0):
                    combined_skips = set(imagegrab_side_skips[i]).union(set(imagegrab_front_skips[i]))
                    mismatches_scores = np.sort(not_intersection(overall_mismatchind, combined_skips))
                    print('Mismatch scores ', mismatches_scores)

        print('Trial type', i)
        print('Skips at jaaba compute side', np.sort(skips_process_side[i]))
        print('Skips at jaaba compute front', np.sort(skips_process_front[i]))
        print('Skips at jaaba grab for cam 0' , np.sort(only_imagegrab_side[i]))
        print('Skips at jaaba grab for cam 1' , np.sort(only_imagegrab_front[i]))
        print('Number of skips in side after jaaba compute', len(skips_process_side[i]))
        print('Number of skips in front after jaaba compute', len(skips_process_front[i]))
        print('Number of jaaba skips at grab for cam 0' ,len(only_imagegrab_side[i]))
        print('Number of jaaba skips at grab for cam 1', len(only_imagegrab_front[i]))
        print('Frames skipped at compute', len(skips_process_side[i])+len(skips_process_front[i]))
        print('Frames skipped at grab' , len(only_imagegrab_side[i]) + len(only_imagegrab_front[i]))
        print('skips at grab + compute' , len(skips_process_side[i])+len(skips_process_front[i])\
              + len(only_imagegrab_side[i]) + len(only_imagegrab_front[i]))

        print('Skipped in both views ', len(both_skips[i]))

        print('********')
        print('length of correct frames', len(correct_frames[i]))
        print('Number of total skipped frames' , ((numFrames-1) - len(correct_frames[i])))
        print('\n')

        ind = mag_skips_side_gt[i] != 0
        print(mag_skips_side_gt[i][ind])
        ax1 = plt.gca()
        ax1.plot(mag_skips_side_gt[i], '.',color='red')
        ax1.plot(mag_skips_front_gt[i], '.', color='blue')
        plt.show()


        '''fig2, ax2 = plt.subplots(no_of_skips_view1, figsize=(15,15))
        scr_gt_diff = np.array(numFrames * [0.0])
        scr_diff = np.array(numFrames * [0.0])
        indexes = np.arange(0, numFrames, 1)
        max_score = max(scores_gt[i][:])
        min_score = min(scores_gt[i][:])

        if(view1_mismatches_scores.size != 0 ):
            for id,vals in enumerate(view1_mismatches_scores):
                min_range = max(0,vals-10)
                max_range = min(vals+10,numFrames-2)
                for ind in range(min_range,max_range):
                    scr_gt_diff[ind] = scores_gt[i][ind-1]
                    scr_diff[ind] = scores_jaaba[i][ind]

                ax2[id].bar(list(indexes[min_range:max_range]),scr_gt_diff[min_range:max_range], width=0.5,alpha =0.2, color='red',align='center')
                ax2[id].bar(list(indexes[min_range:max_range]),scr_diff[min_range:max_range], width=0.5, alpha=0.5, color='blue',align='center')
                ax2[id].set_xticks(np.arange(min_range, max_range+1,2))
                ax2[id].set_yticks(np.arange(min_score,max_score,5.0))
                ax2[id].tick_params(labelsize=15)
        fig2.subplots_adjust(hspace=0.7)'''

        plt.show()



if __name__ == "__main__":
    main()