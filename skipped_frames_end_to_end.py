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

    print("Length of frames that have been skipped", len(arr1), len(arr2))
    print(arr1)
    print(arr2)

    if(len(arr1) != len(arr2)):
        print('Skipped frames do not match with gt')

    if(len(arr1) == len(arr2)):
        print('Indexes match same length')

    num_skips =  len(arr1)
    cnt = 0

    if( len(arr1) != 0 and len(arr2)!=0):
      for x in zip(arr1,arr2):
        if(x[0] == x[1]):
          cnt += 1
          continue
        else:
          print('Mismatched indexes',x[0],x[1])

      if(cnt == num_skips):
          print('All skipped frames int gt match the frames actually skipped')


def match_scores(scr_ind_front, scr_ind_side, scr, scr_side_gt, scr_front_gt, scr_gt):

    mismatch_ind1=[]
    mismatch_ind2=[]
    mismatch_ind3=[]

    cnt = 0
    numScores_match = len(scr_ind_front)

    if(len(scr_ind_front) != 0):
        for ind in scr_ind_front:
            if( scr_side_gt[ind-1] == scr[ind-1]):
                cnt += 1
                continue
            else:
                mismatch_ind1.append(ind)

        if(numScores_match == cnt):
            print('Front scores match')

    cnt=0
    numScores_match = len(scr_ind_side)

    if(len(scr_ind_side) != 0):
        for ind in scr_ind_side:
            if(scr_front_gt[ind-1] == scr[ind-1]):
                cnt += 1
                continue
            else:
                mismatch_ind2.append(ind)

        if(cnt == numScores_match):
            print('Side scores match')

    cnt=0
    numScores_match = len(scr)-1

    if(len(scr) != 0):
        ## score at index 0 corresponds to frame1
        for ind in range(0,len(scr)-1):
            if(scr[ind] == scr_gt[ind]):
                cnt += 1
                continue
            else:
                mismatch_ind3.append(ind+1)

    print('Scores matching gt', cnt)
    print('Scores mismatching the gt', numScores_match - cnt)

    print(mismatch_ind3)
    return (mismatch_ind1 , mismatch_ind2, mismatch_ind3)

def main():

    output_dir = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/2c5ba_9_8_2022/'
    output_dir_gt = 'C:/Users/27rut/BIAS/misc/classifier_trials/classifier_scores_new/'
    trial_type = '1'
    no_of_trials = 2
    numFrames = 2498
    video = True

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
        if(len(imagegrab_front_skips[i]) != 0):
            imagegrab_front_skips[i] = imagegrab_front_skips[i][:-1]
        if(len(imagegrab_side_skips[i]) != 0):
            imagegrab_side_skips[i] = imagegrab_side_skips[i][:-1]
        imagegrab_side_skips_gt.append(np.argwhere(skips_side_gt[i]==1).flatten())
        imagegrab_front_skips_gt.append(np.argwhere(skips_front_gt[i]==1).flatten())

        front_skips.append(np.argwhere(scores_jaaba_view[i] == 1).flatten()+1) # 1 meaning front has been skipped
                                                                                  # reporting score from side
        side_skips.append(np.argwhere(scores_jaaba_view[i] == 2).flatten()+1) # 2 meaning side has been skipped
                                                                                 # and reporting score from front
        both_skips.append(np.argwhere(scores_jaaba_view[i] == -1).flatten()+1) # skipped in both views
        correct_frames.append(np.argwhere(scores_jaaba_view[i] == 3).flatten()+1) # both views correct

        both_skips_grab.append((intersection(imagegrab_side_skips[i],imagegrab_front_skips[i]))) # skipped at grab in both views
        both_skips_compute.append(not_intersection(both_skips[i],both_skips_grab[i])) #skipped in both views at jaaba compute

        only_imagegrab_side.append((not_intersection(both_skips_grab[i],imagegrab_side_skips[i]))) # only frames skipped at
                                                                                                   # imagergab with no frames skipped in both views
        only_imagegrab_front.append((not_intersection(both_skips_grab[i],imagegrab_front_skips[i])))

        skips_process_front.append(not_intersection(front_skips[i],only_imagegrab_front[i])) # frames skipped only
                                                                                            # because of compute
        skips_process_side.append(not_intersection(side_skips[i],only_imagegrab_side[i]))

        # matching skips in imagegrab against gt
        if (len(imagegrab_side_skips[i]) != 0 and len(imagegrab_side_skips_gt[i]) != 0):
            matching_frames(imagegrab_side_skips[i], imagegrab_side_skips_gt[i])
            matching_frames(imagegrab_side_skips_gt[i], np.sort(not_intersection(intersection(side_skips[i], \
                                                         imagegrab_side_skips[i]), both_skips_grab[i])))
            if ((len(side_skips[i]) - len(only_imagegrab_side[i])) == len(skips_process_side[i])):
                print('Frames skipped in side at imagegrab match ones in process Scores')
        if (len(imagegrab_front_skips[i]) != 0 and len(imagegrab_front_skips_gt[i]) != 0):
            matching_frames(imagegrab_front_skips[i], imagegrab_front_skips_gt[i])
            matching_frames(imagegrab_front_skips_gt[i], np.sort(not_intersection(intersection(front_skips[i],imagegrab_front_skips[i])
                                                               ,both_skips_grab[i])))
            if((len(front_skips[i]) - len(only_imagegrab_front[i])) == len(skips_process_front[i])):
                print('Frames skipped in front at imagegrab match ones in process Scores')


        # match the score of indexes where either side/front
        view1_mismatchind, view2_mismatchind, overall_mismatchind = match_scores(front_skips[i], side_skips[i], \
                                        scores_jaaba[i], scores_side_gt[i], scores_front_gt[i],scores_gt[i])
        print(overall_mismatchind)
        for val in both_skips_compute[i]:
            if val in skips_process_front[i]:
                skips_process_front[i].remove(val)
            if val in skips_process_side[i]:
                skips_process_side[i].remove(val)
            if val in only_imagegrab_side[i]:
                only_imagegrab_side[i].remove(val)
            if val in only_imagegrab_front[i]:
                only_imagegrab_front[i].remove(val)

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

        ax2 = plt.gca()
        scr_gt_diff = np.array(numFrames * [0.0])
        scr_diff = np.array(numFrames * [0.0])

        for ind in overall_mismatchind:
            scr_gt_diff[ind] = scores_gt[i][ind]
            scr_diff[ind] = scores_jaaba[i][ind]

        indexes = np.arange(0, numFrames,1)
        ax2.bar(list(indexes[430:450]),scr_gt_diff[430:450], width=0.5,color='red',align='center')
        ax2.bar(list(indexes[430:450]),scr_diff[430:450], width=0.5, color='blue',align='center')
        plt.show()



if __name__ == "__main__":
    main()