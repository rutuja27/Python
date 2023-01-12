import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut

def main():

    numFrames = 2497

    filepath = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/2c5ba_9_8_2022/'
    trial_type = '5'
    classifier_scr_file = filepath + 'classifier_trial' + trial_type + '.csv'
    classifier_scr_side_file = filepath + 'classifier_scr_side_trial' + trial_type + '.csv'
    classifier_scr_front_file = filepath + 'classifier_scr_front_trial' + trial_type + '.csv'
    classifier_scr_side_file_gt = filepath + 'lift_classifier_side.csv'
    classifier_scr_front_file_gt = filepath + 'lift_classifier_front.csv'
    classifier_scr_file_gt = filepath + 'lift_classifier.csv'
    
    scores_jaaba = np.array(numFrames*[0.0])
    scores_jaaba_side = np.array(numFrames*[0.0])
    scores_jaaba_front = np.array(numFrames*[0.0])
    scores_jaaba_gt = np.array(numFrames*[0.0])
    scores_jaaba_side_gt = np.array(numFrames * [0.0])
    scores_jaaba_front_gt = np.array(numFrames * [0.0])

    scores_jaaba_ts = np.array(numFrames*[0.0])
    scores_jaaba_side_ts = np.array(numFrames*[0.0])
    scores_jaaba_front_ts = np.array(numFrames*[0.0])

    score_view = np.array(numFrames*[0.0])

    ## read scores
    ut.read_score(classifier_scr_file, scores_jaaba, 0, 3)
    #ut.read_score(classifier_scr_side_file, scores_jaaba_side, 0, 3)
    #ut.read_score(classifier_scr_front_file, scores_jaaba_front, 0, 3)

    ## read jaaba gnd trth scores
    ut.read_score(classifier_scr_file_gt, scores_jaaba_gt, 0, 1)

    #ut.read_score_view(classifier_scr_file, score_view, scores_jaaba,
    #                   scores_jaaba_side,scores_jaaba_front, 0, 5)

    ## read score ts
    ut.read_score(classifier_scr_file, scores_jaaba_side_ts, 0, 1)
    ut.read_score(classifier_scr_file, scores_jaaba_front_ts, 0, 2)
    ut.read_score(classifier_scr_file, scores_jaaba_ts, 0, 0)
    max_score_ts = np.maximum(scores_jaaba_front_ts, scores_jaaba_side_ts)
    wait_thres = (scores_jaaba_ts - max_score_ts)/1000
    print(np.sum(wait_thres)/(numFrames-1))
    print(np.sum((wait_thres > 1000)))

    #print(score_view)

    plt.figure()
    ax1 = plt.gca()
    ax1.plot(scores_jaaba,scores_jaaba_gt, '.')

    #plt.figure()
    #ax2 = plt.gca()
    #ax2.plot(scores_jaaba_side+scores_jaaba_front,scores_jaaba,'.',color='blue')


    plt.figure()
    ax3 = plt.gca()
    ax3.plot(wait_thres[10:],'.',color='green')


    plt.figure()
    ax4 = plt.gca()

    ax4.plot(scores_jaaba[0:],'.',alpha=1,color='red')
    ax4.plot(scores_jaaba_side[0:], '.', alpha=0.2, color='green')
    ax4.plot(scores_jaaba_front[0:], '.', alpha=0.3, color='blue')
    # ax4.plot(classifier_scr_gt[5:-2],'.',alpha=0.5,color='blue')
    #ax4.plot(classifier_scr_gt[5:-2],'.',alpha=0.5,color='blue')
    #plt.title('Comparison of Lift Classifier Score')
    #plt.ylabel('JAABA Classifier GT Scores')
    #plt.xlabel('JAABA Classifier Scores')
    #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/correlation_biasjaaba_predVsgt_woskip_.pdf')
    plt.show()

if __name__ == "__main__":
    main()