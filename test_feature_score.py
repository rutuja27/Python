import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import csv
import sys
import utils as ut

def main():

    exp_dir = (sys.argv[1])
    numWkcls = 100
    matlab_feat_score_file = exp_dir + "scores.mat"
    cuda_feat_side_score_file = exp_dir + "test_side.csv"
    cuda_feat_front_score_file = exp_dir + "test_front.csv"

    score_feat_matlab = loadmat(matlab_feat_score_file)
    score_feat_matlab = score_feat_matlab['scores'][0]
    #print(score_feat_matlab)

    score_feat_cuda = np.array(numWkcls*[0.0])
    score_feat_side = []
    score_feat_front = []
    score_feat_side_Wkclsid = []
    score_feat_front_Wkclsid = []

    ut.readArray(cuda_feat_side_score_file,score_feat_side_Wkclsid,0)
    ut.readArray(cuda_feat_front_score_file,score_feat_front_Wkclsid,0)
    ut.readArray(cuda_feat_side_score_file, score_feat_side, 1)
    ut.readArray(cuda_feat_front_score_file,score_feat_front,1)

    check_index = 100
    side_cnt =0
    front_cnt=0
    side_scr_cum = 0
    front_scr_cum = 0
    front_scr_len = len(score_feat_front_Wkclsid)
    side_scr_len = len(score_feat_side_Wkclsid)

    print('Classifier id side',score_feat_side_Wkclsid)
    print('Classifier id front',score_feat_front_Wkclsid)
    print('Classifier score side', score_feat_side)
    print('Classifier score front',score_feat_front)

    while(score_feat_side_Wkclsid[side_cnt] < check_index and side_cnt < side_scr_len):
        side_cnt += 1
        if(side_cnt == side_scr_len):
            break
    while(score_feat_front_Wkclsid[front_cnt] < check_index and front_cnt < front_scr_len):
        front_cnt += 1
        if(front_cnt == front_scr_len):
            break
    side_cnt = side_cnt - 1 ## we want to be one index before the check index
    front_cnt = front_cnt - 1

    score_cuda = score_feat_side[side_cnt] + score_feat_front[front_cnt]
    print('Side Count- Front Count', side_cnt, front_cnt)

    print(score_feat_side_Wkclsid[side_cnt] ,score_feat_front_Wkclsid[front_cnt])
    print('Score cuda' ,score_cuda)
    print('Score matlab', score_feat_matlab[check_index-1])


if __name__ == "__main__":
    main()



