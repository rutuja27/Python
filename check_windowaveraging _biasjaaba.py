import numpy as np
import csv
import matplotlib.pyplot as plt
import utils as ut
from ast import literal_eval

def readcsvFile(filename, hoghof_feat,numframes, feat_szie):

    hoghof_handle = open(filename, 'r+');
    hoghof = csv.reader(hoghof_handle, delimiter='\n')
    cnt = 0
    for idx, row in enumerate(hoghof):
        if(cnt < numframes):
           vals = row[0].split(',')
           vals = vals[:-1]
           res = [float(x) for x in vals]
           hoghof_feat[idx][:] = np.array(res)
           cnt = cnt + 1
        else:

            break
    hoghof_handle.close()

def main():

    #filepath = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/c7aa8_4_25_2023/'
    filepath = 'Y:/hantman_data/jab_experiments/STA14/STA14/20230503/STA14_20230503_142259/'
    numframes = 2798
    feat_size_side = 4799
    feat_size_front= 3199
    window_size=5

    hoghof_side = np.zeros((numframes, feat_size_side))
    hoghof_side_gt = np.zeros((numframes, feat_size_side))
    hoghof_avg_side = np.zeros((numframes, feat_size_side))

    hoghof_front = np.zeros((numframes,feat_size_front))
    hoghof_front_gt = np.zeros((numframes, feat_size_front))
    hoghof_avg_front = np.zeros((numframes,feat_size_front))

    readcsvFile(filepath + 'hoghof_side_biasjaaba.csv' , hoghof_side, numframes, feat_size_side)
    readcsvFile(filepath + 'hoghof_avg_side_biasjaaba.csv', hoghof_avg_side, numframes, feat_size_side)
    readcsvFile(filepath + 'hoghof_front_biasjaaba.csv' , hoghof_front, numframes, feat_size_front)
    readcsvFile(filepath + 'hoghof_avg_front_biasjaaba.csv', hoghof_avg_front, numframes, feat_size_front)

    for i in range(0,numframes):
        if i < window_size:
            hoghof_side_gt[i][:] = np.divide(np.sum(hoghof_side[0:i+1][:],axis=0),window_size)
            hoghof_front_gt[i][:] = np.divide(np.sum(hoghof_front[0:i+1][:], axis=0), window_size)

        else:
            hoghof_side_gt[i][:] =  np.divide(np.sum(hoghof_side[i-(window_size-1):i+1][:],axis=0),window_size)
            hoghof_front_gt[i][:] = np.divide(np.sum(hoghof_front[i - (window_size - 1):i + 1][:], axis=0), window_size)

    plt.figure(figsize=(15, 8))
    ax1 = plt.gca()
    ax1.plot(hoghof_avg_side[:],hoghof_side_gt[:],'.')
    plt.xlabel('Avg feat biasjaaba',fontsize=15)
    plt.ylabel('Avg features GT',fontsize=15)
    plt.title('Averaged Window Features',fontsize=20)
    #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/comp_avgwindowfeat_side.jpg')

    plt.figure(figsize=(15, 8))
    ax2=plt.gca()
    ax2.plot(hoghof_avg_front[:],hoghof_front_gt[:],'.')
    plt.xlabel('Avg feat biasjaaba',fontsize=15)
    plt.ylabel('Avg features GT',fontsize=15)
    plt.title('Averaged Window Features',fontsize=20)

    #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/comp_avgwindowfeat_front.jpg')
    plt.show()


if __name__ == "__main__":
    main()