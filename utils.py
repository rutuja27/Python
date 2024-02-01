import csv
import re
import numpy as np
import colorsys

def readConfigFile(filename, config):
    with open(filename, 'r', newline='') as f:

        config_reader = csv.reader(f, delimiter=',')
        keys = list(config.keys())  ### keys in configdata structure
        rows = [[col for col in row] for row in config_reader]  ## key-value pair in csv file

        if len(rows) == len(keys):
            pass
        else:
            print(len(rows))
            print(len(keys))
            print('key-value pair unbalanced')

        for idx, row in enumerate(rows):

            for idy, col in enumerate(row):

                if idy == 0:
                    if row[idy] == keys[idx]:
                        continue
                    else:
                        print(row[idx])
                        break
                if type(config[keys[idx]]) is str:
                    if keys[idx] == 'git_commit':
                        col = col[1:]
                    config[keys[idx]] = str(col)
                elif type(config[keys[idx]]) is list:
                    col = col.strip("[, ], ' ")
                    col = re.split(',', col)
                    if col[0] == '':
                        config[keys[idx]] = []
                    else:
                        config[keys[idx]] = col
                elif type(config[keys[idx]]) is float:
                    config[keys[idx]] = float(col)
                elif type(config[keys[idx]]) is int:
                    config[keys[idx]] = int(col)
                else:
                    continue;

def readcsvFile_nidaq(filename, arr_cam, arr_lat, numFrames, period_ms):

    with open(filename, 'r', newline='') as f:
        frm_id=0
        config_reader = csv.reader(f, delimiter=',')
        for idx,row in enumerate(config_reader):
            if idx < numFrames:
                arr_cam[frm_id] = float(row[0])*period_ms

                arr_lat[frm_id] = (float(row[1])*period_ms)
                frm_id += 1

def readcsvFile_int(filename, arr, numFrames, scaling_factor, row_id):

    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')
    for idx, row in enumerate(data_grab):
        if idx < numFrames:
            arr[idx] = np.int(row[row_id])/scaling_factor

    fhandle.close()

def readcsvFile_int64(filename, arr, numFrames, scaling_factor, row_id):

    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')

    for idx, row in enumerate(data_grab):
        if idx < numFrames:
            arr[idx] = np.uint64(row[row_id])/scaling_factor

    fhandle.close()

def readcsvFile_float(filename, arr, numFrames):
    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')

    for idx, row in enumerate(data_grab):
        if idx < numFrames:
            arr[idx] = np.float(row[0])
    fhandle.close()


def readcsvFile_f2f(filename, arr, numFrames, f2f_flag, cam_id, plugin_prefix):
    # if cam_id == 1 and plugin_prefix == 'jaaba_plugin':
    #    return
    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')
    for idx, row in enumerate(data_grab):
        if idx < numFrames:
            if f2f_flag:
                if idx == 0:
                    prev = np.float(row[0])
                else:
                    arr[idx] = (np.float(row[0]) - prev) / 1000
                    prev = np.float(row[0])
            else:
                arr[idx] = np.float(row[0]) / 1000

    fhandle.close()

def readLatency_data(lat_dat, testconfig, lat_metric, biasmode_prefix, \
                     cam_id):
    no_of_trials = np.int(testconfig['no_of_trials'])
    numFrames = np.int(testconfig['numFrames'])
    plugin_prefix = testconfig['plugin_prefix']
    f2f_flag = 0

    path_dir = testconfig['dir_list'][0]

    if (numFrames > 100000):
        trial_suffix = '_long_trial'
    else:
        trial_suffix = '_short_trial'

    for i in range(0, no_of_trials):

        ## read latency readings from nidaq
        if lat_metric.isnidaq:
            print(biasmode_prefix)
            filename = path_dir + testconfig['nidaq_prefix'] \
                       + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
                       testconfig['date'] + '/' + biasmode_prefix + '_' + \
                       testconfig['nidaq_prefix'] + 'cam' + testconfig['cam_suffix'][cam_id] \
                       + trial_suffix + str(i + 1) + '.csv'
            print(filename)
            readcsvFile_nidaq(filename, lat_dat.lat_nidaq[i], lat_dat.lat_camtrig[i], \
                              cam_id, plugin_prefix)

        # read latency frame to frame from pc timings
        if lat_metric.isframetoframe:

            if f2f_flag:

                filename = path_dir + testconfig['f2f_prefix'] \
                           + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
                           testconfig['date'] + '/' + biasmode_prefix + '_' + \
                           testconfig['f2f_prefix'] + 'cam' + testconfig['cam_suffix'][cam_id] \
                           + trial_suffix + str(i + 1) + '.csv'

                readcsvFile_f2f(filename, lat_dat.lat_f2f[i], 1, cam_id, plugin_prefix)


        # read queue size
        if lat_metric.isqueue:
            filename = path_dir + testconfig['queue_prefix'] \
                       + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
                       testconfig['date'] + '/' + biasmode_prefix + '_' + \
                       testconfig['queue_prefix'] + 'cam' + \
                       testconfig['cam_suffix'][cam_id] + trial_suffix + str(i + 1) + '.csv'

            readcsvFile_int(filename, lat_dat.lat_queue[i], cam_id, plugin_prefix)

        if lat_metric.isnidaqThres:
            filename = path_dir + testconfig['nidaq_prefix'] \
                       + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
                       testconfig['date'] + '/' + biasmode_prefix + '_' + 'nidaq_thres' + 'cam' + \
                       testconfig['cam_suffix'][cam_id] + trial_suffix + str(i + 1) + '.csv'

            # print(filename)
            readcsvFile_float(filename, lat_dat.lat_nidaq_filt[i], cam_id, plugin_prefix)

        if lat_metric.isProcessTime:
            filename = path_dir + testconfig['nidaq_prefix'] \
                       + '/' + testconfig['cam_dir'] + '/' + testconfig['git_commit'] + '_' + \
                       testconfig['date'] + '/' + biasmode_prefix + '_' + 'process_time' + 'cam' + \
                       testconfig['cam_suffix'][cam_id] + trial_suffix + str(i + 1) + '.csv'

            readcsvFile_f2f(filename, lat_dat.lat_process_time[i], 0, cam_id, plugin_prefix)


def setflag(flag_prefix):

    if flag_prefix != '':
        return 1
    else:
        return 0


## distance - distance between peaks
## height -  definition of height of a peak
def maxPeak(latency_arr, latency_filt_arr, height):

    peaks = []
    numFrames = len(latency_arr)
    i = 0

    while i < numFrames:
        bout_count = 0
        if latency_arr[i] >= height:
            peaks.append(i)
            latency_filt_arr[i] = latency_arr[i]
            i += 1
            while i < numFrames and latency_arr[i] >= height:
                latency_filt_arr[i] = 2.2
                bout_count += 1
                i += 1
        else:
            latency_filt_arr[i] = latency_arr[i]
            i += 1

    # ## get filtered indexes
    # for aa in peaks:
    #     latency_filt_arr[aa] = latency_arr[aa]

    return [len(peaks), peaks]

## index is the column to read from score file
def read_score(filename,arr_scr, numFrames, index):

    with open(filename, 'r', newline='') as f:
        config_reader = csv.reader(f, delimiter=',')
        for idx,row in enumerate(config_reader):
            if idx < numFrames:
               if not row[0].isnumeric():
                   continue
               else:
                  arr_scr[idx-1] = np.float(row[index])
    f.close()

def read_score_view(filename, arr_scr_view, arr_scr, arr_scr_side,
                    arr_scr_front, numFrames, flag_gt, index):

    if flag_gt:
        total_lines_to_read = numFrames
    else:
        total_lines_to_read = numFrames + 1

    with open(filename, 'r', newline='') as f:
        config_reader = csv.reader(f, delimiter=',')
        for idx,row in enumerate(config_reader):
            if idx < total_lines_to_read:
                if(not flag_gt):
                    if (idx == 0):
                        continue
                    arr_scr_view[idx-1] = np.int(row[index])
                    if(arr_scr_view[idx-1] == 1):
                        arr_scr_side[idx-1] = np.float(row[3])
                    elif(arr_scr_view[idx-1] == 2):
                        arr_scr_front[idx-1] = np.float(row[3])
                    else:
                        arr_scr[idx-1] = np.float(row[3])
    f.close()

def readScoreData(filename, scr_obj, numFrames, flag_gt, numBehs):

    if flag_gt:
        total_lines_to_read = numFrames
    else:
        total_lines_to_read = numFrames + 1

    total_cols = 3 + numBehs +  2;#first three columns in the score file are the score timestamps
                                     ## last two columns are the framecount and the view

    with open(filename, 'r', newline='') as f:
        config_reader = csv.reader(f, delimiter=',')
        for idx,row in enumerate(config_reader):
            if idx < total_lines_to_read:
                if not flag_gt:
                    if idx ==0:
                        continue
                    else:
                        #first three columns in the score file are the score timestamps
                        scr_obj.score_ts[idx - 1] = np.float(row[0])
                        scr_obj.score_side_ts[idx - 1] = np.float(row[1])
                        scr_obj.score_front_ts[idx - 1] = np.float(row[2])
                        #scr_obj.scores[idx - 1][0] = np.float(row[3])

                        ## last two columns are the framecount and the view
                        scr_obj.frameCount[idx - 1] = np.int(row[total_cols-1])
                        scr_obj.view[idx -1 ] = np.int(row[total_cols-1])

def readArray(filename, arr, index):

    with open(filename, 'r', newline='') as f:
        config_reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(config_reader):
            arr.append(np.float(row[index]))

    f.close()

#https://saturncloud.io/blog/darken-or-lighten-a-color-in-matplotlib-a-guide/
def adjust_lightness(color, amount):

    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

def read_classifierparams(filename, cls_model, paramdims):

    with open(filename, 'r', newline='') as f:
        config_reader = csv.reader(f, delimiter=',')
        for idx,val in enumerate(config_reader):
            if(idx < paramdims):
                cls_model[0][idx] = np.float(val[0])
                cls_model[1][idx] = np.float(val[1])
                cls_model[2][idx] = np.float(val[2])
                cls_model[3][idx] = np.float(val[3])
                cls_model[4][idx] = np.float(val[4])

    f.close()

