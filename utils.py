import csv
import re
import numpy as np

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

def readcsvFile_nidaq(filename, arr_lat, arr_cam, cam_id, plugin_prefix):
    # if cam_id == 1 and plugin_prefix == 'jaaba_plugin':
    #    return

    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')

    for idx, row in enumerate(data_grab):

        if cam_id == 0:

            arr_cam[idx] = ((np.float(row[0])))  ## will store the count corresponding to camera trigger
            arr_lat[idx] = (((np.float(row[1]) - np.float(row[0])) * 0.02))  ## latency calculation between
            ## event and camera trigger, fast clock is 50khz
            ## hence multiplying factor is (1/50khz- period) 0.02 to calculate latency

        else:
            arr_lat[idx] = (((np.float(row[1]) - arr_cam[idx]) * 0.02))

    fhandle.close()


def readcsvFile_int(filename, arr, cam_id, plugin_prefix):
    # if cam_id == 1 and plugin_prefix == 'jaaba_plugin':
    #    return

    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')

    for idx, row in enumerate(data_grab):
        arr[idx] = np.int(row[0])
    fhandle.close()


def readcsvFile_float(filename, arr, cam_id, plugin_prefix):
    fhandle = open(filename)
    data_grab = csv.reader(fhandle, delimiter=',')

    for idx, row in enumerate(data_grab):
        arr[idx] = np.float(row[0])
    fhandle.close()


def readcsvFile_f2f(filename, arr, f2f_flag, cam_id, plugin_prefix):
    # if cam_id == 1 and plugin_prefix == 'jaaba_plugin':
    #    return

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




