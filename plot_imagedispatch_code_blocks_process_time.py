import skipped_frames_correlation as sfc
import numpy as np
import matplotlib.pyplot as plt


def plot_data(arr, shape, color, alpha, ax_handle, label, Config):

    ax_handle.plot(arr[:], '.', \
                   color=color, \
                   marker=shape, \
                   alpha=0.6, ms=12, label=label)

def readImagedispatchData(Config, arr1):

    no_of_trials = Config['no_of_trials']
    numFrames = Config['numFrames']
    numCameras = Config['numCameras']
    plugin_prefix = Config['plugin_prefix']

    isPlugin = 0
    isJaaba = 0
    islogging = 0
    isCamOnly = 0

    ## mode to run in BIAS
    plugin_prefix = Config['plugin_prefix']
    logging_prefix = Config['logging_prefix']
    framegrab_prefix = Config['framegrab_prefix']

    ## set BIAS mode configuration flags
    islogging = sfc.setFlags(logging_prefix)
    isCamOnly = sfc.setFlags(framegrab_prefix)
    isPlugin = sfc.setFlags(plugin_prefix)

    if plugin_prefix == 'jaaba_plugin':
        isJaaba = 1

    latency_metric =sfc.LatencyMetric(1, 1, 0, 1)

    biasConfig_mode = sfc.BiasConfigMode(isCamOnly, islogging, isPlugin, isJaaba)

    bias_config = 'imagedispatch'
    print(bias_config)
    cam_id = 0
    sfc.readLatency_data(arr1, Config, latency_metric, \
                     bias_config, cam_id)



def main():

    filelist = ['ace32_6_9_2022', 'a959a_6_10_2022', 'b67a7_6_13_2022']

    imagedispatch_config_filedir = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/config_files/short/jaaba_plugin_multicamera_shorttrial_run_'

    Config = {

        'filename': '',
        'numCameras': 0,
        'cam_suffix': [],
        'dir_len': 0,
        'dir_list': [],
        'numFrames': 0,
        'no_of_trials': 0,
        'framerate': 0,
        'latency_threshold': 0.0,
        'cam_dir': '',
        'nidaq_prefix': '',

        'f2f_prefix': '',
        'queue_prefix': '',
        'plugin_prefix': '',
        'logging_prefix': '',
        'framegrab_prefix': '',
        'git_commit': '',
        'date': '',
        'count_latencyspikes_nidaq': [],
        'average_normspikes_nidaq': [],
        'mean_spikes_nidaq': [],
        'std_spikes_nidaq': [],
        'spikes_per_sec_nidaq': [],
        'max_skippedFrames_nidaq': 0,
        'fracIntwspikes_nidaq': [],
        'count_latencyspikes_f2f': [],
        'average_normspikes_f2f': [],
        'mean_spikes_f2f': [],
        'std_spikes_f2f': [],
        'spikes_per_sec_f2f': [],
        'fracIntwspikes_f2f': [],

    }

    # plotting variables

    fig, ax_handle = sfc.set_plot_var('',1)
    shape= ['+', '*', 'x', '.', '^', 'o']
    color = ['r', 'b','g', 'm', 'c', 'k']
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    title= 'Latencies from different threads Camera '
    labels = ['pop and front from queue', 'queue wait to grab', 'nidaq latency from cam to Imagediaptch' ]

    latency_threshold = 6.0
    numFrames = 0.0
    no_of_trials = 0.0

    for i in range(0,2):

        filename = imagedispatch_config_filedir + filelist[i] + '.csv'
        sfc.readConfigFile(filename, Config)
        no_of_trials = Config['no_of_trials']
        numFrames = Config['numFrames']
        arr = sfc.LatencyData(np.array(no_of_trials*[numFrames * [0.0]]), \
                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                np.array(no_of_trials*[numFrames * [0.0]]),\
                                np.array(no_of_trials*[numFrames * [0.0]]))
        readImagedispatchData(Config, arr)
        plot_data(arr.lat_f2f[1], shape[i], color[i], alpha[i+2], ax_handle, labels[i], Config)
        if i==1:
            plot_data(arr.lat_nidaq_filt[1], shape[i+1], color[i+1], alpha[i+3], ax_handle, labels[i+1], Config)
    ax_handle.plot(np.array(latency_threshold * np.ones(numFrames)), \
                   label='_nolegend_')
    plt.xlabel('Frames', fontsize=12)
    plt.ylabel('Milliseconds ms', fontsize=12)
    plt.legend()
    plt.show()




    # plot data
    #plot_matching_data(latency_data_imagegrab_cam1, latency_data_imagedispatch_cam1,
    #               latency_data_jaaba_cam1, Config, 0)
    #plot_matching_data(latency_data_imagegrab_cam2, latency_data_imagedispatch_cam2,
    #               latency_data_jaaba_cam2, Config, 1)

if __name__ == "__main__":
    main()