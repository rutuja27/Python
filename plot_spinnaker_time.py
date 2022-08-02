import skipped_frames_correlation as sfc
import numpy as np
import matplotlib.pyplot as plt
import plotting_code as pc
import utils as ut
import lat_defs as ld

def main():


    # plotting variables

    fig, ax_handle = sfc.set_plot_var('', 1)
    shape = ['+', '*', 'x', '.', '^', 'o']
    color = ['r', 'b', 'g', 'm', 'c', 'k']
    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    title = 'Spinnaker Distribution of Latency'

    labels = ['Latency grabbing frame from camera - spinnaker backend']
    filelist = ['a22dc_6_27_2022', '256e8_7_12_2022']

    for id in range(0, len(filelist)):

        imagegrab_config_filedir = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/config_files/short/jaaba_plugin_multicamera_shorttrial_run_'

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

        # Read config file
        filename = imagegrab_config_filedir + filelist[id] + '.csv'
        sfc.readConfigFile(filename, Config)


        isPlugin = 0
        isJaaba = 0
        islogging = 0
        isCamOnly = 0

        ##mode to run in BIAS
        plugin_prefix = Config['plugin_prefix']
        logging_prefix = Config['logging_prefix']
        framegrab_prefix = Config['framegrab_prefix']
        print(plugin_prefix)

        latency_metric = ld.LatencyMetric(1, 0, 0, 0, 0)

        latency_threshold = 6.0
        numFrames = 0.0
        no_of_trials = 0.0

        for i in range(0,1):

            no_of_trials = Config['no_of_trials']
            numFrames = Config['numFrames']
            arr = ld.LatencyData(np.array(no_of_trials * [numFrames * [0.0]]), \
                                  np.array(no_of_trials * [numFrames * [0.0]]), \
                                  np.array(no_of_trials * [numFrames * [0.0]]), \
                                  np.array(no_of_trials * [numFrames * [0.0]]), \
                                  np.array(no_of_trials * [numFrames * [0.0]]), \
                                  np.array(no_of_trials * [numFrames * [0.0]]),\
                                  np.array(no_of_trials * [numFrames * [0.0]]))

            sfc.readLatency_data(arr, Config, latency_metric, framegrab_prefix, cam_id=0)
            pc.plot_single_trial(arr.lat_nidaq[0][1000:2000], '.', color[id], alpha[5], ax_handle, labels[0] + ' ' + filelist[id], Config)

    ax_handle.plot(np.array(latency_threshold * np.ones(1000)), \
                   label='_nolegend_')

    savefigfile = Config['dir_list'][0] + Config['nidaq_prefix'] + '/' + Config['cam_dir'] + '/' \
                  + Config['git_commit'] + '_' + Config['date'] + '/' + 'latency_camtoimagegrab_spinbackend_1000-2000frames.pdf'
    print(savefigfile)
    plt.xlabel('Frames', fontsize=12)
    plt.ylabel('Milliseconds ms', fontsize=12)
    plt.legend()
    plt.savefig(savefigfile)
    plt.show()

if __name__ == "__main__":
    main()