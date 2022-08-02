import plotting_code as pc
import utils as ut
import lat_defs as ld
import numpy as np
import matplotlib.pyplot as plt


def main():

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

    config_file = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/config_files/short/jaaba_plugin_multicamera_shorttrial_run_33cc6_8_1_2022.csv'
    ut.readConfigFile(config_file, Config)

    no_of_trials = Config['no_of_trials']
    numFrames = Config['numFrames']
    numCameras = Config['numCameras']


    # mode to run in BIAS
    plugin_prefix = Config['plugin_prefix']
    logging_prefix = Config['logging_prefix']
    framegrab_prefix = Config['framegrab_prefix']

    # set BIAS mode configuration flags
    islogging = ut.setflag(logging_prefix)
    isCamOnly = ut.setflag(framegrab_prefix)
    isPlugin = ut.setflag(plugin_prefix)

    if plugin_prefix == 'jaaba_plugin':
        isJaaba=1

    latency_metric = ld.LatencyMetric(1, 0, 0, 0, 0)
    #biasConfig_mode = ld.BiasConfigMode(isCamOnly, islogging, isPlugin, isJaaba)

    if isPlugin:
        if isJaaba:
            bias_config = Config['plugin_prefix']
    cam_id=0
    latency_data_jaaba_cam1 = ld.LatencyData(np.array(no_of_trials * [numFrames * [0.0]]),\
                                          np.array(no_of_trials * [numFrames * [0.0]]),\
                                          np.array(no_of_trials * [numFrames * [0.0]]),\
                                          np.array(no_of_trials * [numFrames * [0.0]]),\
                                          np.array(no_of_trials * [numFrames * [0.0]]),\
                                          np.array(no_of_trials * [numFrames * [0.0]]),\
                                          np.array(no_of_trials * numFrames * [0.0]))

    ut.readLatency_data(latency_data_jaaba_cam1, Config, latency_metric,
                        bias_config, cam_id)
    if numCameras == 2:
        latency_data_jaaba_cam2 = ld.LatencyData(np.array(no_of_trials * [numFrames * [0.0]]),\
                                              np.array(no_of_trials * [numFrames * [0.0]]),\
                                              np.array(no_of_trials * [numFrames * [0.0]]),\
                                              np.array(no_of_trials * [numFrames * [0.0]]),\
                                              np.array(no_of_trials * [numFrames * [0.0]]),\
                                              np.array(no_of_trials * [numFrames * [0.0]]),\
                                              np.array(no_of_trials * numFrames * [0.0]))

        cam_id=1
        ut.readLatency_data(latency_data_jaaba_cam2, Config, latency_metric,
                        bias_config, cam_id)

    latency_data_jaaba_cam1.lat_total = np.add(latency_data_jaaba_cam1.lat_nidaq,
                                               latency_data_jaaba_cam1.lat_process_time)
    fig, ax_handle = plt.subplots(1,1)
    norm_factor = no_of_trials * numFrames
    #pc.plot_cdf(latency_data_jaaba_cam1.lat_process_time, norm_factor, ax_handle)
    #pc.plot_cdf(latency_data_jaaba_cam1.lat_nidaq, norm_factor, ax_handle)
    pc.plot_cdf(latency_data_jaaba_cam1.lat_total, norm_factor, ax_handle)
    plt.xlabel('Latency in ms')
    plt.ylabel('CDF')
    plt.title('Cumulative Density Function of Jaaba Latency')
    plt.show()


if __name__ == "__main__":
    main()
