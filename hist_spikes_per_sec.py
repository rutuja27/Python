import csv
import matplotlib.pyplot as plt
import numpy as np
import plotting_code as pc
import read_csvconfigfile as rc

def main():

    config_file_path = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/config_files/short/'
    config_file_commits = ['d417d_11_9_2022','d417d_12_5_2022', 'd417d_12_12_2022', '3ed1c_11_29_2022','ba2f3_12_1_2022']
    bias_modes = ['imagegrab', 'jaaba']
    config_file_prefix = 'jaaba_plugin_multicamera_'
    config_file_suffix = '_shorttrial_run_'
    numCommits = len(config_file_commits)

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

    spikes_per_sec_nidaq_cam0 = [[] for i in range(0,numCommits)]
    spikes_per_sec_nidaq_cam1 = [[] for i in range(0, numCommits)]
    xtick = np.arange(0,numCommits*len(bias_modes))
    x= np.arange(0,len(bias_modes))

    fig, ax_handle = plt.subplots(figsize=(10,10))
    markersize = 6
    lab_names = ['Cam 0' , 'Cam 1']
    shape = ['+', '*', 'x', '.', '1', '2']
    color = ['r', 'b', 'g', 'm', 'c', 'k']
    aplha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    scalar_shifts = np.arange(0,numCommits)
    title = 'Spikes/sec comparison across commits '
    x_ticklabels = [ '3ed1c_11_30_2022','a3ed1c_11_30_2022', 'b3ed1c_11_30_2022']

    for i in range(0, numCommits):
        for mode_id in range(0,len(bias_modes)):
            config_file = config_file_path + config_file_prefix + bias_modes[mode_id] + config_file_suffix + config_file_commits[i] + '.csv'
            rc.readConfigFile(config_file, Config)
            print(float(Config['spikes_per_sec_nidaq'][0]))
            spikes_per_sec_nidaq_cam0[i].append(float(Config['spikes_per_sec_nidaq'][0]))
            spikes_per_sec_nidaq_cam1[i].append(float(Config['spikes_per_sec_nidaq'][1]))
        print('camera 0', spikes_per_sec_nidaq_cam0[i])
        print('camera 1', spikes_per_sec_nidaq_cam1[i])
        pp = pc.plot_bar(x, spikes_per_sec_nidaq_cam0[i],spikes_per_sec_nidaq_cam1[i], color=color[i],
                    alpha=1.0, width=0.2, scalar_shift=scalar_shifts[i], label=lab_names, ax_handle=ax_handle)
        for p in pp:
            height = p.get_height()
            ax_handle.text(x=p.get_x() + p.get_width()/2, s=config_file_commits[i], y=height)

    ytick0 = max(max(sublist)  for sublist in spikes_per_sec_nidaq_cam0)
    ytick1 = max(max(sublist) for sublist in spikes_per_sec_nidaq_cam1)
    ax_handle.set_ylabel('Spikes/sec')
    ax_handle.set_xticks(x)
    ax_handle.set_yticks(np.arange(0.0, ytick0+ytick1,0.01))
    ax_handle.set_xticklabels(bias_modes)
    ax_handle.legend(lab_names)
    ax_handle.set_title(title, fontweight='bold', fontsize=13)

    fig.tight_layout()
    #plt.savefig('C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/spikes_per_sec_comp_jaaba_totallatency.jpg')
    plt.show()



if __name__ == "__main__":
    main()
