import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat
import matplotlib as mt


def set_plot_var(queue_prefix, no_of_trials, xlim, ylim):
    if (no_of_trials > 1):

        fig, axes = plt.subplots(no_of_trials, 1, figsize=(12, 10))
        ax = axes[0].get_gridspec()

    else:
        fig, axes = plt.subplots(1, 1, figsize=(12, 10))
        ax = axes.get_gridspec()

    fig.subplots_adjust(hspace=0.8)

    # if no_of_trials/ > 1:
    #     axes[ax.nrows-1, ax.ncols-2] = plt.subplot2grid((rows, cols), (2,0), colspan = 2)

    if queue_prefix:
        plt.setp(axes, yticks=np.arange(0, ylim, 1), ylim=[0, ylim])
    else:
        plt.setp(axes, yticks=np.arange(1, ylim, 1), ylim=[1,ylim])

    return fig, axes


def plot_raw_data(arr1,arr2,shape, color, alpha, labels, ax_handle, \
                  no_of_trials, latency_threshold, numFrames, title, \
                  cam_id, marker_size):
    if cam_id == 0:
        color_id = 0
    else:
        color_id = 3

    if (no_of_trials > 1):

        ax = ax_handle[0].get_gridspec()

        for ix in range(0, ax.nrows):

            if cam_id == 0:
                if (ix) == 0:
                    ax_handle[ix].plot(arr1[ix], '.',
                                       color=color[color_id], \
                                       marker=shape[ix], \
                                       alpha=1, ms=marker_size, label='Cam' + str(cam_id))
                    ax_handle[ix].plot(arr2[ix], '.',
                                       color=color[color_id+1], \
                                       marker=shape[ix], \
                                       alpha=1, ms=marker_size, label='Cam' + str(cam_id))
                else:
                    ax_handle[ix].plot(arr1[ix], '.',
                                       color=color[color_id], \
                                       marker=shape[ix], \
                                       alpha=1, ms=marker_size)
                    ax_handle[ix].plot(arr2[ix], '.',
                                       color=color[color_id+1], \
                                       marker=shape[ix+1], \
                                       alpha=1, ms=marker_size)
            ax_handle[ix].set_title('Trial ' + str(ix + 1), fontsize=8, fontweight='bold')
            ax_handle[ix].tick_params(axis='x', labelsize=8)
            ax_handle[ix].tick_params(axis='y', labelsize=8)
            # ax_handle[ix].plot(np.array(latency_threshold*np.ones(numFrames)),\
            #                      label='_nolegend_')
            ax_handle[ix].set_xlabel('Frames', fontsize='8')
            ax_handle[ix].set_ylabel('Milliseconds', fontsize='8')
        plt.suptitle(title, fontsize=17)

    else:
        # ax = ax_handle.get_gridspec()
        ax_handle.plot(arr1[0][:], '.', \
                       color=color[0], \
                       marker=shape[0], \
                       alpha=0.4, ms=marker_size, label='Cam' + str(cam_id))
        ax_handle.plot(arr1[0][:], '.', \
                       color=color[1], \
                       marker=shape[1], \
                       alpha=0.6, ms=marker_size, label='Cam' + str(cam_id))
        ax_handle.plot(arr1[0][:], '.', \
                       color=color[2], \
                       marker=shape[2], \
                       alpha=0.8, ms=marker_size, label='Cam' + str(cam_id))
        ax_handle.plot(np.array(latency_threshold * np.ones(25)), \
                       label='_nolegend_')
        plt.suptitle(title + str(cam_id), fontsize=17)
        plt.setp(ax_handle, xlabel='Frames', ylabel='Milliseconds')
        #plt.legend(legend)

def plot_raw_single_axis(arr1,arr2,shape, color, alpha, labels, ax_handle, \
                  no_of_trials, latency_threshold, numFrames, title, \
                  cam_id, marker_size):
    if cam_id == 0:
        color_id = 0
    else:
        color_id = 3

    if (no_of_trials > 1):

        ax = ax_handle[0].get_gridspec()

        for ix in range(0, ax.nrows):

            if cam_id == 0:
                if (ix) == 0:
                    ax_handle[ix].plot(arr1[ix], '.',
                                       color=color[color_id], \
                                       marker=shape[ix], \
                                       alpha=1, ms=marker_size, label='Cam' + str(cam_id))
                    ax_handle[ix].plot(arr2[ix], '.',
                                       color=color[color_id+1], \
                                       marker=shape[ix], \
                                       alpha=1, ms=marker_size, label='Cam' + str(cam_id))
                else:
                    ax_handle[ix].plot(arr1[ix], '.',
                                       color=color[color_id], \
                                       marker=shape[ix], \
                                       alpha=1, ms=marker_size)
                    ax_handle[ix].plot(arr2[ix], '.',
                                       color=color[color_id+1], \
                                       marker=shape[ix+1], \
                                       alpha=1, ms=marker_size)
            ax_handle[ix].set_title('Trial ' + str(ix + 1), fontsize=8, fontweight='bold')
            ax_handle[ix].tick_params(axis='x', labelsize=8)
            ax_handle[ix].tick_params(axis='y', labelsize=8)
            # ax_handle[ix].plot(np.array(latency_threshold*np.ones(numFrames)),\
            #                      label='_nolegend_')
            ax_handle[ix].set_xlabel('Frames', fontsize='8')
            ax_handle[ix].set_ylabel('Milliseconds', fontsize='8')

        plt.suptitle(title, fontsize=17)
        plt.legend(labels)

    else:

        for ix in range(0, 5):
           ax_handle.plot(arr1[ix], '.', \
                       color=color[ix], \
                       marker=shape[ix], \
                       alpha=0.4, ms=marker_size, label='Cam' + str(cam_id))
        ax_handle.tick_params(axis='x', labelsize=10)
        ax_handle.tick_params(axis='y', labelsize=10)
           # ax_handle[ix].plot(np.array(latency_threshold*np.ones(numFrames)),\
           #                      label='_nolegend_')
        ax_handle.set_xlabel('Frames', fontsize='12')
        ax_handle.set_ylabel('Milliseconds', fontsize='12')
        plt.suptitle(title, fontsize=17)

def plot_single_trial(arr, shape, color, alpha, ax_handle, label, Config):

    ax_handle.plot(arr[:], '.',\
                   color=color,\
                   marker=shape,\
                   alpha=0.6, ms=4, label=label)


def plot_cdf(arr, norm_factor, plt_handle):

    bins = np.arange(np.min(arr), np.max(arr), 1, dtype=int)

    count, bins_count = np.histogram(arr, bins=len(bins))
    print(bins_count)
    print(count)
    pdf = count / np.sum(count)
    cdf = np.cumsum(pdf)
    plt_handle.plot(bins, cdf, marker='o')

def plot_histogram(arr, plt_handle):

    bins = np.arange(np.floor(np.min(arr)), np.ceil(np.max(arr)), 0.5, dtype=float)
    count, bins_count = np.histogram(arr, bins=len(bins))
    count = count / sum(count)
    plt.bar(bins, count, width=0.4)

def plot_bar(arr_keys,arr_vals1, arr_vals2, color, alpha, width, scalar_shift, label, ax_handle):

     pp1 =ax_handle.bar(arr_keys + (scalar_shift * width), arr_vals1,color=color, width=width, label=label[0])
     pp2 = ax_handle.bar(arr_keys+(scalar_shift*width), arr_vals2, bottom=arr_vals1, color=color,alpha=0.1, width=width, label=label[1])
     return pp2
