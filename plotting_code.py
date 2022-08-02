import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stat


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

