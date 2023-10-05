import numpy as np
import utils as ut
import sys
import matplotlib.pyplot as plt

def plot_jaaba_latency_profile(filepath, jaaba_lat_prefix_1, jaaba_lat_prefix_2, numRuns, numFrames,
                               start_range, end_range):

    rangeVal = np.arange(start_range,end_range)
    print(rangeVal)

    jaaba_latnecy_profile_1 = np.array(numFrames * [0])
    jaaba_latnecy_profile_2 = np.array(numFrames* [0])
    legend_handle = []
    color = ['C0', 'C1', 'C2']
    shape = ['.', '2']
    legend_handle = ['cuda no set device', 'cuda set device']

    for i in range(0,numRuns):

        jaaba_lat_prefix_file_1 = filepath + jaaba_lat_prefix_1 + str(i) + '.csv'
        jaaba_lat_prefix_file_2 = filepath + jaaba_lat_prefix_2 + str(i) + '.csv'

        ut.readcsvFile_int(jaaba_lat_prefix_file_1, jaaba_latnecy_profile_1, 1, 0)
        ut.readcsvFile_int(jaaba_lat_prefix_file_2, jaaba_latnecy_profile_2, 1, 0)

        plt.plot(jaaba_latnecy_profile_1[rangeVal]/1000.0,shape[0], alpha=0.3 , color=ut.adjust_lightness(color[i],1.3))
        plt.plot(jaaba_latnecy_profile_2[rangeVal]/1000.0, shape[1], alpha=0.8, color=ut.adjust_lightness(color[i],0.8))

    plt.legend(legend_handle)
    plt.xlabel('Frames')
    plt.ylabel('Time in ms')
    plt.title('Jaaba latency profile with and without cuda device set')

def main():

    print('Number of arguments', len(sys.argv))
    print('Argument list', str(sys.argv))

    if(len(sys.argv) < 8):
        print('Insufficient arguments')
        print('Argument options\n' +
              '-filepath\n' +
              '-jaaba per frame latency offline 1 prefix\n' +
              '-jaaba per frame latency offline 2 prefix\n' +
              '-number of runs\n' +
              '-numFrames per run\n' +
              '-start_range of frames to plot\n' +
              '-end_range of frame to plot\n'
              )
    else:
        filepath = sys.argv[1]
        jaaba_lat_prefix_1 = sys.argv[2]
        jaaba_lat_prefix_2 = sys.argv[3]
        numRuns = np.int(sys.argv[4])
        numFrames = np.int(sys.argv[5])
        start_range = np.int(sys.argv[6])
        end_range = np.int(sys.argv[7])

        plot_jaaba_latency_profile(filepath, jaaba_lat_prefix_1, jaaba_lat_prefix_2, numRuns, numFrames,
                                   start_range, end_range)

        plt.show()


if __name__ == "__main__":
    main()