import matplotlib.pyplot as plt
import numpy as np
import utils as rcs
import skipped_frames_correlation as sfc
import plotting_code as pc




def main():

  filedir = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/043cb_10_27_2022/'
  filesavepath = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/figs/'
  no_of_trials = 5
  numFrames = 2498
  f2f_flag=0
  cam_id=0
  plugin_prefix=''

  imagegrab_process_time_cam0 = np.array(no_of_trials*[numFrames * [0.0]])
  imagegrab_process_time_cam1 = np.array(no_of_trials*[numFrames * [0.0]])

  jaaba_process_time_cam0 = np.array(no_of_trials*[numFrames * [0.0]])
  jaaba_process_time_cam1 = np.array(no_of_trials*[numFrames * [0.0]])

  imagegrab_start_time_cam0 = np.array(no_of_trials*[numFrames * [0.0]])
  imagegrab_start_time_cam1 = np.array(no_of_trials*[numFrames * [0.0]])

  jaaba_start_time_cam0 = np.array(no_of_trials*[numFrames * [0.0]])
  jaaba_start_time_cam1 = np.array(no_of_trials*[numFrames * [0.0]])

  jaaba_endtoend_cam0 = np.array(no_of_trials*[numFrames * [0.0]])
  jaaba_endtoend_cam1 = np.array(no_of_trials*[numFrames * [0.0]])

  fig, ax_handle = pc.set_plot_var(0, no_of_trials, numFrames, 8.0)
  markersize=6
  labels = ['JAABA end to end process time']
  shape = ['+', '*', 'x', '.', '1', '2']
  color = ['r', 'b', 'g', 'm', 'c', 'k']
  alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
  title = 'JAABA end to end process time '

  latency_threshold = 5

  for trial_type in range(1,no_of_trials+1):
      print(trial_type)
      img_proc_cam0 = filedir + 'imagegrab_process_timecam0_short_trial' + str(trial_type) + '.csv'
      img_proc_cam1 = filedir + 'imagegrab_process_timecam1_short_trial' + str(trial_type) + '.csv'

      img_start_cam0 = filedir + 'imagegrab_start_timecam0_short_trial' + str(trial_type) + '.csv'
      img_start_cam1 = filedir + 'imagegrab_start_timecam1_short_trial' + str(trial_type) + '.csv'

      jaaba_proc_cam0 = filedir + 'jaaba_plugin_process_timecam0_short_trial' + str(trial_type) + '.csv'
      jaaba_proc_cam1 = filedir + 'jaaba_plugin_process_timecam1_short_trial' + str(trial_type) + '.csv'

      jaaba_end_cam0 = filedir + 'jaaba_plugin_start_timecam0_short_trial' + str(trial_type) + '.csv'
      jaaba_end_cam1 = filedir + 'jaaba_plugin_start_timecam1_short_trial' + str(trial_type) + '.csv'

      rcs.readcsvFile_f2f(img_proc_cam0, imagegrab_process_time_cam0[trial_type-1], f2f_flag, 0,'')
      rcs.readcsvFile_f2f(img_proc_cam1, imagegrab_process_time_cam1[trial_type-1], f2f_flag, 1,'')

      rcs.readcsvFile_float(img_start_cam0, imagegrab_start_time_cam0[trial_type-1], 0,'')
      rcs.readcsvFile_float(img_start_cam1, imagegrab_start_time_cam1[trial_type-1], 1,'')

      rcs.readcsvFile_f2f(jaaba_proc_cam0, jaaba_process_time_cam0[trial_type-1],f2f_flag, 0,'')
      rcs.readcsvFile_f2f(jaaba_proc_cam1, jaaba_process_time_cam1[trial_type-1],f2f_flag, 1,'')

      rcs.readcsvFile_float(jaaba_end_cam0, jaaba_start_time_cam0[trial_type-1], 0,'')
      rcs.readcsvFile_float(jaaba_end_cam1, jaaba_start_time_cam1[trial_type-1], 1,'')

      jaaba_endtoend_cam0[trial_type-1] = (jaaba_start_time_cam0[trial_type-1] - imagegrab_start_time_cam0[trial_type-1])*0.001
      jaaba_endtoend_cam1[trial_type - 1] = (jaaba_start_time_cam1[trial_type - 1] - imagegrab_start_time_cam1[
          trial_type - 1])*0.001

      print(jaaba_start_time_cam0[1] - imagegrab_start_time_cam0[1])

  pc.plot_raw_data(jaaba_endtoend_cam0, jaaba_endtoend_cam1,shape, color, alpha, labels, ax_handle, \
                  no_of_trials, latency_threshold, numFrames, title, \
                  0,markersize)
  plt.show()

  plt.figure()
  ax2 = plt.gca()
  jaaba = list(jaaba_endtoend_cam0.flatten())
  pc.plot_histogram(jaaba, ax2)
  plt.title('JAABA end to end latency')
  plt.xlabel('Latency in ms')
  plt.savefig(filesavepath + 'jaaba_end_to_end_biasvideo.jpg')
  plt.show()


if __name__ == "__main__":
    main()