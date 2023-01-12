import numpy as np
import csv

def main():

   dir = 'C:/Users/27rut/BIAS/misc/jaaba_plugin_day_trials/plugin_latency/nidaq/multi/fcbdf_8_29_2022/'
   filename = 'imagegrab_process_timecam1_short_trial'
   data_arr = np.array(5*[2495*[0.0]])
   print(data_arr)

   for trial in range(0,5):
       fd = ''
       fd = dir + filename + str(trial+1) + '.csv'
       print(fd)
       count=0
       with open(fd) as csvfile:
           file_reader = csv.reader(csvfile, delimiter=',')
           for row in file_reader:
               data_arr[trial][count] = np.float(row[0])
               count = count + 1

   mean_read_time = np.mean(np.mean(data_arr,1),0)
   std_read_time = np.std(np.std(data_arr,1),0)
   print('Mean ', mean_read_time)
   print('Std ', std_read_time)







if __name__ == "__main__":
    main()