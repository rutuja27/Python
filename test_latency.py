import numpy as np
import csv
import matplotlib.pyplot as plt


def main():

    file = 'C:/Users/Public/Documents/National Instruments/NI-DAQ/Examples/DAQmx ANSI C/Counter/Count Digital Events/' \
           'Cnt-Buf-Cont-ExtClk/x64/Release/latency.csv'
    file_handle = open(file, 'r+');
    lat_read = csv.reader(file_handle, delimiter=',')
    delay_test_nidaq = []

    for idx,row in enumerate(lat_read):
        delay_test_nidaq.append(float(row[0]))
    file_handle.close()

    plt.plot(delay_test_nidaq, '.')
    plt.show()

if __name__ == "__main__":
    main()