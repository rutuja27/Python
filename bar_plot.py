import matplotlib.pyplot as plt

def main():

    beh_classification_time = [120,2]
    ax = plt.gca()
    x_val = ['Offline', 'Online']
    plt.ylabel('Latency ms')
    plt.bar(x_val,beh_classification_time, width=0.3)
    plt.show()

if __name__ == "__main__":
        main()