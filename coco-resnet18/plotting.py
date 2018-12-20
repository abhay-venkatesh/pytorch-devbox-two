from matplotlib import colors
from matplotlib import pyplot
from matplotlib.ticker import PercentFormatter
import csv
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat

# Get data and compute times
data_times = []
compute_times = []
with open("logs.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data_times.append(float(row[0]))
        compute_times.append(float(row[1]))

def twoplots(data_times, compute_times, font_size):
    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    fig.set_figheight(2.5)
    fig.set_figwidth(5)

    axs[0].set_title("Data Time", fontsize=font_size)
    axs[0].set_xlabel("Frequency", fontsize=font_size)
    axs[0].set_ylabel("Time (s)", fontsize=font_size)
    axs[0].hist(data_times, bins=32, color="skyblue")
    # axs[0].xaxis.set_tick_params(labelsize=14)
    # axs[0].yaxis.set_tick_params(labelsize=14)
    print("Data Time statistics:")
    print("Mean = " + str(stat.mean(data_times)))
    print("Median = " + str(stat.median(data_times)))
    print("Standard Deviation = " + str(stat.stdev(data_times)))

    axs[1].set_title("Compute Time", fontsize=font_size)
    axs[1].set_xlabel("Frequency", fontsize=font_size)
    axs[1].set_ylabel("Time (s)", fontsize=font_size)
    axs[1].hist(compute_times, bins=32, color="pink")
    # axs[1].xaxis.set_tick_params(labelsize=14)
    print("Compute Time statistics:")
    print("Mean = " + str(stat.mean(compute_times)))
    print("Median = " + str(stat.median(compute_times)))
    print("Standard Deviation = " + str(stat.stdev(compute_times)))

    # plt.show()
    plt.savefig("coco-resnet-18-data-vs-compute.png")
    
def oneplot(data_times, compute_times):
    pyplot.hist(data_times, 32, alpha=0.5, label='Data Times')
    pyplot.hist(compute_times, 32, alpha=0.5, label='Compute Times')
    pyplot.legend(loc='upper right')
    pyplot.show()

twoplots(data_times, compute_times, 14)