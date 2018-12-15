import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import csv
import statistics as stat

n_bins = 20

# Get data and compute times
data_times = []
compute_times = []
with open("logs.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data_times.append(float(row[0]))
        compute_times.append(float(row[1]))


fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

axs[0].set_title("Data Time")
axs[0].set_xlabel("Frequency")
axs[0].set_ylabel("Time")
axs[1].set_title("Compute Time")
axs[1].set_xlabel("Frequency")
axs[1].set_ylabel("Time")

axs[0].hist(data_times, bins=10)
axs[1].hist(compute_times, bins=10)
print("Data Time statistics:")
print("Mean = " + str(stat.mean(data_times)))
print("Median = " + str(stat.median(data_times)))

print("Standard Deviation = " + str(stat.stdev(data_times)))
print("Compute Time statistics:")
print("Mean = " + str(stat.mean(compute_times)))
print("Median = " + str(stat.median(compute_times)))
print("Standard Deviation = " + str(stat.stdev(compute_times)))


plt.savefig("histogram.png")