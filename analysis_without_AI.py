import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
import os
import time as runtime

plt.style.use('extensys')
progress = 0
itime = runtime.time()


def fitfunction(x, T, phi, amp):
    return abs(np.sin(x * 2 * np.pi / T + phi)) * amp + 0.5


for file in os.listdir('Measurements'):
    data = pd.read_csv("Measurements/" + file)
    data.rename(columns={"Time (s) Run #1": "Time", "Light Intensity (lx) Run #1": "Light"}, inplace=True)
    data.drop(data[data.Light > 260].index, inplace=True)

    time = np.array(data.Time)
    light = np.array(data.Light)
    detectime = []

    for i in range(1, len(time) - 1):
        if light[i - 1] >= light[i] <= light[i + 1]:
            detectime.append(time[i])

    detectime = np.array(detectime)
    intervals = detectime[1:] - detectime[0:-1]

    median = np.median(intervals)
    median = 0.5

    popt, pcov = curve_fit(fitfunction, detectime[1:], [i if i > median else 1 - i for i in intervals], p0=[405, 0, 0.5], maxfev=10000)

    fity = [fitfunction(i, *popt) for i in detectime]

    if progress != 0:
        eta = str((runtime.time() - itime) * (len(os.listdir('Measurements'))-progress) / progress)[:3] + "s"
    else:
        eta = "Not applicable"
    print(file + " " + str(progress/len(os.listdir('Measurements'))*100)[:4] + "%" + " ETA " + eta)
    plt.scatter(detectime[1:], intervals)
    plt.plot(detectime, np.ones(len(detectime)) * median, color='red')
    plt.plot(detectime, fity, color='cyan')
    # plt.plot(detectime, fitfunction(detectime, *pcov), color='cyan')
    plt.xlabel("Time [s]")
    plt.ylabel("Interval between detection [s]")
    plt.title(file + " " + "Amplitude period: " + str(popt[0]) + "±" + str(np.sqrt(pcov[0][0])))
    plt.savefig("Graphs/" + file + ".png", dpi=500)
    plt.clf()
    progress = progress + 1
print(os.listdir("Measurements"))

finalx = np.zeros((len(os.listdir('Measurements'))))
finaly = np.zeros(len(finalx))
j = 0

for file in os.listdir('Measurements'):
    data = pd.read_csv("Measurements/" + file)
    print(file)
    data.rename(columns={"Time (s) Run #1": "Time", "Light Intensity (lx) Run #1": "Light"}, inplace=True)
    data.drop(data[data.Light > 260].index, inplace=True)

    time = np.array(data.Time)
    # time = time[0]
    light = np.array(data.Light)
    detectime = []

    for i in range(1, len(time) - 1):
        if light[i - 1] >= light[i] <= light[i + 1]:
            detectime.append(time[i])

    detectime = np.array(detectime)
    intervals = detectime[1:] - detectime[0:-1]

    finalx[j] = intervals[0]
    finaly[j] = detectime[np.where(intervals == np.amax(intervals))[0][0]]
    j = j + 1

finalx = finalx * 2 * np.pi / 1.98

for i in range(len(finalx)):
    while finalx[i] > 2 * np.pi:
        finalx[i] = finalx[i] - 2 * np.pi

xint = []
yint = []

for m in range(len(finalx)):
    if finalx[m] < 3 and finaly[m] > 20:
        xint.append(finalx[m])
        yint.append(finaly[m])

finalx = xint
finaly = yint


# Linear least sqaures fit

def linear(x, a, b):
    return np.multiply(x, a) + b


popt, pcov = curve_fit(linear, finalx, finaly, bounds=(0, 1e5))
print(popt[0], np.sqrt(np.diag(pcov))[0])

# Plotting

plt.scatter(finalx, finaly, color='green')
# plt.plot(finalx, linear(finalx, *popt))

plt.xlabel('Initial phase difference [rad]')
plt.ylabel('Synchronization time [s]')
theta = np.arange(0, np.pi + np.pi / 4, step=(np.pi / 4))
# theta = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
# theta = [0,1/4,1/2,3/4,1, 5/4, 3/2, 7/4, 2]
plt.xticks(theta, ['0', 'π/4', 'π/2', '3π/4', 'π'])
plt.title("Initial phase difference versus synchronization time")
plt.savefig("Phasevstime.png")
plt.clf()
