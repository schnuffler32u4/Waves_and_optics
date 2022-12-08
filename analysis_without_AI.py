import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.style.use('extensys-ms')

data = pd.read_csv('Measurements/Measurement1_01_12.csv')
data.rename(columns={"Time (s) Run #1": "T", "Light Intensity (lx) Run #1": "Light"}, inplace=True)
data.drop(data[data.Light > 260].index, inplace=True)

time = np.array(data.T)
time = time[0]
light = np.array(data.Light)
# light = light[0]
detectime = []

for i in range(1,len(time)-1):
    if light[i-1] >= light[i] <= light[i+1]:

        detectime.append(time[i])

#print(time)
detectime = np.array(detectime)
print(detectime)

intervals = detectime[1:] - detectime[0:-1]
print(intervals)

#plt.plot(detectime[1:], intervals)
#plt.show()

print(os.listdir("Measurements"))

finalx = np.zeros((len(os.listdir('Measurements'))))
finaly = np.zeros(len(finalx))
j = 0

for file in os.listdir('Measurements'):
    data = pd.read_csv("Measurements/"+file)
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
    while finalx[i] > 2*np.pi:
        finalx[i] = finalx[i] - 2*np.pi

xint = []
yint = []

for m in range(len(finalx)):
    if finalx[m] < 3 and finaly[m] > 20:
        xint.append(finalx[m])
        yint.append(finaly[m])

finalx = xint
finaly = yint

plt.scatter(finalx, finaly, color='green')

plt.xlabel('Initial phase difference [rad]')
plt.ylabel('Synchronization time [s]')
theta = np.arange(0, np.pi + np.pi/4, step=(np.pi/4))
# theta = [0, np.pi/4, np.pi/2, np.pi, 3*np.pi/2]
#theta = [0,1/4,1/2,3/4,1, 5/4, 3/2, 7/4, 2]
plt.xticks(theta, ['0', 'π/4', 'π/2', '3π/4', 'π'])
plt.show()



