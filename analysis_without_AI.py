import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

plt.plot(detectime[1:], intervals)
plt.show()
#for i in range(len(detectime)):


# time, intensity = np.loadtxt("Measurements/Measurement1_01_12.csv", delimiter=',', skiprows=1, unpack=True, encoding='UTF-8')

