import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
from scipy.signal import medfilt

path = "C:/Users/Zechen Zhang/Desktop//BMGF/Phase1_data/Ground_truth/BP_raw_1KHz/F001-T1-BP.txt"
temp = np.loadtxt(path)
print(temp.shape)
med_filted_temp = medfilt(temp, 41)
print(med_filted_temp.shape)
# tt_frame = int((temp.shape[0] / 1000 * 25) // 10 * 10)
tt_frame = int(temp.shape[0] // 10 * 10)

print(tt_frame)

avg_temp = np.zeros(tt_frame)
for i in range(0, int(tt_frame / 40)):
    avg_temp[40 * i:40 * (i + 1)] = mean(temp[40 * i:40 * (i + 1)])

order_temp = np.zeros(tt_frame)
for i in range(0, int(tt_frame / 40)):
    order_temp[40 * i:40 * (i + 1)] = temp[i * 40]

plt.plot(temp[0:2000], label="original")
# plt.plot(avg_temp[0:2000], label="mean")
plt.plot(med_filted_temp[0:2000],label="median filtered")
# plt.plot(order_temp[0:2000], label='40-based filter')
plt.legend()
plt.show()
