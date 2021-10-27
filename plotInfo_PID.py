import rosbag
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# This script refer to the path folder defined below, it does not apply to other data folder path
path = '../kratos_quat_crazyflie_dataset/data_pid/10202021/record_16_10202021.bag'

bag = rosbag.Bag(path)

data_ext = []

for topic, msg, t in bag.read_messages(topics=['/crazyflie/data_ext']):
    data_ext.append(msg)

data_ext_list = []

for i in range(len(data_ext)):
    data_ext_list.append([data_ext[i].header.stamp.to_nsec()/1e9, data_ext[i].conPad, data_ext[i].offset, data_ext[i].ztarget, data_ext[i].zcurrent,
                        data_ext[i].Pterm, data_ext[i].Iterm, data_ext[i].Dterm])


data_ext_arr = np.array(data_ext_list)

data_ext_mod = data_ext_arr[data_ext_arr[:,0] > 0]

data_ext_mod[:,0] = data_ext_mod[:,0]-data_ext_mod[0,0]


vel_z = np.diff(data_ext_mod[:,4])/0.1

fig, ax = plt.subplots(6,1,sharey=False)

ax[0].plot(data_ext_mod[:,0],data_ext_mod[:,1])
ax[1].plot(data_ext_mod[:,0],data_ext_mod[:,3],'g--')
ax[1].plot(data_ext_mod[:,0],data_ext_mod[:,4])
ax[2].plot(data_ext_mod[:,0],data_ext_mod[:,5])
ax[3].plot(data_ext_mod[:,0],data_ext_mod[:,6])
ax[4].plot(data_ext_mod[:,0],-data_ext_mod[:,7])
ax[5].plot(data_ext_mod[:-1,0],vel_z)


ax[1].legend(['Z_desired','Z_current'])


ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[3].grid(True)
ax[4].grid(True)
ax[5].grid(True)


ax[0].set_ylabel('ConPad')
ax[1].set_ylabel('Altitude(cm)')
ax[2].set_ylabel('Pterm')
ax[3].set_ylabel('Iterm')
ax[4].set_ylabel('Dterm')
ax[5].set_ylabel('Velocity_Z')








# fig, ax = plt.subplots(3,1,sharey=False)

# ax[0].plot(time_arr,roll)
# ax[1].plot(time_arr,pitch)
# ax[2].plot(time_arr,yaw)



plt.show()