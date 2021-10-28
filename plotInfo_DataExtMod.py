import rosbag
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# This script refer to the path folder defined below, it does not apply to other data folder path
path = '../kratos_quat_crazyflie_dataset/data_ext_mod/10272021/record_5_10272021.bag'

bag = rosbag.Bag(path)

data_ext_mod = []

for topic, msg, t in bag.read_messages(topics=['/crazyflie/data_ext_mod']):
    data_ext_mod.append(msg)

data_ext_mod_list = []

for i in range(len(data_ext_mod)):
    data_ext_mod_list.append([data_ext_mod[i].header.stamp.to_nsec()/1e9, data_ext_mod[i].conPadM3, data_ext_mod[i].conPadM4,data_ext_mod[i].padDir, 
                        data_ext_mod[i].offsetM3, data_ext_mod[i].offsetM4, data_ext_mod[i].heading, data_ext_mod[i].ztarget,
                        data_ext_mod[i].zcurrent, data_ext_mod[i].ycurrent, data_ext_mod[i].xcurrent, 
                        data_ext_mod[i].Pterm, data_ext_mod[i].Iterm, -data_ext_mod[i].Dterm])


data_ext_mod_arr = np.array(data_ext_mod_list)

data_ext_mod = data_ext_mod_arr[data_ext_mod_arr[:,0] > 0]

data_ext_mod[:,0] = data_ext_mod[:,0]-data_ext_mod[0,0]


ts = data_ext_mod[:,0]
conPadM3 = data_ext_mod[:,1]
conPadM4 = data_ext_mod[:,2]
padDir = data_ext_mod[:,3]
heading = data_ext_mod[:,6]
ztarget = data_ext_mod[:,7]
zcurrent = data_ext_mod[:,8]
ycurrent = data_ext_mod[:,9]
xcurrent = data_ext_mod[:,10]
Pterm = data_ext_mod[:,11]
Iterm = data_ext_mod[:,12]
Dterm = data_ext_mod[:,13]

for i in range(1,xcurrent.shape[0]):
    if xcurrent[i] == 0:
        xcurrent[i] = xcurrent[i-1]
    if ycurrent[i] == 0:
        ycurrent[i] = ycurrent[i-1]
    
    

# vel_z = np.diff(data_ext_mod[:,4])/0.1

fig, ax = plt.subplots(7,1,sharey=False)

ax[0].plot(ts,conPadM3,'r--')
ax[0].plot(ts,conPadM4,'b--')
ax[1].plot(ts,ztarget,'g--')
ax[1].plot(ts,zcurrent)
ax[2].plot(ts,heading)
ax[3].plot(ts,padDir)
ax[4].plot(ts,ycurrent)
ax[5].plot(ts,xcurrent)
ax[6].plot(xcurrent,ycurrent)

ax[0].legend(['M3','M4'])
ax[1].legend(['Z_desired','Z_current'])


ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[3].grid(True)
ax[4].grid(True)
ax[5].grid(True)
ax[6].grid(True)



ax[0].set_ylabel('ConPad')
ax[1].set_ylabel('Z axis(cm)')
ax[2].set_ylabel('Heading')
ax[3].set_ylabel('padDir')
ax[4].set_ylabel('Y axis(cm)')
ax[5].set_ylabel('X axis(cm)')








# fig, ax = plt.subplots(3,1,sharey=False)

# ax[0].plot(time_arr,roll)
# ax[1].plot(time_arr,pitch)
# ax[2].plot(time_arr,yaw)



plt.show()