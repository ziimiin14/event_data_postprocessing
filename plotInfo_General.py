import rosbag
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt

# This script refer to the path folder defined below, it does not apply to other data folder path
path = '../kratos_quat_crazyflie_dataset/data_vrpn_pos_ext/record_1_09302021.bag'

bag = rosbag.Bag(path)

opti = []
cmd_pos = []

for topic, msg, t in bag.read_messages(topics=['/vrpn_client_node/zm_kratos/pose','/crazyflie/cmd_position']):
    if topic == '/crazyflie/cmd_position':
        cmd_pos.append(msg)
    if topic == '/vrpn_client_node/zm_kratos/pose':
        opti.append(msg)

pose_list =[]
quat_list =[]
time_list=[]
cmd_pos_list = []

for i in range(len(opti)):
    pose_list.append([opti[i].pose.position.x,opti[i].pose.position.y,opti[i].pose.position.z])
    quat_list.append([opti[i].pose.orientation.w,opti[i].pose.orientation.x,opti[i].pose.orientation.y,opti[i].pose.orientation.z])
    time_list.append(opti[i].header.stamp.to_nsec()/1e9)

for j in range(len(cmd_pos)):
    cmd_pos_list.append([cmd_pos[j].header.stamp.to_nsec()/1e9,cmd_pos[j].x,cmd_pos[j].y])

time_arr = np.array(time_list)
pose_arr = np.array(pose_list)
quat_arr = np.array(quat_list)
cmd_pos_arr = np.array(cmd_pos_list)

time_arr = time_arr-time_arr[0]
pose_arr = pose_arr- pose_arr[0,:]
cmd_pos_arr[:,0] = cmd_pos_arr[:,0]-cmd_pos_arr[0,0]



yy = 2*((quat_arr[:,0]*quat_arr[:,1]) + (quat_arr[:,2]*quat_arr[:,3]))
xx = 1-2*(quat_arr[:,1]**2+quat_arr[:,2]**2)
roll = np.arctan2(yy,xx)
roll = roll *180/np.pi
# roll[roll<0] = 360 + roll[roll<0]

pitch = np.arcsin(2*((quat_arr[:,0]*quat_arr[:,2]) - (quat_arr[:,3]*quat_arr[:,1])))
pitch = pitch *180/np.pi
# pitch[pitch<0] = 360 + pitch[pitch<0]


yy2 = 2*((quat_arr[:,0]*quat_arr[:,3]) + (quat_arr[:,1]*quat_arr[:,2]))
xx2 = 1-2*(quat_arr[:,2]**2+quat_arr[:,3]**2)
yaw = np.arctan2(yy2,xx2)
yaw = yaw *180/np.pi
yaw[yaw<0] = 360 + yaw[yaw<0]

yaw_rate= []

for i in range(1,len(yaw)):
    nume = yaw[i]-yaw[i-1]
    deno = 0.005
    if nume < 0 :
        nume += 360
    temp = nume/deno
    # if temp < 0:
    #     temp = yaw_rate[-1]
    if temp > 4000:
        temp = yaw_rate[-1]
    yaw_rate.append(temp)

yaw_rate = np.array(yaw_rate)


fig, ax = plt.subplots(5,1,sharey=False)

ax[0].plot(time_arr,pose_arr[:,0])
ax[1].plot(time_arr,pose_arr[:,1])
ax[2].plot(time_arr,pose_arr[:,2])
ax[3].plot(time_arr[1:],yaw_rate)
ax[4].plot(cmd_pos_arr[:,0],cmd_pos_arr[:,1]*100/65500)


ax[0].grid(True)
ax[1].grid(True)
ax[2].grid(True)
ax[3].grid(True)
ax[4].grid(True)


# fig, ax = plt.subplots(3,1,sharey=False)

# ax[0].plot(time_arr,roll)
# ax[1].plot(time_arr,pitch)
# ax[2].plot(time_arr,yaw)



plt.show()