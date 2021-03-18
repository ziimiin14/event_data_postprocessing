import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

# def quaternion_multiply(quaternion1, quaternion0):
#     w0, x0, y0, z0 = quaternion0
#     w1, x1, y1, z1 = quaternion1
#     return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
#                      x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
#                      -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
#                      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)


# data = np.loadtxt('6300ERPM_2000_Opti_Tf.txt',delimiter=',')
quat = np.fromfile('../dynamic_data_06112020/shape_rotation_groundtruth.bin',dtype=np.float64)
# cmgd = np.fromfile('../cmgd_all_tg.bin',dtype=np.float64)
cmgd = np.fromfile('../dynamic_data_06112020/cmgd_all_tg_1.bin',dtype=np.float64)

cmgd = cmgd.reshape(3,int(cmgd.shape[0]/3))
# quat = quat.reshape(5,int(quat.shape[0]/5)) # this is for dynamic groundtruth
quat = quat.reshape(int(quat.shape[0]/5),5) # this is for shape rotation groundtruth
cmgd = cmgd.T
# quat = quat.T

cmgd = cmgd[:-1]
quat = quat[:-3]



quat_only = quat[:,1:]
quat_time = quat[:,0]
quat_conj = quat_only.copy()
quat_conj[:,:-1] = -quat_conj[:,:-1]

# cmgd = cmgd*180*100/np.pi
# Compute theta and axis (x,y,z)
angle = np.linalg.norm(cmgd,axis=1)
axis_x = cmgd[:,0]/angle
axis_y = cmgd[:,1]/angle
axis_z = cmgd[:,2]/angle
cmgd_x = cmgd[:,0]
cmgd_y = cmgd[:,1]
cmgd_z = cmgd[:,2]

cmgd_angle = []

for i in range(len(axis_x)):
    quat_temp = Quaternion(axis=[axis_x[i],axis_y[i],axis_z[i]],angle=angle[i])
    cmgd_angle.append(quat_temp.yaw_pitch_roll)


cmgd_angle = np.array(cmgd_angle)
cmgd_angle = cmgd_angle*180*100/np.pi




# x_prev = 0
# y_prev = 0
# z_prev = 0
# t = 10/1000 # 10ms

# cmgd_abs_x =[x_prev]
# cmgd_abs_y =[y_prev]
# cmgd_abs_z = [z_prev]


# for i in range(cmgd.shape[0]-1):
#     temp_x = cmgd_x[i+1]*t + cmgd_abs_x[i]
#     temp_y = cmgd_y[i+1]*t + cmgd_abs_y[i]
#     temp_z = cmgd_z[i+1]*t + cmgd_abs_z[i]
#     cmgd_abs_x.append(temp_x)
#     cmgd_abs_y.append(temp_y)
#     cmgd_abs_z.append(temp_z)

cmgd_time=np.arange(quat_time[0],quat_time[-1],0.01)

#Rearrange the order from x y z w to w x y z
quat_only = np.roll(quat_only,1,axis=1)
quat_conj = np.roll(quat_conj,1,axis=1)

first_quat_conj = Quaternion(quat_conj[0,:])
first_quat = Quaternion(quat_only[0,:])

final_quat= []

for i in range(quat_conj.shape[0]):
    temp = first_quat_conj*Quaternion(quat_only[i,:])
    final_quat.append([temp[1],temp[2],temp[3],temp[0]])

final_quat = np.array(final_quat)
r = R.from_quat(final_quat)
final_angle = r.as_euler('ZYX',degrees=True)




quat_modified  = []
# first = Quaternion(quat_conj[0,:])*first_quat
# quat_modified.append(first)

for i in range(quat_conj.shape[0]-1):
    temp = Quaternion(quat_conj[i+1,:])*Quaternion(quat_only[i,:])
    quat_modified.append(temp.yaw_pitch_roll)


quat_modified = np.array(quat_modified)
euler_modified = quat_modified

# for i in range(quat_modified.shape[0]):
#     euler_modified.append(quat_modified[i].yaw_pitch_roll)

euler_modified = np.array(euler_modified)
diff_time = np.diff(quat_time)

euler_z_modified = euler_modified[:,0]/diff_time
euler_y_modified = euler_modified[:,1]/diff_time
euler_x_modified = euler_modified[:,2]/diff_time

euler_x_modified = euler_x_modified*180/np.pi
euler_y_modified = euler_y_modified*180/np.pi
euler_z_modified = euler_z_modified*180/np.pi


# ans_temp =[]

# for i in range(quat_conj.shape[0]-1):
#     nextTemp = Quaternion(quat_conj[i+1,:])
#     prevTemp = Quaternion(quat_only[i,:])
#     ans = prevTemp*nextTemp
#     # final_angle.append(ans.yaw_pitch_roll)
#     ans_temp.append(ans)

# final_angle = []
# final_angle.append(ans_temp[0].yaw_pitch_roll)
# updatedTemp = ans_temp[0]
# for i in range(len(ans_temp)-1):
#     updatedTemp *= ans_temp[i+1]
#     final_angle.append(updatedTemp.yaw_pitch_roll)
#     #print(i)
    
# Convert from rad to degree




## Now try to convert the absolute angle to angular velocity (deg/s)
#Assign rotation angle about x,y,z
rot_x = final_angle[:,2]
rot_y = final_angle[:,1]
rot_z = final_angle[:,0]

#Find the difference of angle and time
diff_rot_x = rot_x[1:]-rot_x[:-1]
diff_rot_y = rot_y[1:]-rot_y[:-1]
diff_rot_z = rot_z[1:]-rot_z[:-1]
diff_time = quat_time[1:]-quat_time[:-1]

# diff_rot_x = np.diff(rot_x)
# diff_rot_y = np.diff(rot_y)
# diff_rot_z = np.diff(rot_z)
# diff_time = np.diff(quat_time)

#Calculate angular vel(deg/s)
angVel_x = diff_rot_x/diff_time
angVel_y = diff_rot_y/diff_time
angVel_z = diff_rot_z/diff_time

# Calculate the ground truth max and min for each axis(x,y,z)
temp_x =  angVel_x.reshape(-1,1)
temp_y =  angVel_y.reshape(-1,1)
temp_z =  angVel_z.reshape(-1,1)

temp_zyx = np.concatenate((temp_z,temp_y,temp_x),axis=1)
temp_zyx = temp_zyx / 100
res = R.from_euler('zyx',temp_zyx,degrees=True)
res_aa = res.as_rotvec()
z_max = res_aa[:,0].max()
z_min = res_aa[:,0].min()
y_max = res_aa[:,1].max()
y_min = res_aa[:,1].min()
x_max = res_aa[:,2].max()
x_min = res_aa[:,2].min()

print(z_max,z_min,y_max,y_min,x_max,x_min)
# time = data[:,0]
# #time = time/1e9
# quat = data[:,1:5]
# quat = quat[:-2]
# r = R.from_quat(quat_only)
# euler = r.as_euler('xyz',degrees=True)
# euler_x = euler[:,0]
# euler_y = euler[:,1]
# euler_z = euler[:,2]

plt.plot(quat_time[:],final_angle[:,0],color='orange')
plt.plot(quat_time[:],final_angle[:,1],color='r')
plt.plot(quat_time[:],final_angle[:,2],color='b')
# plt.plot(quat_time[:-1],angVel_x,color='r')
# plt.plot(quat_time[:-1],angVel_y,color='b')
# plt.plot(quat_time[:-1],angVel_z,color='orange')
# plt.plot(quat_time[:-1],-euler_z_modified)
# plt.plot(cmgd_time[:-1],cmgd_angle[:,0])
# plt.plot(cmgd_time[:-1],cmgd_x*180*100/np.pi,color='r')
# plt.xlim(0,60)
# plt.ylim(-600,600)

# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(quat_time[:],-final_angle[:,1],color='r')
# ax2.plot(quat_time[:-1],angVel_z,color='orange')
# fig.show()

# temp_y = angVel_y[::2]
# error = np.absolute((cmgd_angle[:,1]-temp_y)/temp_y)*100
# # error = np.absolute(cmgd_angle[:,1]-temp_y)
# error_mean=np.sum(error)/error.shape[0]
# print(error_mean)

plt.xlabel('Time(s)')
plt.ylabel('Euler angles vel(deg/s)')
plt.show()
# plt.grid()
# plt.show()

# plt.scatter()