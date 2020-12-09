import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt


def t(x):
    return np.tan(np.deg2rad(x))

def c(x):
    return np.cos(np.deg2rad(x))

def s(x):
    return np.sin(np.deg2rad(x))

def sc(x):
    return 1/c(x)

# data = np.loadtxt('6300ERPM_2000_Opti_Tf.txt',delimiter=',')
quat = np.fromfile('../dynamic_data_06112020/dynamic_groundtruth.bin',dtype=np.float64)
# cmgd = np.fromfile('../cmgd_all_tg.bin',dtype=np.float64)
cmgd = np.fromfile('../dynamic_data_06112020/cmgd_all_mg.bin',dtype=np.float64)

imu = np.fromfile('../dynamic_data_06112020/imu.bin',dtype=np.float64)

# Reshape and tranpose the array
cmgd = cmgd.reshape(3,int(cmgd.shape[0]/3))
quat = quat.reshape(5,int(quat.shape[0]/5))
imu=imu.reshape(7,int(imu.shape[0]/7))
cmgd = cmgd.T
quat = quat.T
imu = imu.T

# Assign imu ang_vel to g
gx = imu[:,4]*180/np.pi
gy = imu[:,5]*180/np.pi
gz = imu[:,6]*180/np.pi
imu_time = imu[:,0]

# Slicing the array into correct dimension so that cmgd and quat can match each other
cmgd = cmgd[:-1]
quat = quat[:-3]


# Assign specific terms
quat_only = quat[:,1:]
quat_time = quat[:,0]
cmgd_time=np.arange(quat_time[0],quat_time[-1],0.01)

r = R.from_quat(quat_only)
euler = r.as_euler('ZYX',degrees=True)

# Conver euler (zyx) to euler(xyz)
# euler[:,[0,2]] = euler[:,[2,0]]
euler_x = euler[:,2]
euler_y = euler[:,1]
euler_z = euler[:,0]

diff_eul_x = np.diff(euler_x)
diff_eul_y = np.diff(euler_y)
diff_eul_z = np.diff(euler_z)

diff_time = np.diff(quat_time)

angVel_x = diff_eul_x/diff_time
angVel_y = diff_eul_y/diff_time
angVel_z = diff_eul_z/diff_time
angVel_x = np.deg2rad(angVel_x)
angVel_y = np.deg2rad(angVel_y)
angVel_z = np.deg2rad(angVel_z)

angVel_world= np.vstack((angVel_x,angVel_y,angVel_z)).T

x = euler_x
y = euler_y
z = euler_z
angVel_body = []

for i in range(euler_x.shape[0]-1):
    T = [ [1 , s(x[i+1])*t(y[i+1]) ,  c(x[i+1])*t(y[i+1])] , [0 , c(x[i+1]) , -s(x[i+1])] , [0 , s(x[i+1])/c(y[i+1]) , c(x[i+1])/c(y[i+1])] ]
    # T = [ [1 , s(x[i])*t(y[i]) ,  c(x[i])*t(y[i])] , [0 , c(x[i]) , -s(x[i])] , [0 , s(x[i])/c(y[i]) , c(x[i])/c(y[i])] ]
    T = np.array(T)
    temp = np.dot(np.linalg.inv(T),angVel_world[i])
    angVel_body.append(temp)

angVel_body = np.array(angVel_body)
angVel_body = angVel_body*180/np.pi

#CMGD
angle = np.linalg.norm(cmgd,axis=1)
axis_x = cmgd[:,0]/angle
axis_y = cmgd[:,1]/angle
axis_z = cmgd[:,2]/angle
axis_x = axis_x.reshape(-1,1)
axis_y = axis_y.reshape(-1,1)
axis_z = axis_z.reshape(-1,1)
axis = np.concatenate((axis_x,axis_y,axis_z),axis=1)
# cmgd_x = cmgd[:,0]
# cmgd_y = cmgd[:,1]
# cmgd_z = cmgd[:,2]

# cmgd_quat = []
cmgd_angVel = []
for i in range(len(axis_x)):
    temp = R.from_rotvec(angle[i]*axis[i,:])
    temp_angle = temp.as_euler('ZYX')
    cmgd_angVel.append(temp_angle)
    # temp = Quaternion(axis=[axis_x[i],axis_y[i],axis_z[i]],angle=angle[i])
    # cmgd_angle.append(quat_temp.yaw_pitch_roll)
    # cmgd_quat.append([temp[1],temp[2],temp[3],temp[0]])

# cmgd_quat = np.array(cmgd_quat)
# r = R.from_quat(cmgd_quat)
# cmgd_angVel = r.as_euler('ZYX',degrees=True)


cmgd_angVel = np.array(cmgd_angVel)
cmgd_angVel = cmgd_angVel*180*100/np.pi


temp_y = gaussian_filter1d(angVel_body[::2,1],3)
# error=np.absolute((cmgd_angVel[1400:1900,1]-temp_y[1402:1902])/temp_y[1402:1902])*100
# print(np.sum(error)/500)
error = np.absolute(cmgd_angVel[:,1]-temp_y)
mean_sumErr = np.sum(error)/5974
error = mean_sumErr*100/500
print(error)
#Plotting
# plt.plot(cmgd_time[:-1],cmgd_angVel[:,1])
plt.plot(cmgd_time[:-1],cmgd_angVel[:,1],'--',color='b')
# plt.plot(cmgd_time[:-1],gaussian_filter1d(angVel_body[:,0],2.5),color='b')
plt.plot(cmgd_time[:-1],gaussian_filter1d(angVel_body[::2,1],3),color='r')
# plt.plot(cmgd_time[:-1],gaussian_filter1d(angVel_body[::2,2],2.5),color='orange')
# plt.plot(cmgd_time[:-1],medfilt(angVel_body[::2,0],7),'-',color='b')
# plt.plot(cmgd_time[:-1],medfilt(angVel_body[::2,1],7),'-',color='r')
# plt.plot(cmgd_time[:-1],medfilt(angVel_body[::2,2],7),'-',color='orange')
# plt.ylim(-600,600)

# plt.plot(quat_time[:-1],angVel_world[:,2]*180/np.pi)

# plt.plot(imu_time,gx,color='b')
# plt.plot(imu_time,gy,color='r')
# plt.plot(imu_time,gz,color='orange')


# plt.plot(quat_time,euler[:,0],color='orange')
# plt.plot(quat_time,euler[:,1],color='r')
# plt.plot(quat_time,euler[:,2],color='b')
plt.show()
