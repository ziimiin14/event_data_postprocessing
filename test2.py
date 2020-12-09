import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
from scipy.ndimage import gaussian_filter1d



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
cmgd = np.fromfile('../dynamic_data_06112020/dynamic_cmgd_bg_5sec.bin',dtype=np.float64)

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
quat = quat[::2]



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
cmgd = cmgd[0:499]
angle = np.linalg.norm(cmgd,axis=1)
axis_x = cmgd[:,0]/angle
axis_y = cmgd[:,1]/angle
axis_z = cmgd[:,2]/angle
axis_x = axis_x.reshape(-1,1)
axis_y = axis_y.reshape(-1,1)
axis_z = axis_z.reshape(-1,1)
axis = np.concatenate((axis_x,axis_y,axis_z),axis=1)
cmgd_x = cmgd[:,0]
cmgd_y = cmgd[:,1]
cmgd_z = cmgd[:,2]

# cmgd_quat = []
cmgd_angVel = []
for i in range(len(axis_x)):
    temp = R.from_rotvec(angle[i]*axis[i,:])
    temp_angle = temp.as_euler('ZYX')
    cmgd_angVel.append(temp_angle)



cmgd_angVel = np.array(cmgd_angVel)
cmgd_angVel = cmgd_angVel*180*100/np.pi
# cmgd_angVel = np.roll(cmgd_angVel[:],2,axis=0)
cmgd_angVel[2:,:] = cmgd_angVel[0:-2,:]

# temp_x = gaussian_filter1d(angVel_body[:-1,0],3)
# temp_y = gaussian_filter1d(angVel_body[:-1,1],3)
# temp_z = gaussian_filter1d(angVel_body[:-1,2],3)


# error_x = np.absolute(cmgd_angVel[:,2]-temp_x)
# mean_sumErr_x = np.sum(error_x)/5974
# err_x = mean_sumErr_x*100/max(temp_x)

# error_y = np.absolute(cmgd_angVel[:,1]-temp_y)
# mean_sumErr_y = np.sum(error_y)/5974
# err_y = mean_sumErr_y*100/max(temp_y)

# error_z = np.absolute(cmgd_angVel[:,0]-temp_z)
# mean_sumErr_z = np.sum(error_z)/5974
# err_z = mean_sumErr_z*100/max(temp_z)
# print(err_x,err_y,err_z)
# print(max(temp_x),max(temp_y),max(temp_z))
# print(err_y)




#Plotting
plt.plot(cmgd_time[0:499],cmgd_angVel[:,2],'--')
# plt.plot(cmgd_time[0:499],cmgd_angVel[:,1],'--')
# plt.plot(cmgd_time[0:499],cmgd_angVel[:,0],'--')
# plt.plot(cmgd_time[:-1],cmgd_angVel[:,1])
# plt.plot(cmgd_time[:-2],cmgd_angVel[:,1],'--',color='b')
plt.plot(cmgd_time[:-1],gaussian_filter1d(angVel_body[:,0],2.5),color='r')
# plt.plot(cmgd_time[:-1],gaussian_filter1d(angVel_body[:,1],2.5),color='r')
# plt.plot(cmgd_time[:-1],gaussian_filter1d(angVel_body[:,2],2.5),color='orange')
# plt.plot(cmgd_time[:-2],gaussian_filter1d(angVel_body[:-1,1],3),color='r')
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
