import numpy as np
import cv2
import matplotlib.pyplot as plt

def data_load(path):
    #obj = np.load('../test/6300ERPM/static_rotation_6300ERPM_polEvents.npz')
    #obj = np.load('../test/6300ERPM/static_rotation_6300ERPM_time.npz')
    obj = np.load(path)
    namelist = obj.zip.namelist()
    obj.zip.extract(namelist[0])
    data = np.load(namelist[0],allow_pickle=True)
    return data


imu = data_load('../test/Rotation/6300ERPM/imu.npz')

if ((imu.shape[0]%8) == 0):
    imu = imu.reshape(int(imu.shape[0]/8),8)

else:
    remain = imu.shape[0]%8
    imu = imu[remain:]
    imu = imu.reshape(int(imu.shape[0]/8),8)

imu = imu.astype('float32')
ang_vel = imu[:,4:7]
time = imu[:,0]
# ang_vel = np.array([[0,0,0]],dtype=np.float32)
# time = np.array([[0,]],dtype=np.float32)

# for k,v in data.items():
#     if k == 0:
#         pass
#     else:
#         ang_vel = np.insert(ang_vel,ang_vel.shape[0],v[:,4:7],axis=0)
#         time = np.insert(time,time.shape[0],v[:,0].reshape(v[:,0].shape[0],1),axis=0)

time = time/1e6


fig,(ax1,ax2,ax3) = plt.subplots(3)


ax1.plot(time,ang_vel[:,0],color='red')
ax1.set_title('Angular Velocity around X-axis (degree/s)')
ax1.grid()
ax2.plot(time,ang_vel[:,1],color='blue')
ax2.set_title('Angular Velocity around Y-axis (degree/s)')
ax2.grid()
ax3.plot(time,ang_vel[:,2],color='green')
ax3.set_title('Angular Velocity around Z-axis (degree/s)')
ax3.grid()
plt.show()
