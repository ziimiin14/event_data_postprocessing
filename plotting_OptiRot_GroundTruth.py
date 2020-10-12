import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d

# data = np.loadtxt('6300ERPM_2000_Opti_Tf.txt',delimiter=',')
quat = np.fromfile('../bin_file/3780ERPM_Quat.bin',dtype=np.float64)
time = np.fromfile('../bin_file/3780ERPM_Time.bin',dtype=np.int64)
quat = quat.reshape(int(len(quat)/4),4)
time = time.reshape(-1,1)
time = time/1e9
# time = data[:,0]
# #time = time/1e9
# quat = data[:,1:5]
# quat = quat[:-2]
r = R.from_quat(quat)
euler = r.as_euler('xyz',degrees=True)
yaw = euler[:,2]
#cum_angle = np.cumsum(angle)
neg = np.where(yaw<0)
yaw[neg] += 360


diff_angle = np.diff(yaw)

# find negative values
neg = np.where(diff_angle<-1)
diff_angle[neg] += 360

# time_mod = np.arange(time[0],time[-1],1/180)
# diff_time = np.diff(time_mod)
diff_time = np.diff(time,axis=0)
diff_time = diff_time.reshape(-1)
cum_time = np.cumsum(diff_time)

angular_vel = diff_angle/diff_time


angular_vel_mod = gaussian_filter1d(angular_vel, 5.0)

plt.plot(cum_time,angular_vel,color='red')
plt.xlabel('Time(s)')
plt.ylabel('Angular velocity(deg/s)')
plt.grid()
plt.show()
