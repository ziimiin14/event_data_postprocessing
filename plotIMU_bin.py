import matplotlib.pyplot as plt
import numpy as np


path = '../dvs240_data_set/eventStruct_07162021_fly/bin_file/kdvs240_IMU_07162021_1.bin'

imu = np.fromfile(path,dtype=np.float64)

imu = imu.reshape(-1,4)

imu[:,0] = imu[:,0] - imu[0,0]



fig,(ax1,ax2,ax3) = plt.subplots(3)

ax1.plot(imu[:,0],imu[:,3],color='red')
ax1.set_title('Angular Velocity around Z-axis (rad/s)')
ax1.grid()

ax2.plot(imu[:,0],imu[:,2],color='blue')
ax2.set_title('Angular Velocity around Y-axis (rad/s)')
ax2.grid()

ax3.plot(imu[:,0],imu[:,1],color='green')
ax3.set_title('Angular Velocity around X-axis (rad/s)')
ax3.grid()
plt.show()