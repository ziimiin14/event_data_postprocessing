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


quat = np.fromfile('../../dynamic_data_06112020/dynamic_groundtruth.bin',dtype=np.float64)


quat = quat.reshape(5,int(quat.shape[0]/5)) # this is for dynamic groundtruth
quat = quat.T
quat_only = quat[:,1:]
quat_time = quat[:,0]
quat_conj = quat_only.copy()
quat_conj[:,:-1] = -quat_conj[:,:-1]

#Rearrange the order from x y z w to w x y z
quat_only = np.roll(quat_only,1,axis=1)
quat_conj = np.roll(quat_conj,1,axis=1)

first_quat_conj = Quaternion(quat_conj[0,:])
first_quat = Quaternion(quat_only[0,:])

final_quat= []


for i in range(quat_conj.shape[0]):
    temp = first_quat_conj*Quaternion(quat_only[i,:])
    final_quat.append([temp[1],temp[2],temp[3],temp[0]])

# for i in range(quat_only.shape[0]):
#     temp = Quaternion(quat_only[i,:])
#     final_quat.append([temp[1],temp[2],temp[3],temp[0]])

final_quat = np.array(final_quat)
r = R.from_quat(final_quat)
final_angle = r.as_euler('ZYX',degrees=True)



plt.plot(quat_time[:],final_angle[:,0],color='orange')
plt.plot(quat_time[:],final_angle[:,1],color='r')
plt.plot(quat_time[:],final_angle[:,2],color='b')


plt.xlabel('Time(s)')
plt.ylabel('Euler angles vel(deg/s)')
plt.show()
