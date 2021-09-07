import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


path = '../dvs240_data_set/eventStruct_09032021_fly/bin_file/kdvs240_Opti_09032021.bin'

quat = np.fromfile(path,dtype=np.float64)

quat = quat.reshape(-1,5)
quat[:,0] = quat[:,0] - quat[0,0]

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

final_quat = np.array(final_quat)
r = R.from_quat(final_quat)
final_angle = r.as_euler('ZYX',degrees=True)


fig,(ax1,ax2,ax3) = plt.subplots(3)

ax1.plot(quat_time,final_angle[:,0],color='red')
ax1.set_title('Absolute angle around Z-axis (Deg)')
ax1.grid()

ax2.plot(quat_time,final_angle[:,1],color='blue')
ax2.set_title('Absolute angle around Y-axis (Deg)')
ax2.grid()

ax3.plot(quat_time,final_angle[:,2],color='green')
ax3.set_title('Absolute angle around X-axis (Deg)')
ax3.grid()
plt.show()