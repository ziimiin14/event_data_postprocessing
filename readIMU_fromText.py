import numpy as np


imu = np.loadtxt('../dynamic_data_06112020/shape_rotation_imu.txt')

# time = data[:,0]
# time = time.reshape(-1,1)

# quat = np.concatenate((time, data[:,4:]),axis=1)

print(imu.shape)

# quat.tofile('../dynamic_data_06112020/shape_rotation_groundtruth.bin')
imu.tofile('../dynamic_data_06112020/shape_rotation_imu.bin')


