import numpy as np


data = np.loadtxt('../dynamic_data_06112020/shape_rotation_groundtruth.txt')

time = data[:,0]
time = time.reshape(-1,1)

quat = np.concatenate((time, data[:,4:]),axis=1)

print(quat.shape)

# quat.tofile('../dynamic_data_06112020/shape_rotation_groundtruth.bin')
quat.tofile('../dynamic_data_06112020/shape_rotation_groundtruth.bin')


