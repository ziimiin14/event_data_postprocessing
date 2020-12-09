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

def readBin(path,data_type,col):
    data = np.fromfile(path,dtype=data_type)
    
    ## Reading bin file generated from matlab (column wise)
    data = data.reshape(col,int(data.shape[0]/col))
    data = data.T

    ## Reading bin file generated from python (row wise)
    # data = data.reshape(int(data.shape[0]/col),col)

    return data

def quat_to_euler(quat):
    r = R.from_quat(quat_only)
    # Rotation seq 'ZYX' = 'xyz'
    euler = r.as_euler('ZYX',degrees=True)

    return euler

def calculate_angVel(euler,time):
    euler_x = euler[:,2]
    euler_y = euler[:,1]
    euler_z = euler[:,0]

    diff_eul_x = np.diff(euler_x)
    diff_eul_y = np.diff(euler_y)
    diff_eul_z = np.diff(euler_z)
    diff_time = np.diff(time)
    
    angVel_x = diff_eul_x/diff_time
    angVel_y = diff_eul_y/diff_time
    angVel_z = diff_eul_z/diff_time
    angVel_x = np.deg2rad(angVel_x)
    angVel_y = np.deg2rad(angVel_y)
    angVel_z = np.deg2rad(angVel_z)

    return angVel_x,angVel_y,angVel_z

def transform_body_to_world(x,y):
    T = [ [1 , s(x)*t(y) ,  c(x)*t(y)] , [0 , c(x) , -s(x)] , [0 , s(x)/c(y) , c(x)/c(y)] ]
    T = np.array(T)
    return T

def aa_to_euler(cmgd):
    angle = np.linalg.norm(cmgd,axis=1)
    axis_x = cmgd[:,0]/angle
    axis_y = cmgd[:,1]/angle
    axis_z = cmgd[:,2]/angle
    axis_x = axis_x.reshape(-1,1)
    axis_y = axis_y.reshape(-1,1)
    axis_z = axis_z.reshape(-1,1)
    axis = np.concatenate((axis_x,axis_y,axis_z),axis=1)

    cmgd_angVel = []
    for i in range(len(axis_x)):
        temp = R.from_rotvec(angle[i]*axis[i,:])
        temp_angle = temp.as_euler('ZYX')
        cmgd_angVel.append(temp_angle)
    
    cmgd_angVel = np.array(cmgd_angVel)

    return cmgd_angVel

def compute_error(measured_data,groundtruth_data,N):
    error = np.absolute(measured_data-groundtruth_data)
    mean_sumErr = np.sum(error)/N
    max_gt = max(groundtruth_data)
    err = (mean_sumErr/max_gt)*100

    return err



# if __name__ == "__main__":
 
quat = readBin('../dynamic_data_06112020/dynamic_groundtruth.bin',np.float64,col=5)
cmgd = readBin('../dynamic_data_06112020/cmgd_all_mg_11112020.bin',np.float64,col=3)
imu = readBin('../dynamic_data_06112020/imu.bin',np.float64,col=7)


# Assign imu ang_vel to g (deg)
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

# Convert quat_only to euler
euler = quat_to_euler(quat_only)

# Calculate angular velocity relative to world frame
angVel_x,angVel_y,angVel_z = calculate_angVel(euler,quat_time)
angVel_world= np.vstack((angVel_x,angVel_y,angVel_z)).T

# Computer angular velocity relative to body frame
x = euler[:,2]
y = euler[:,1]
z = euler[:,0]
angVel_body = []

for i in range(x.shape[0]-1):
    T = transform_body_to_world(x[i+1],y[i+1])
    # T = [ [1 , s(x[i])*t(y[i]) ,  c(x[i])*t(y[i])] , [0 , c(x[i]) , -s(x[i])] , [0 , s(x[i])/c(y[i]) , c(x[i])/c(y[i])] ]
    temp = np.dot(np.linalg.inv(T),angVel_world[i])
    angVel_body.append(temp)

angVel_body = np.array(angVel_body)
angVel_body = angVel_body*180/np.pi

# Compute cmgd angular velocity from axis angle to euler
cmgd_angVel = aa_to_euler(cmgd)
cmgd_angVel = cmgd_angVel*180*100/np.pi

# Roll the column
cmgd_angVel[2:,:] = cmgd_angVel[0:-2,:]

# Divide in to 4 pieces

temp_x = gaussian_filter1d(angVel_body[:-1,0],1.5)
temp_y = gaussian_filter1d(angVel_body[:-1,1],1.5)
temp_z = gaussian_filter1d(angVel_body[:-1,2],1.5)

n  = temp_x.shape[0]//4
n_range = np.arange(0,temp_x.shape[0],n)
# n_range[-1] = n_range[-1]+2

for i in range(n_range.shape[0]-1):
    prev = n_range[i]
    curr = n_range[i+1]

    err_x = compute_error(cmgd_angVel[prev:curr,2],temp_x[prev:curr],n)
    err_y = compute_error(cmgd_angVel[prev:curr,1],temp_y[prev:curr],n)
    err_z = compute_error(cmgd_angVel[prev:curr,0],temp_z[prev:curr],n)



    print(err_x,err_y,err_z)
    # print(max(temp_x),max(temp_y),max(temp_z))
# print(err_y)
print(cmgd_time.shape,cmgd_time[0],cmgd_time[-4])



#Plotting
plt.plot(cmgd_time[:-2],cmgd_angVel[:,2],'--',color='r')
plt.plot(cmgd_time[:-2],cmgd_angVel[:,1],'--',color='b')
plt.plot(cmgd_time[:-2],cmgd_angVel[:,0],'--',color='orange')

plt.plot(cmgd_time[:-2],temp_x,color='r')
plt.plot(cmgd_time[:-2],temp_y,color='b')
plt.plot(cmgd_time[:-2],temp_z,color='orange')

# plt.plot(cmgd_time[:-2],cmgd[:,0]*180*100/np.pi)
# plt.plot(cmgd_time[:-2],cmgd[:,1]*180*100/np.pi)
# plt.plot(cmgd_time[:-2],cmgd[:,2]*180*100/np.pi)


# plt.ylim(-200,200)
# plt.xlim(14,20)
# plt.xlabel('Time [s]')
# plt.ylabel('Angular Velocities [deg/s]')
# plt.legend(['X_measured','Y_measured','Z_measured','X_groundtruth','Y_groundtruth','Z_groundtruth'])






# plt.plot(imu_time,gx,color='r')
# plt.plot(imu_time,gy,color='b')
# plt.plot(imu_time,gz,color='orange')

plt.show()
