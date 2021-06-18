import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import cv2
import os

# Initialize image dimension
width =320
height=240
    
# Initialize K matrix
# K= np.matrix([[326.1387442, 0., 158.28624894], [0., 326.84650981, 112.96798711],[ 0., 0., 1.]]) # very low bias
K= np.matrix([[322.27115293, 0., 156.60442714], [0., 323.8544765, 116.09022504],[ 0., 0., 1.]]) # default
K_arr = np.array(K)
K_I_arr = np.array(K.I)

# Load events data from bin file
event = np.fromfile('../event_data_06022021/bin_file/kratos_eventOnly_06022021.bin',dtype=np.uint16)
event = event.reshape(-1,3)

# Load time data (event) from bin file
time_sec = np.fromfile('../event_data_06022021/bin_file/kratos_eventTime_06022021.bin',dtype=np.float64)
time_sec = time_sec.reshape(-1,1) 
time_interval = 1/100

# Load opti track data
imu = np.fromfile('../event_data_06022021/bin_file/kratos_IMU_06022021.bin',dtype=np.float64)
imu = imu.reshape(-1,4)
imu_time = imu[:,0]
imu_time = imu_time.reshape(-1,1)
imu_ang = imu[:,1:]
imu_ang = imu_ang*180/np.pi

#Calcaulate angle rotated
diff_imu_time = np.diff(imu_time,axis=0)
diff_angle = diff_imu_time*imu_ang[0:-1,:]

angle = np.zeros(imu_ang.shape)

for i in range(1,angle.shape[0]):
    angle[i] = diff_angle[i-1] + angle[i-1]

angle[:,1] = -angle[:,1]
angle[:,2] = -angle[:,2]

# Find the start time and end time for opti track data to sync with event data time
findFirst = np.where(time_sec[:,0]<imu_time[0,0])[0][-1] + 1
findLast = np.where(time_sec[:,0]>imu_time[-1,0])[0][0]


# Slice the time_sec and event from findFirst to findLast
time_sec = time_sec[findFirst:findLast,:]
event = event[findFirst:findLast,:]

# Construct time histogram with respect to the opti_time (findFirst and findLast)
time_init = imu_time[0,:]
time_end = imu_time[-1,:]
time_range = np.arange(time_init,time_end+time_interval,time_interval)
time_hist,time_binEdge = np.histogram(time_sec,bins=time_range)
time_hist_cum = np.cumsum(time_hist)

# Maximum frame and initialize first time interval event frame
max_frame= time_hist.shape[0]
i = 0
prev = 0
current = time_hist_cum[i]

txt_file = open('../event_data_06022021/txt_file/kratos_event_06022021.txt','w')

txt_file.write("{} {}\n".format(width,height))
while i < max_frame:
    ## For each frame:
    # Declare a specific_event and specific_time for specific frame requested
    specific_event = event[prev:current,:]
    specific_time = time_sec[prev:current,:]

    # Find the initial opti time index(temp_init) that is lesser than first specific time and
    # last opti time index(temp_last) that is more that last specific time
    temp_init = np.where(imu_time[:,0]<=specific_time[0,0])[0][-1]
    temp_last = np.where(imu_time[:,0]>=specific_time[-1,0])[0][0]

    # Define the rotation ratio 
    diff_time_numerator = specific_time-imu_time[temp_init,0]
    diff_time_denominator = imu_time[temp_last,0]-imu_time[temp_init,0]
    ratio = diff_time_numerator/diff_time_denominator

    # Declare the quaternions with respect to the initial and last opti time index
    a1 = angle[temp_init]
    a2 = angle[temp_last]

    # Rotate (q2) to the desired point first and then apply rotation (q1.conjugate) to get rotation of desired position relative to point 1
    euler = a2-a1

    # q3 = q2*q1.conjugate

    # Convert quaternion to euler with the sequence of ZYX
    euler[0],euler[2] = euler[2],euler[0]
    euler = euler.reshape(1,-1)

    # Compute euler with rotation ratio
    # to obtain euler_arr (rotation with respect to each specific event time frame))
    euler_arr = np.dot(ratio,euler)
    euler_arr = euler_arr-euler_arr[0,:]

    # Convert the euler arr to dcm
    r1 = R.from_euler('ZYX',euler_arr,degrees=True)
    dcm = r1.as_dcm()
    dcm_T = np.einsum('iab->iba',dcm)

    # Compute a 3 by N dimension array with respect to the specific events
    x_arr = specific_event[:,0]
    y_arr = specific_event[:,1]
    z_arr = np.ones(specific_event.shape[0],dtype=np.uint16)
    specific_pos_pixel = np.reshape((x_arr,y_arr,z_arr),(3,-1)) # Points in pixel frame

    # Compute points in camera frame
    specific_pos_camera = K_I_arr@specific_pos_pixel
    # print(dcm_T.shape,specific_pos_camera.shape)

    # Back propagate points in camera frame
    BxC = np.einsum('iab,bi->ai',dcm_T,specific_pos_camera)

    final_pos_pixel = K_arr@BxC
    final_pos_pixel = final_pos_pixel.T
    final_pos_pixel[:,0], final_pos_pixel[:,1],final_pos_pixel[:,2]=final_pos_pixel[:,0]/final_pos_pixel[:,2],final_pos_pixel[:,1]/final_pos_pixel[:,2],final_pos_pixel[:,2]/final_pos_pixel[:,2]
    final_pos_pixel = np.round(final_pos_pixel)
    final_pos_pixel = final_pos_pixel.astype(int)
    final_pos_pixel[:,2] = event[prev:current,2]
    # con,img = gauss3sigma(final_pos_pixel[:,:2])


    current_event = final_pos_pixel
    boolean = (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0)
    current_event  = current_event[boolean,:]
    current_time  = specific_time[boolean,:]
    
    final_txt_data = np.hstack([current_time,current_event])

    for d1,d2,d3,d4 in final_txt_data:
        d_all = "{} {} {} {}\n".format(d1,int(d2),int(d3),int(d4))
        txt_file.write(d_all)

    i += 1
    if i == max_frame:
        break

    prev,current = time_hist_cum[i-1], time_hist_cum[i]
    while prev == current:
        i += 1
        prev,current = time_hist_cum[i-1], time_hist_cum[i]

    print(i)

print(max_frame-1)

txt_file.close()

    
    
