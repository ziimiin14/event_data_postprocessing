import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import clip
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import cv2

# Initialize image dimension
width =320
height=240
    
# Initialize K matrix
K= np.matrix([[330.1570953377, 0., 161.9624665569], [0., 329.536232838, 110.80414596744],[ 0., 0., 1.]])
K_arr = np.array(K)
K_I_arr = np.array(K.I)

# Load events data from bin file
event = np.fromfile('../event_data_06022021/bin_file/kratos_eventOnly_06022021_2.bin',dtype=np.uint16)
event = event.reshape(-1,3)

# Load time data (event) from bin file
time_sec = np.fromfile('../event_data_06022021/bin_file/kratos_eventTime_06022021_2.bin',dtype=np.float64)
time_sec = time_sec.reshape(-1,1) 
time_interval = 1/100

# Load opti track data
imu = np.fromfile('../event_data_06022021/bin_file/kratos_IMU_06022021_2.bin',dtype=np.float64)
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
# findFirst = np.where(imu_time[:,0]<time_sec[0,0])[0][-1] 
findFirst = np.where(time_sec[:,0]<imu_time[0,0])[0][-1] + 1
# findLast = np.where(imu_time[:,0]>time_sec[-1,0])[0][0]+1
findLast = np.where(time_sec[:,0]>imu_time[-1,0])[0][0]


# Slice the time_sec and event from findFirst to findLast
time_sec = time_sec[findFirst:findLast,:]
event = event[findFirst:findLast,:]

time_sec = time_sec - imu_time[0,0]
imu_time = imu_time - imu_time[0,0]

time_sec = time_sec.astype(np.float32)
imu_time = imu_time.astype(np.float32)

# Construct time histogram with respect to the opti_time (findFirst and findLast)
time_init = imu_time[0,:]
time_end = imu_time[-1,:]
time_range = np.arange(time_init,time_end+time_interval,time_interval)
time_hist,time_binEdge = np.histogram(time_sec,bins=time_range)
time_hist_cum = np.cumsum(time_hist)

# User input
max_frame= time_hist.shape[0]
user_input = input('Please choose the specific frame from 0 - '+str(max_frame-1)+': ')
user_input = int(user_input)
i= user_input
current = time_hist_cum[i]
current_time = time_range[i+1]
if i == 0:
    prev = 0
else:
    prev = time_hist_cum[i-1]

xedges = np.arange(0,width+1,1)
yedges = np.arange(0,height+1,1)


while(True):
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


    black_img_1  = np.zeros((height,width),dtype =np.uint8)
    black_img  = np.zeros((height,width),dtype =np.uint8)
    current_event_1 = event[prev:current,:].copy()
    current_event = final_pos_pixel
    boolean = (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0)
    current_event = current_event[boolean,:]

    ## Normalize the image from 0-1 by clipping
    # boolean = (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0) & (current_event[:,2] == 0)
    # boolean1 = (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0) & (current_event[:,2] == 1)
    # curr_event  = current_event[boolean,:]
    # curr_event1  = current_event[boolean1,:]

    black_img,yed,xed=np.histogram2d(current_event[:,1],current_event[:,0],bins=(yedges,xedges))
    black_img_1,yed,xed=np.histogram2d(current_event_1[:,1],current_event_1[:,0],bins=(yedges,xedges))

    # curr_img,yed,xed=np.histogram2d(curr_event[:,1],curr_event[:,0],bins=(yedges,xedges))
    # curr_img1,yed,xed=np.histogram2d(curr_event1[:,1],curr_event1[:,0],bins=(yedges,xedges))
    # curr_img[curr_img<2] = 0
    # curr_img1[curr_img1<2] = 0
    # black_img = curr_img1- curr_img

    
    # Normalize the image
    black_img = black_img/ black_img.max()
    black_img_1 = black_img_1/ black_img_1.max()

    # black_img = np.clip(black_img,-15,15)
    # black_img = (black_img+15)/float(30)

    

    # Map it back to 0-255
    black_img  = black_img * 255
    black_img_1  = black_img_1 * 255


    black_img = black_img.astype(np.uint8)
    black_img_1 = black_img_1.astype(np.uint8)
    
    # ones_1 = np.where(current_event_1[:,2]==1)
    # zeros_1 = np.where(current_event_1[:,2]==0)
    # zeros=np.where((current_event[:,2]==0) & (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
    # ones=np.where((current_event[:,2]==1) & (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
    # black_img_1[current_event_1[ones_1][:,1],current_event_1[ones_1][:,0],:] = [255,0,0]
    # black_img_1[current_event_1[zeros_1][:,1],current_event_1[zeros_1][:,0],:] = [0,0,255]
    # black_img[current_event[ones][:,1],current_event[ones][:,0],:] = [255,0,0]
    # black_img[current_event[zeros][:,1],current_event[zeros][:,0],:] = [0,0,255]


    # Show the appended black_img
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 640,480)
    cv2.imshow('image',black_img)
    cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image1', 640,480)
    cv2.imshow('image1',black_img_1)


    k = cv2.waitKey(5000)

    if k == ord('c'):
        i += 1
        prev,current = time_hist_cum[i-1], time_hist_cum[i]
        while prev==current:
            i += 1
            prev,current = time_hist_cum[i-1], time_hist_cum[i]

        continue
    
    elif k == ord('x'):
        i -= 1
        prev,current = time_hist_cum[i-1], time_hist_cum[i]
        while prev == current:
            i -= 1
            prev,current = time_hist_cum[i-1], time_hist_cum[i]
        continue

    elif k == ord('n'):
        user_input = input('Please choose a new specific frame from 1 - '+str(max_frame)+': ')
        user_input = int(user_input)
        i= user_input-1
        current = time_hist_cum[i]
        if i == 0:
            prev = 0
        else:
            prev = time_hist_cum[i-1]
        continue

    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('frame'+str(i)+'.png', black_img)
        data = np.concatenate([time_sec[prev:current,:],current_event],axis=1)
        np.savetxt('frame'+str(i)+'.csv',data,delimiter=',')
        continue

    elif k == ord('q'):
        break
    
    else:
        continue
    
    
    
cv2.destroyAllWindows() 


