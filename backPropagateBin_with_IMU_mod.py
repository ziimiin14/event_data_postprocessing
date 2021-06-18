import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import find
from numpy.core.fromnumeric import clip
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import cv2

# Initialize image dimension
width =320
height=240
    
# Initialize K matrix
K= np.matrix([[322.27115293, 0., 156.60442714], [0., 323.8544765, 116.09022504],[ 0., 0., 1.]])
K_arr = np.array(K)
K_I_arr = np.array(K.I)

# Load events data from bin file
event = np.fromfile('../event_data_06152021/bin_file/kratos_eventOnly_06152021_1.bin',dtype=np.uint16)
event = event.reshape(-1,3)

# Load time data (event) from bin file
time_sec = np.fromfile('../event_data_06152021/bin_file/kratos_eventTime_06152021_1.bin',dtype=np.float64)
time_sec = time_sec.reshape(-1,1) 
time_interval = 1/100

# Load opti track data
imu = np.fromfile('../event_data_06152021/bin_file/kratos_IMU_06152021_1.bin',dtype=np.float64)
imu = imu.reshape(-1,4)
imu_time = imu[:,0]
imu_time = imu_time.reshape(-1,1)
imu_ang = imu[:,1:]

# Find the start time and end time for opti track data to sync with event data time
if time_sec[0,0] < imu_time[0,0] and imu_time[-1,0] < time_sec[-1,0]:
    findFirst = np.where(time_sec[:,0]<imu_time[0,0])[0][-1] + 1
    findLast = np.where(time_sec[:,0]>imu_time[-1,0])[0][0]

    # Slice the time_sec and event from findFirst to findLast
    time_sec = time_sec[findFirst:findLast,:]
    event = event[findFirst:findLast,:]

    time_sec = time_sec - imu_time[0,0]
    imu_time = imu_time - imu_time[0,0]

    time_sec = time_sec.astype(np.float32)
    imu_time = imu_time.astype(np.float32)

    time_init = imu_time[0,:]
    time_end = imu_time[-1,:]

elif imu_time[0,0] < time_sec[0,0] and time_sec[-1,0] < imu_time[-1,0]:
    findFirst = np.where(imu_time[:,0]<time_sec[0,0])[0][-1] 
    findLast = np.where(imu_time[:,0]>time_sec[-1,0])[0][0]+1

    time_sec = time_sec - imu_time[findFirst,0]
    imu_time = imu_time - imu_time[findFirst,0]

    time_sec = time_sec.astype(np.float32)
    imu_time = imu_time.astype(np.float32)

    time_init = imu_time[findFirst,:]
    time_end = imu_time[findLast,:]


elif time_sec[0,0]< imu_time[0,0] and time_sec[-1,0] < imu_time[-1,0]:
    findFirst = np.where(time_sec[:,0]<imu_time[0,0])[0][-1] + 1
    findLast = np.where(imu_time[:,0]>time_sec[-1,0])[0][0]+1



    # Slice the time_sec and event 
    time_sec  = time_sec[findFirst:]
    event = event[findFirst:]

    time_sec = time_sec - imu_time[0,0]
    imu_time = imu_time - imu_time[0,0]

    time_sec = time_sec.astype(np.float32)
    imu_time = imu_time.astype(np.float32)

    time_init = imu_time[0,:]
    time_end = imu_time[findLast,:]

else:
    findFirst = np.where(imu_time[:,0]<time_sec[0,0])[0][-1]
    findLast = np.where(time_sec[:,0]>imu_time[-1,0])[0][0]

    # Slice the time_sec and event
    time_sec = time_sec[:findLast,:]
    event = event[:findLast,:]

    time_sec = time_sec - imu_time[findFirst,0]
    imu_time = imu_time - imu_time[findFirst,0]

    time_sec = time_sec.astype(np.float32)
    imu_time = imu_time.astype(np.float32)

    time_init = imu_time[findFirst,:]
    time_end = imu_time[-1,:]


# Construct time histogram with respect to the opti_time (findFirst and findLast)
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

    curr_angV = imu_ang[temp_init:temp_last+1].mean(axis=0)



    # Compute a 3 by N dimension array with respect to the specific events
    x_arr = specific_event[:,0]
    y_arr = specific_event[:,1]
    z_arr = np.ones(specific_event.shape[0],dtype=np.uint16)
    specific_pos_pixel = np.reshape((x_arr,y_arr,z_arr),(3,-1)) # Points in pixel frame

    dt_temp = specific_time- specific_time[0,0]
    dt_temp = dt_temp.reshape(-1)


    # Convert points in from image frame to camera frame
    specific_pos_camera = K_I_arr@specific_pos_pixel


    w3_y = curr_angV[2]*specific_pos_camera[1,:]
    w3_x = curr_angV[2]*specific_pos_camera[0,:]
    w1_y = curr_angV[0]*specific_pos_camera[1,:]
    w2_x = curr_angV[1]*specific_pos_camera[0,:]

    # Back propagate points in camera frame
    x_new = specific_pos_camera[0,:] + ((curr_angV[1]-w3_y)*dt_temp)
    y_new = specific_pos_camera[1,:] + ((w3_x-curr_angV[0])*dt_temp)
    z_new = specific_pos_camera[2,:] + ((w1_y-w2_x)*dt_temp)

    BxC = np.vstack([x_new,y_new,z_new])
    final_pos_pixel = K_arr@BxC
    final_pos_pixel = final_pos_pixel.T
    final_pos_pixel[:,0], final_pos_pixel[:,1],final_pos_pixel[:,2]=final_pos_pixel[:,0]/final_pos_pixel[:,2],final_pos_pixel[:,1]/final_pos_pixel[:,2],final_pos_pixel[:,2]/final_pos_pixel[:,2]
    final_pos_pixel = final_pos_pixel.astype(int)
    final_pos_pixel[:,2] = event[prev:current,2]

    
    # Black white  histogram 
    black_img_1  = np.zeros((height,width),dtype =np.uint8)
    black_img  = np.zeros((height,width),dtype =np.uint8)
    current_event_1 = specific_event
    current_event = final_pos_pixel
    boolean = (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0)
    current_event = current_event[boolean,:]



    black_img,yed,xed=np.histogram2d(current_event[:,1],current_event[:,0],bins=(yedges,xedges))
    black_img_1,yed,xed=np.histogram2d(current_event_1[:,1],current_event_1[:,0],bins=(yedges,xedges))

    # Normalize the image
    black_img = black_img/ black_img.max()
    black_img_1 = black_img_1/ black_img_1.max()



    ## Normalize the image from 0-1 by clipping
    # current_event_1 = specific_event
    # current_event = final_pos_pixel
    # boolean = (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0) & (current_event[:,2] == 0)
    # boolean1 = (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0) & (current_event[:,2] == 1)
    # curr_event  = current_event[boolean,:]
    # curr_event1  = current_event[boolean1,:]

    # curr_img,yed,xed=np.histogram2d(curr_event[:,1],curr_event[:,0],bins=(yedges,xedges))
    # curr_img1,yed,xed=np.histogram2d(curr_event1[:,1],curr_event1[:,0],bins=(yedges,xedges))
    # black_img = curr_img1- curr_img


    # black_img = np.clip(black_img,-3,3)
    # black_img = (black_img+3)/6.0


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


