import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import cv2

    
# Initialize K matrix
K= np.matrix([[330.1570953377, 0., 161.9624665569], [0., 329.536232838, 110.80414596744],[ 0., 0., 1.]])
K_arr = np.array(K)
K_I_arr = np.array(K.I)

# Load events data from bin file
event = np.fromfile('../event_data_05122021/bin_file/kratos_eventOnly_05122021_4.bin',dtype=np.uint16)
event = event.reshape(-1,3)

# Load time data (event) from bin file
time_sec = np.fromfile('../event_data_05122021/bin_file/kratos_eventTime_05122021_4.bin',dtype=np.float64)
time_sec = time_sec.reshape(-1,1) 
time_interval = 1/100

# Load opti track data
opti = np.fromfile('../event_data_05122021/bin_file/kratos_quat_05122021_4.bin',dtype=np.float64)
opti = opti.reshape(-1,5)
opti_time = opti[:,0]
opti_time = opti_time.reshape(-1,1)
opti_quat = opti[:,1:]


# Find the start time and end time for opti track data to sync with event data time
findFirst = np.where(opti_time[:,0]<time_sec[0,0])[0][-1] 
findLast = np.where(opti_time[:,0]>time_sec[-1,0])[0][0]+1

# Construct time histogram with respect to the opti_time (findFirst and findLast)
time_init = opti_time[findFirst,:]
time_end = opti_time[findLast,:]
time_range = np.arange(time_init,time_end,time_interval)
time_hist,time_binEdge = np.histogram(time_sec,bins=time_range)
time_hist_cum = np.cumsum(time_hist)

# Compute rotation relative to the first frame
# opti_quat = opti_quat[findFirst:findLast,:]
quat = np.roll(opti_quat,1,axis=1)
first_quat = Quaternion(quat[0,:])
first_quat_conj = first_quat.conjugate
quat_rel = []
quat_world = []

# Rotation from previous point to next point
# for i in range(quat.shape[0]-1):
#     temp = Quaternion(quat[i+1,:])*Quaternion(quat[i,:]).conjugate 
#     quat_rel.append([temp[1],temp[2],temp[3],temp[0]])

# Rotation from first frame to ith frame
# for i in range(quat.shape[0]):
#     temp = Quaternion(quat[i,:])*first_quat_conj
#     quat_rel.append([temp[1],temp[2],temp[3],temp[0]])

for i in range(quat.shape[0]):
    quat_world.append(Quaternion(quat[i,:]))


# quat_rel = np.array(quat_rel)
# r = R.from_quat(quat_rel)
# dcm = r.as_dcm()
# dcm_T = np.einsum('abc->acb',dcm)
# euler = r.as_euler('ZYX',degrees=True)


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


while(True):
    ## For each frame:
    print(i)
    # Declare a specific_event and specific_time for specific frame requested
    specific_event = event[prev:current,:]
    specific_time = time_sec[prev:current,:]

    # Find the initial opti time index(temp_init) that is lesser than first specific time and
    # last opti time index(temp_last) that is more that last specific time
    temp_init = np.where(opti_time[:,0]<=specific_time[0,0])[0][-1]
    temp_last = np.where(opti_time[:,0]>=specific_time[-1,0])[0][0]
    print(temp_init,temp_last)

    # Define the rotation ratio 
    diff_time_numerator = specific_time-opti_time[temp_init,0]
    diff_time_denominator = opti_time[temp_last,0]-opti_time[temp_init,0]
    ratio = diff_time_numerator/diff_time_denominator

    # Declare the quaternions with respect to the initial and last opti time index
    q1 = quat_world[temp_init]
    q2 = quat_world[temp_last]

    # Rotate (q1.conjugate) from world frame to initial frame first and then apply rotation(q2) to the point
    q3 = q2*q1.conjugate

    # Convert quaternion to euler with the sequence of ZYX
    q = np.array([q3[1],q3[2],q3[3],q3[0]])
    r = R.from_quat(q)
    euler = r.as_euler('ZYX',degrees=True)
    euler = euler.reshape(1,-1)
    print(euler)
    print(ratio)

    # Compute euler with rotation ratio
    # to obtain euler_arr (rotation with respect to each specific event time frame))
    euler_arr = np.dot(ratio,euler)
    # print(euler_arr)
    euler_arr[:,1] = euler_arr[:,0]
    # print(euler_arr)
    euler_arr[:,0] = 0
    euler_arr[:,2] = 0
    euler_arr = euler_arr-euler_arr[0,:]

    print(euler_arr)

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

    width =320
    height=240

    black_img_1  = np.zeros((height,width,3),dtype =np.uint8)
    black_img  = np.zeros((height,width,3),dtype =np.uint8)
    current_event_1 = event[prev:current,:]
    current_event = final_pos_pixel
    ones_1 = np.where(current_event_1[:,2]==1)
    zeros_1 = np.where(current_event_1[:,2]==0)
    zeros=np.where((current_event[:,2]==0) & (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
    ones=np.where((current_event[:,2]==1) & (current_event[:,0]<width) & (current_event[:,1]<height) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
    black_img_1[current_event_1[ones_1][:,1],current_event_1[ones_1][:,0],:] = [255,0,0]
    black_img_1[current_event_1[zeros_1][:,1],current_event_1[zeros_1][:,0],:] = [0,0,255]
    black_img[current_event[ones][:,1],current_event[ones][:,0],:] = [255,0,0]
    black_img[current_event[zeros][:,1],current_event[zeros][:,0],:] = [0,0,255]


    # Show the appended black_img
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 960,720)
    cv2.imshow('image',black_img)
    cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image1', 960,720)
    cv2.imshow('image1',black_img_1)
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', 960,720)
    #cv2.imshow('image',img)
    # plt.imshow(img)
    # plt.show()

    k = cv2.waitKey(5000)

    if k == ord('c'):
        i += 1
        prev,current = time_hist_cum[i-1], time_hist_cum[i]
        continue
    
    elif k == ord('x'):
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



# test = np.zeros((height,width))
# current_event = final_pos_pixel
# chosen=np.where((current_event[:,0]<240) & (current_event[:,1]<180) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
# for x in range(len(current_event[chosen])):
#     test[current_event[chosen][x,1],current_event[chosen][x,0]] +=1

# res = test/test.max()


# while(True):
#     #res = res.astype('uint8')
#     cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('image', 960,720)
#     cv2.imshow('image',res)
#     k = cv2.waitKey(10000)

#     if k ==ord('q'):
#         break




# r = R.from_quat(quat)
# euler = r.as_euler('xyz',degrees=True)
# yaw = euler[:,2]
# neg = np.where(yaw<0)
# yaw[neg] += 360
# r1=R.from_euler('xyz',euler,degrees=True)
# dcm = r1.as_dcm()
# np.round()
# R.from_euler()