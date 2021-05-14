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
event = np.fromfile('../event_data_05122021/bin_file/kratos_eventOnly_05122021_4.bin',dtype=np.uint8)
event = event.reshape(-1,3)

# Load time data (event) from bin file
time_sec = np.fromfile('../event_data_05122021/bin_file/kratos_eventTime_05122021_4.bin',dtype=np.float64)
time_sec = time_sec.reshape(-1,1) 
time_interval = 1/150

# Load opti track data
opti = np.fromfile('../event_data_05122021/bin_file/kratos_quat_05122021_4.bin',dtype=np.float64)
opti = opti.reshape(-1,5)
opti_time = opti[:,0]
opti_time = opti_time.reshape(-1,1)
opti_quat = opti[:,1:]


# Find the start time and end time for opti track data to sync with event data time
findFirst = np.where(opti_time[:,0]<time_sec[0,0])[0][-1] 
findLast = np.where(opti_time[:,0]>time_sec[-1,0])[0][0]

# Construct time histogram with respect to the opti_time (findFirst and findLast)
time_init = opti_time[findFirst,:]
time_end = opti_time[findLast,:]
time_range = np.arange(time_init,time_end,time_interval)
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



# Compute rotation relative to the first frame
quat = np.roll(opti_quat,1,axis=1)
first_quat = Quaternion(quat[0,:])
first_quat_conj = first_quat.conjugate
quat_rel = []

for i in range(quat.shape[0]-1):
    temp = Quaternion(quat[i,:]).conjugate * Quaternion(quat[i+1,:])
    quat_rel.append([temp[1],temp[2],temp[3],temp[0]])



quat_n1 = np.array(quat_n1)
r = R.from_quat(quat_n1)
euler = r.as_euler('ZYX',degrees=True)


r = R.from_quat(opti_quat)


while(True):
    temp = time_binEdge[i]
    temp1 = time_binEdge[i+1]
    prev_1,current_1 = np.where((opti_time<=temp))[0][-1],np.where((opti_time>=temp1))[0][0]
    a = opti_time[prev_1]
    b = opti_time[current_1]
    c = [a.item(),b.item()]
    e = rotvec[prev_1,1]-rotvec[prev_1,1]
    f = abs(rotvec[current_1,1]-rotvec[prev_1,1])
    d = [e,-f if f<1 else (f-2*np.pi)]
    x = time_sec[prev:current]
    rotvec_interp = np.interp(x,c,d)
    rotvec_interp -= rotvec_interp[0,0]
    diff_rot=np.zeros((rotvec_interp.shape[0],3))
    diff_rot[:,1] = rotvec_interp[:,0]
    #print(diff_rot)

    specific_event = event[prev:current,:]
    print(specific_event.shape)
    specific_time = time_sec[prev:current,:]
    diff_spec_time = specific_time-specific_time[0]
    #diff_deg = diff_spec_time*1800

    x_arr = specific_event[:,0]
    y_arr = specific_event[:,1]
    z_arr = np.ones(specific_event.shape[0],dtype=np.uint8)

    specific_pos_pixel = np.reshape((x_arr,y_arr,z_arr),(3,specific_event.shape[0]))
    specific_pos_world = K_I_arr.dot(specific_pos_pixel)

    #diff_rot = diff_deg*np.array([0,-1,0])
    r = R.from_euler('xyz',diff_rot)
    dcm = r.as_dcm()

    specific_pos_world = specific_pos_world.reshape(3,specific_event.shape[0],1)
    BxC = np.einsum('iab,bid->iad',dcm,specific_pos_world)

    final_pos_pixel = np.einsum('ab,ibd->ia',K_arr,BxC)
    final_pos_pixel[:,0], final_pos_pixel[:,1],final_pos_pixel[:,2]=final_pos_pixel[:,0]/final_pos_pixel[:,2],final_pos_pixel[:,1]/final_pos_pixel[:,2],final_pos_pixel[:,2]/final_pos_pixel[:,2]
    final_pos_pixel = np.round(final_pos_pixel)
    final_pos_pixel = final_pos_pixel.astype(int)
    final_pos_pixel[:,2] = event[prev:current,2]
    # con,img = gauss3sigma(final_pos_pixel[:,:2])

    width =240
    height=180

    black_img_1  = np.zeros((height,width,3),dtype =np.uint8)
    black_img  = np.zeros((height,width,3),dtype =np.uint8)
    current_event_1 = event[prev:current,:]
    current_event = final_pos_pixel
    ones_1 = np.where(current_event_1[:,2]==1)
    zeros_1 = np.where(current_event_1[:,2]==0)
    zeros=np.where((current_event[:,2]==0) & (current_event[:,0]<240) & (current_event[:,1]<180) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
    ones=np.where((current_event[:,2]==1) & (current_event[:,0]<240) & (current_event[:,1]<180) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
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

    k = cv2.waitKey(10000)

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