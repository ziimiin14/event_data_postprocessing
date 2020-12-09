import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
import cv2

#K= np.matrix([[329.81456747559736, 0., 119.46793030265114], [0., 329.80049922231348, 80.956489477048663],[ 0., 0., 1.]])
K= np.matrix([[184.4818474, 0., 121.9591], [0., 184.56342, 89.686064731],[ 0., 0., 1.]])
#K= np.matrix([[334.9343, 0., 118.23137], [0., 334.668432, 88.399],[ 0., 0., 1.]])
K_arr = np.array(K)
K_I_arr = np.array(K.I)
#K= np.matrix([[291.90324780145994, 0., 128.71047242177775], [0., 291.61735408383589, 86.059164015558267],[ 0., 0., 1.]])

#event = np.fromfile('6300ERPM_2000_Events.bin',dtype=np.uint8)
event = np.fromfile('../event_data_14102020/8820ERPM_Events_14102020.bin',dtype=np.uint8)
event = event.reshape(int(event.shape[0]/3),3)

# Starting from second time
#time_micro = data_load('../test/Rotation/6300ERPM/time_1.npz')
#time_sec = np.fromfile('6300ERPM_2000_Time.bin',dtype=np.float64)
time_nano = np.fromfile('../event_data_14102020/8820ERPM_Time_14102020.bin',dtype=np.int64)
time_nano = time_nano.reshape(int(time_nano.shape[0]),1) 
time_sec = np.fromfile('../event_data_14102020/8820ERPM_Time_14102020.bin',dtype=np.int64)
time_sec = time_sec.reshape(int(time_sec.shape[0]),1) 
time_sec = time_sec/1e9
time_interval = 1/100

time_init = time_sec[0,:]
time_end = time_sec[-1,:]
time_range = np.arange(time_init,time_end+time_interval,time_interval)
time_hist,time_binEdge = np.histogram(time_sec,bins=time_range)
time_hist_cum = np.cumsum(time_hist)

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


# Load opti track data
# data = np.loadtxt('6300ERPM_2000_Opti_Tf.txt',delimiter=',')
# time = data[:,0]
# time = time/1e9
# quat = data[:,1:5]

while(True):

    specific_event = event[prev:current,:]
    specific_time = time_sec[prev:current,:]
    diff_spec_time = specific_time-specific_time[0]
    diff_deg = diff_spec_time*0

    x_arr = specific_event[:,0]
    y_arr = specific_event[:,1]
    z_arr = np.ones(specific_event.shape[0],dtype=np.uint8)

    specific_pos_pixel = np.reshape((x_arr,y_arr,z_arr),(3,specific_event.shape[0]))
    specific_pos_world = np.matmul(K.I,specific_pos_pixel)

    diff_rot = diff_deg*np.array([0,-1,0])
    r = R.from_euler('xyz',diff_rot,degrees=True)
    dcm = r.as_dcm()

    specific_pos_world = np.array(specific_pos_world)
    specific_pos_world = specific_pos_world.T
    specific_pos_world = np.reshape(specific_pos_world,(specific_event.shape[0],3,1))
    BxC = np.matmul(dcm,specific_pos_world)
    
    final_pos_pixel = np.matmul(K,BxC)
    final_pos_pixel = np.array(final_pos_pixel)
    final_pos_pixel[:,0], final_pos_pixel[:,1],final_pos_pixel[:,2]=final_pos_pixel[:,0]/final_pos_pixel[:,2],final_pos_pixel[:,1]/final_pos_pixel[:,2],final_pos_pixel[:,2]/final_pos_pixel[:,2]
    final_pos_pixel = np.round(final_pos_pixel)
    final_pos_pixel = final_pos_pixel.astype(int)
    final_pos_pixel[:,2] = event[prev:current,2]

    width =240
    height=180

    black_img  = np.zeros((height,width,3),dtype =np.uint8)
    current_event = final_pos_pixel
    current_event = current_event.astype('uint8')

    print(np.array_equal(specific_event,current_event))
    zeros=np.where((current_event[:,2]==0) & (current_event[:,0]<240) & (current_event[:,1]<180) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
    ones=np.where((current_event[:,2]==1) & (current_event[:,0]<240) & (current_event[:,1]<180) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
    black_img[current_event[ones][:,1],current_event[ones][:,0],:] = [255,0,0]
    black_img[current_event[zeros][:,1],current_event[zeros][:,0],:] = [0,0,255]

    print(time_nano[prev:current,:].dtype)
    print(current_event.dtype)
    # Show the appended black_img
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 960,720)
    cv2.imshow('image',black_img)
    #print(black_img.max())
    #print(prev,current)
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
        cv2.imwrite('frame'+str(i+1)+'.png', black_img)
        #data = np.concatenate([time_nano[prev:current,:],current_event],axis=1)
        time_nano[prev:current,:].tofile('8820ERPM_10ms_time_14012020.bin')
        current_event.tofile('8820ERPM_10ms_event_14102020.bin') 
        print(time_nano[prev:current,:].shape)
        print(current_event.shape)
        #np.savetxt('frame'+str(i+1)+'.csv',data,delimiter=',')
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