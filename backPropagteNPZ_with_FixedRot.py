import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
import cv2
from mpl_toolkits import mplot3d


def data_load(path):
    #obj = np.load('../test/6300ERPM/static_rotation_6300ERPM_polEvents.npz')
    #obj = np.load('../test/6300ERPM/static_rotation_6300ERPM_time.npz')
    obj = np.load(path)
    namelist = obj.zip.namelist()
    obj.zip.extract(namelist[0])
    data = np.load(namelist[0])
    return data


K= np.matrix([[329.81456747559736, 0., 119.46793030265114], [0., 329.80049922231348, 80.956489477048663],[ 0., 0., 1.]])
#K= np.matrix([[291.90324780145994, 0., 128.71047242177775], [0., 291.61735408383589, 86.059164015558267],[ 0., 0., 1.]])

# Starting from second event
event = data_load('../test/Rotation/6300ERPM/polEvents_1.npz')
#event = data_load('data/polEvents.npz')
#check=np.where(event[:,3] == 1)
#event = event[check[0],:3]
event = event[:,:3]


# Starting from second time
time_micro = data_load('../test/Rotation/6300ERPM/time_1.npz')
#time_micro = data_load('data/time.npz')
time_sec = time_micro/1e6
#time_sec = time_sec[check[0]]

time_interval = 10/1000

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
    #print(specific_event.shape)
    specific_time = time_sec[prev:current,:]
    diff_spec_time = specific_time-specific_time[0]
    diff_deg = diff_spec_time*1800
    x_arr = specific_event[:,0]
    y_arr = specific_event[:,1]
    z_arr = np.ones(specific_event.shape[0],dtype=np.uint8)
    specific_pos_pixel = np.reshape((x_arr,y_arr,z_arr),(3,specific_event.shape[0]))
    specific_pos_world = np.matmul(K.I,specific_pos_pixel)
    diff_rot = -diff_deg*np.array([0,1,0])
    r = R.from_euler('xyz',diff_rot,degrees=True)
    dcm = r.as_dcm()
    specific_pos_world_arr = specific_pos_world.copy()
    specific_pos_world_arr = np.array(specific_pos_world_arr)
    specific_pos_world_arr = specific_pos_world_arr.T
    specific_pos_world_arr = np.reshape(specific_pos_world_arr,(specific_event.shape[0],3,1))
    BxC = np.matmul(dcm,specific_pos_world_arr)
    final_pos_pixel = np.matmul(K,BxC)
    final_pos_pixel = np.array(final_pos_pixel)
    final_pos_pixel[:,0], final_pos_pixel[:,1],final_pos_pixel[:,2]=final_pos_pixel[:,0]/final_pos_pixel[:,2],final_pos_pixel[:,1]/final_pos_pixel[:,2],final_pos_pixel[:,2]/final_pos_pixel[:,2]
    #final_pos_pixel[:,1]=final_pos_pixel[:,1]/final_pos_pixel[:,2]
    #final_pos_pixel[:,2]=final_pos_pixel[:,2]/final_pos_pixel[:,2]
    final_pos_pixel = np.round(final_pos_pixel)
    final_pos_pixel = final_pos_pixel.astype(int)
    final_pos_pixel[:,2] = event[prev:current,2]

    width =240
    height=180

    black_img  = np.zeros((height,width,3),dtype =np.uint8)
    current_event = final_pos_pixel
    zeros=np.where((current_event[:,2]==0) & (current_event[:,0]<240) & (current_event[:,1]<180) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
    ones=np.where((current_event[:,2]==1) & (current_event[:,0]<240) & (current_event[:,1]<180) & (current_event[:,1]>=0) & (current_event[:,0]>=0))
    black_img[current_event[ones][:,1],current_event[ones][:,0],:] = [255,0,0]
    black_img[current_event[zeros][:,1],current_event[zeros][:,0],:] = [0,0,255]
    

    # Show the appended black_img
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 960,720)
    cv2.imshow('image',black_img)
    #print(black_img.max())
    #print(prev,current)
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(specific_time,final_pos_pixel[:,0],final_pos_pixel[:,1],cmap='Greens')
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
        cv2.imwrite('frame'+str(i+1)+'.png', black_img)
        data = np.concatenate([time_micro[prev:current,:],current_event],axis=1)
        np.savetxt('frame'+str(i+1)+'.csv',data,delimiter=',')
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