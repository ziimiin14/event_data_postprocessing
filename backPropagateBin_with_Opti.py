import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R
import cv2


def gauss3sigma(events):
# expects events in (n,2) format
# where events(1,2) -> (x,y) in pixels

    event_round = np.rint(events)
    event_round = event_round.astype(np.uint8)
    event_image = np.zeros((180,240))
    for i in range(events.shape[0]):
        hor = event_round[i][0]
        ver = event_round[i][1]
        if hor < 240 and ver>=0 and ver < 180 and ver>=0:
            X,Y = np.meshgrid(np.arange(max(hor-3,0), min(hor+4,239)),np.arange(max(ver-3,0), min(ver+4,179)))
            exponent = ((X-hor)**2 + (Y-ver)**2)/2
            amplitude = 1/ (np.sqrt(2*np.pi))
            val = amplitude* np.exp(-exponent)
            event_image[Y.min():Y.max()+1,X.min():X.max()+1] += val
            #event_image[Y.min().astype(int):Y.max().astype(int)+1,X.min().astype(int):X.max().astype(int)+1] += val

    contrast = -np.var(event_image)

    return contrast,event_image
    

K= np.matrix([[329.81456747559736, 0., 119.46793030265114], [0., 329.80049922231348, 80.956489477048663],[ 0., 0., 1.]])
K_arr = np.array(K)
K_I_arr = np.array(K.I)
#K= np.matrix([[291.90324780145994, 0., 128.71047242177775], [0., 291.61735408383589, 86.059164015558267],[ 0., 0., 1.]])

#event = np.fromfile('data_set1_sync/6300ERPM_Max_EventMod.bin',dtype=np.uint8)
event = np.fromfile('../6300ERPM_Events_02102020.bin',dtype=np.uint8)
event = event.reshape(int(event.shape[0]/3),3)

# Starting from second time
#time_micro = data_load('../test/Rotation/6300ERPM/time_1.npz')
#time_sec = np.fromfile('data_set1_sync/6300ERPM_Max_TimeMod.bin',dtype=np.int64)
time_sec = np.fromfile('../6300ERPM_Time_02102020.bin',dtype=np.int64)
time_sec = time_sec.reshape(int(time_sec.shape[0]),1) 
time_sec = time_sec/1e9
time_interval = 1/180

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


#Load opti track data
opti_quat = np.fromfile('data_set1_sync/6300ERPM_Max_Opti_QuatMod.bin',dtype=np.float64)
opti_quat = opti_quat.reshape(int(opti_quat.shape[0]/4),4)
opti_time = np.fromfile('data_set1_sync/6300ERPM_Max_Opti_TimeMod.bin',dtype=np.int64)
opti_time = opti_time.reshape(int(opti_time.shape[0]),1)
opti_time = opti_time/1e9
r = R.from_quat(opti_quat)
rotvec = r.as_rotvec()
temp = rotvec[:,1].copy()
rotvec[:,1] = rotvec[:,2]
rotvec[:,2] = temp
rotvec[:,0] = 0
rotvec[:,2] = 0
rotvec[:,1] += np.pi
#rotvec[:,1] -= rotvec[0,1]
#r1 = R.from_rotvec(rotvec)
#dcm = r1.as_dcm()



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