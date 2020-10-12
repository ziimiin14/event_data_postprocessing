import numpy as np
import cv2



# Starting from second event
#event = data_load('../test/Rotation/6300ERPM/polEvents_1.npz')
event = np.fromfile('6300ERPM_2000_Events.bin',dtype=np.uint8)
event = event.reshape(int(event.shape[0]/3),3)

# Starting from second time
# #time_micro = data_load('../test/Rotation/6300ERPM/time_1.npz')
time_sec = np.fromfile('6300ERPM_2000_Time.bin',dtype=np.float64)
time_sec = time_sec.reshape(int(time_sec.shape[0]),1) 

# Set time interval
time_interval = 1/180

# Setup histogram for each time interval
time_init = time_sec[0,:]
time_end = time_sec[-1,:]
time_range = np.arange(time_init,time_end+time_interval,time_interval)
time_hist,time_binEdge = np.histogram(time_sec,bins=time_range)
time_hist_cum = np.cumsum(time_hist)
#print(time_hist,time_hist.min(),time_hist.shape)

#User input for specific frame required
max_frame= time_hist.shape[0]
user_input = input('Please choose the specific frame from 0 - '+str(max_frame-1)+': ')
user_input = int(user_input)

# Assign width,height value,
width =240
height=180

# i relative to user_input
i= user_input
j = user_input-1
current = time_hist_cum[i]
current_1 = time_hist_cum[j-1]
current_time = time_range[i+1]
if i == 0:
    prev = 0
else:
    prev = time_hist_cum[i-1]
    prev_1 = time_hist_cum[j-1]

while(True):
    black_img  = np.zeros((height,width,3),dtype =np.uint8)
    black_img_1  = np.zeros((height,width,3),dtype =np.uint8)
    current_event = event[prev:current,:]
    current_event_1 = event[prev_1:current_1,:]
    #black_img[current_event[:,0],current_event[:,1],:] = ([255,0,0],[0,0,255]) [check[i,2]==]
    #black_img[check[:,0],check[:,1],:] = [255,0,0] if check[:,2] ==1 else [0,0,255]
    
    # Look for index where 1 or 0 happens
    ones = np.where(current_event[:,2]==1)
    zeros = np.where(current_event[:,2]==0)
    ones_1 = np.where(current_event_1[:,2]==1)
    zeros_1 = np.where(current_event_1[:,2]==0)

    # Assign [255,0,0] to where index 1 happens, [0,0,255] to where index 0 happens
    black_img[current_event[ones][:,1],current_event[ones][:,0],:] = [255,0,0]
    black_img[current_event[zeros][:,1],current_event[zeros][:,0],:] = [0,0,255]
    black_img_1[current_event_1[ones_1][:,1],current_event_1[ones_1][:,0],:] = [255,0,0]
    black_img_1[current_event_1[zeros_1][:,1],current_event_1[zeros_1][:,0],:] = [0,0,255]
    img = black_img - black_img_1
    # Show the appended black_img
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 960,720)
    cv2.imshow('image',black_img)
    cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image1', 960,720)
    cv2.imshow('image1',img)
    #print(black_img.max())
    #print(prev,current)
    print(current)
    print(current_time)
    k = cv2.waitKey(10000)

    if k == ord('c'):
        i += 1
        j += 1
        prev,current = time_hist_cum[i-1], time_hist_cum[i]
        prev_1,current_1 = time_hist_cum[j-1], time_hist_cum[j]
        current_time = time_range[i+1]
        continue
    
    elif k == ord('x'):
        i -= 1
        j -= 1
        prev,current = time_hist_cum[i-1], time_hist_cum[i]
        prev_1,current_1 = time_hist_cum[j-1], time_hist_cum[j]
        current_time = time_range[i+1]
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
        data = np.concatenate([time_sec[prev:current,:],current_event],axis=1)
        np.savetxt('frame'+str(i+1)+'.csv',data,delimiter=',')
        continue

    elif k == ord('q'):
        break
    
    else:
        continue
    
    
    
cv2.destroyAllWindows()  