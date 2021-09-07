import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion



# Initialize image dimension
width =240
height=180
    
# Initialize K matrix
# K= np.matrix([[322.27115293, 0., 156.60442714], [0., 323.8544765, 116.09022504],[ 0., 0., 1.]]) # dvxplorer k-matrix
K= np.matrix([[323.72678681198357, 0., 120.54931481123528], [0., 323.06065962539299, 87.519021071803550],[ 0., 0., 1.]],dtype=np.float32) # dvs240 k-matrix 
K_arr = np.array(K)
K_I_arr = np.array(K.I)

# Load events data from bin file
event = np.fromfile('../dvs240_data_set/eventStruct_09032021_fly/bin_file/kdvs240_Event_09032021_3.bin',dtype=np.uint8)

event = event.reshape(-1,3)

# Load time data (event) from bin file
# time_sec = np.fromfile('../dvxplorer_data_set/event_data_05122021/bin_file/kratos_eventTime_05122021_2.bin',dtype=np.float64)
time_sec = np.fromfile('../dvs240_data_set/eventStruct_09032021_fly/bin_file/kdvs240_Time_09032021_3.bin',dtype=np.float32)
time_sec = time_sec.reshape(-1,1) 
time_sec = time_sec - time_sec[0,0]
time_interval = 1/100

# Construct time histogram wrt event time
time_init = time_sec[0,0]
time_end = time_sec[-1,0]
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

xedges = np.arange(0,width+1,1)
yedges = np.arange(0,height+1,1)

while(True):
    ## For each frame:
    # Declare a specific_event and specific_time for specific frame requested
    specific_event = event[prev:current,:]
    specific_time = time_sec[prev:current,:]


    black_img_1  = np.zeros((height,width,3),dtype =np.uint8)

    # Normalize the image over the max value of the pixel
    black_img_1,yed,xed=np.histogram2d(specific_event[:,1],specific_event[:,0],bins=(yedges,xedges))
    black_img_1 = black_img_1/ black_img_1.max()


    ## Red Blue Event Frame
    # ones_1 = np.where(specific_event[:,2]==1)
    # zeros_1 = np.where(specific_event[:,2]==0)
    # black_img_1[specific_event[ones_1][:,1],specific_event[ones_1][:,0],:] = [255,0,0]
    # black_img_1[specific_event[zeros_1][:,1],specific_event[zeros_1][:,0],:] = [0,0,255]


    # Show the appended black_img
    cv2.namedWindow('image1',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image1', 640,480)
    cv2.imshow('image1',black_img_1)

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


