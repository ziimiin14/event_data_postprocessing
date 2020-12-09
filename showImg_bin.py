import numpy as np
import cv2

img = np.fromfile('../event_data_14102020/1260ERPM_Image_14102020.bin',dtype = np.uint8)
time_nano = np.fromfile('../event_data_14102020/1260ERPM_Image_14102020.bin',dtype = np.int64)
time_sec = time_nano/1e9
time_sec = time_sec-time_sec[0]


img = img.reshape(361,43200)


#User input for specific frame required
max_frame= img.shape[0]
user_input = input('Please choose the specific frame from 0 - '+str(max_frame-1)+': ')
user_input = int(user_input)

# Assign width,height value,
width =240
height=180

while(True):

    curr_img = img[user_input]
    curr_img = curr_img.reshape(height,width)


    # Show the appended black_img
    #cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image', 960,720)
    cv2.imshow('image',curr_img)
    #print(black_img.max())
    #print(prev,current)
    k = cv2.waitKey(10000)
    # print(curr_img.dtype)

    if k == ord('c'):
        user_input += 1
        print(user_input)
        continue
    
    elif k == ord('x'):
        user_input -= 1
        print(user_input)
        continue

    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('frame'+str(user_input)+'.png', curr_img)
        curr_img.tofile('frame'+str(user_input)+'.bin')
        print(curr_img.shape)

        # data = np.concatenate([time_sec[prev:current,:],current_event],axis=1)
        # np.savetxt('frame'+str(i+1)+'.csv',data,delimiter=',')
        continue

    elif k == ord('q'):
        break
    
    else:
        continue
    


cv2.destroyAllWindows()