import rosbag
import numpy as np

#path = 'bag_file/6300ERPM_Max_Events.bag'
path = '../event_data_14102020/3780ERPM_Image_14102020.bag'

bag = rosbag.Bag(path)

img = []

for topic, msg, t in bag.read_messages(topics=['/dvs/image_raw']):
    img.append(msg)


img_list =[]
time_list = []

for i in range(len(img)):
    img_list.append(list(img[i].data))
    time_list.append(img[i].header.stamp.to_nsec()) 
#init = events[0].events[0].ts.to_nsec()

# for i in range(len(events)):
#     for j in range(len(events[i].events)):
#         events_list.append([events[i].events[j].ts.to_nsec(), events[i].events[j].x,events[i].events[j].y,int(events[i].events[j].polarity)])



img_arr = np.array(img_list)

time_arr = np.array(time_list)

img_arr = img_arr.astype(np.uint8)
print(img_arr.shape,time_arr.shape)

img_arr.tofile('../event_data_14102020/3780ERPM_Image_14102020.bin')
time_arr.tofile('../event_data_14102020/3780ERPM_Time_14102020.bin')
