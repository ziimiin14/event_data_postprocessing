import rosbag
import numpy as np

#path = 'bag_file/6300ERPM_Max_Events.bag'
path = '../2-702/1260ERPM_Image_09102020.bag'

bag = rosbag.Bag(path)

events = []

for topic, msg, t in bag.read_messages(topics=['/dvs/image_raw']):
    events.append(msg)


events_list =[]
#init = events[0].events[0].ts.to_nsec()

# for i in range(len(events)):
#     for j in range(len(events[i].events)):
#         events_list.append([events[i].events[j].ts.to_nsec(), events[i].events[j].x,events[i].events[j].y,int(events[i].events[j].polarity)])



events_arr = np.array(events_list)

time_arr = events_arr[:,0]

events_arr = events_arr[:,1:]

events_arr = events_arr.astype(np.uint8)


events_arr.tofile('../2-702/8820ERPM_Events_09102020.bin')
time_arr.tofile('../2-702/8820ERPM_Time_09102020.bin')
