import rosbag
import numpy as np
import argparse
import os


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("bag",help = "ROS bag file to extract")

    parser.add_argument("--time_output_file",default="time_extracted_data",help="binary file to extract the time data")
    parser.add_argument("--event_output_file",default="event_extracted_data",help="binary file to extract the event data")
    args = parser.parse_args()

    path = args.bag
    bag = rosbag.Bag(path)

    output_filename = open(args.event_output_file,'wb')
    output_filename1 = open(args.time_output_file,'wb')


    count = 0 
    event = []
    time = []

    for topic, msg, t in bag.read_messages(topics=['/dvs/eventStruct']):
        for a in msg.eventArr.data:
            event.append(a)
        for b in msg.eventTime.data:
            time.append(b)
        eventArr = np.array(event,dtype=np.uint8)
        timeArr = np.array(time,dtype=np.int64)
        eventArr.tofile(output_filename)
        timeArr.tofile(output_filename1)
        event = []
        time = []
            
        count += 1
        print(count)

    output_filename.close()
    output_filename1.close()


