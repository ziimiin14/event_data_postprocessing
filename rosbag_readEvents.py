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

    output_filename = open(args.time_output_file,'wb')
    output_filename1 = open(args.event_output_file,'wb')

    count = 0 

    for topic, msg, t in bag.read_messages(topics=['/dvs/events']):
        for e in msg.events:
            data = np.array([e.ts.to_nsec()/1e9])
            data1 = np.array([int(e.x),int(e.y),int(e.polarity)],dtype=np.uint8)
            data.tofile(output_filename)
            data1.tofile(output_filename1)
            
        count += 1
        print(count)

    output_filename.close()
    output_filename1.close()


