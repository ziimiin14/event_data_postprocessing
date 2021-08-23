import rosbag
import numpy as np
import argparse
import os


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("bag",help = "ROS bag file to extract")
    parser.add_argument("--imu_output_file",default="time_extracted_data",help="binary file to extract the imu data")
    args = parser.parse_args()

    path = args.bag
    bag = rosbag.Bag(path)

    output_filename = open(args.imu_output_file,'wb')

    count = 0 

    for topic, msg, t in bag.read_messages(topics=['/dvs/imu']):
        data = [float(int(msg.header.stamp.secs)+(int(msg.header.stamp.nsecs)/1e9)),msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z]
        data = np.array(data,dtype=np.float64)
        data.tofile(output_filename)

        count += 1
        print(count)


    output_filename.close()


