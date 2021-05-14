import rosbag
import numpy as np
import argparse
import os


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("bag",help = "ROS bag file to extract")
    parser.add_argument("--quat_output_file",default="quat_extracted_data",help="binary file to extract the time data")
    args = parser.parse_args()

    path = args.bag
    output_filename = args.quat_output_file

    bag = rosbag.Bag(path)

    opti = []

    for topic, msg, t in bag.read_messages(topics=['/vrpn_client_node/RigidBody007/pose']):
        opti.append(msg)

    pose_list =[]
    quat_list =[]
    time_list=[]

    for i in range(len(opti)):
        # pose_list.append([opti[i].pose.position.x,opti[i].pose.position.y,opti[i].pose.position.z])
        quat_list.append([opti[i].header.stamp.to_nsec()/1e9,opti[i].pose.orientation.x,opti[i].pose.orientation.y,opti[i].pose.orientation.z,opti[i].pose.orientation.w])
        # time_list.append([opti[i].header.stamp.to_nsec()/1e9])

    # pose_arr = np.array(pose_list)
    quat_arr = np.array(quat_list)
    # time_arr = np.array(time_list)

    print(quat_arr.dtype),print(quat_arr.shape)
    quat_arr.tofile(output_filename)

