import rosbag
import numpy as np

path = '../bag_file/8820ERPM_Opti.bag'

bag = rosbag.Bag(path)

opti = []

# tf = []

for topic, msg, t in bag.read_messages(topics=['/vrpn_client_node/RigidBody021/pose']):
    opti.append(msg)

# for topic, msg, t in bag.read_messages(topics=['/tf']):
#     tf.append(msg)


pose_list =[]
quat_list =[]
time_list=[]
pose_tf_list = []
quat_tf_list = []
time_tf_list = []
init = opti[0].header.stamp.to_nsec()
# init1 = tf[0].transforms[0].header.stamp.to_nsec()

for i in range(len(opti)):
    pose_list.append([opti[i].pose.position.x,opti[i].pose.position.y,opti[i].pose.position.z])
    quat_list.append([opti[i].pose.orientation.x,opti[i].pose.orientation.y,opti[i].pose.orientation.z,opti[i].pose.orientation.w])
    time_list.append([opti[i].header.stamp.to_nsec()])

# for i in range(len(tf)):
#     for j in range(len(tf[i].transforms)):
#         pose_tf_list.append([tf[i].transforms[j].transform.translation.x,tf[i].transforms[j].transform.translation.y,tf[i].transforms[j].transform.translation.z])
#         quat_tf_list.append([tf[i].transforms[j].transform.rotation.x,tf[i].transforms[j].transform.rotation.y,tf[i].transforms[j].transform.rotation.z,tf[i].transforms[j].transform.rotation.w])
#         time_tf_list.append([tf[i].transforms[j].header.stamp.to_nsec()])
pose_arr = np.array(pose_list)
quat_arr = np.array(quat_list)
time_arr = np.array(time_list,dtype=np.int64)
# pose_tf_arr = np.array(pose_tf_list)
# quat_tf_arr = np.array(quat_tf_list)
# time_tf_arr = np.array(time_tf_list,dtype=np.int64)

# rigid_arr = np.concatenate((quat_arr,pose_arr),axis=1)
# tf_arr = np.concatenate((quat_tf_arr,pose_tf_arr),axis=1)

print(quat_arr.dtype),print(quat_arr.shape),print(time_arr.dtype),print(time_arr.shape)
quat_arr.tofile('../bin_file/8820ERPM_Quat.bin')
time_arr.tofile('../bin_file/8820ERPM_Time.bin')
# rigid_arr.tofile('6300ERPM_Max_Opti_Rigid.bin')
# time_arr.tofile('6300ERPM_Max_Opti_Rigid_Time.bin')
#np.savetxt('6300ERPM_2000_Opti_Rigid.txt',rigid_arr,delimiter=',')
#np.savetxt('6300ERPM_2000_Opti_Rigid_Time.txt',time_arr,delimiter=',')
#np.savetxt('6300ERPM_2000_Opti_Tf.txt',tf_arr,delimiter=',')
