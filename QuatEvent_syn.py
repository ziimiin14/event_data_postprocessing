import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d

time_opti = np.fromfile('data_set1/6300ERPM_Max_Opti_Rigid_Time.bin',dtype=np.int64)
time_event = np.fromfile('data_set1/6300ERPM_Max_Time.bin',dtype=np.int64)
opti = np.fromfile('data_set1/6300ERPM_Max_Opti_Rigid.bin',dtype=np.float64)
event = np.fromfile('data_set1/6300ERPM_Max_Events.bin',dtype=np.uint8)


# time_opti = np.fromfile('data_set1/6300ERPM_Max_Opti_TimeMod.bin',dtype=np.int64)
# time_event = np.fromfile('data_set1/6300ERPM_Max_TimeMod.bin',dtype=np.int64)
# opti = np.fromfile('data_set1/6300ERPM_Max_Opti_QuatMod.bin',dtype=np.float64)
# event = np.fromfile('data_set1/6300ERPM_Max_EventMod.bin',dtype=np.uint8)

time_opti = time_opti.reshape(int(time_opti.shape[0]),1)
time_event = time_event.reshape(int(time_event.shape[0]),1)
opti = opti.reshape(int(opti.shape[0]/7),7)
event = event.reshape(int(event.shape[0]/3),3)

time_interval = 5555555

time_init = time_event[0]
time_end = time_event[-1]
time_range = np.arange(time_init,time_end+time_interval,time_interval)
time_hist,time_binEdge = np.histogram(time_event,bins=time_range)
time_hist_cum = np.cumsum(time_hist)


check1 = np.where(time_hist>3000)
time_before_rotate= time_binEdge[check1[0][0]]-time_binEdge[0]
#time_before_rotate1=


time_opti_new = np.arange(time_opti[0],time_opti[-1],time_interval)
quat = opti[:,0:4]
quat = quat[:time_opti_new.shape[0],:]
r = R.from_quat(quat)
euler = r.as_euler('xyz',degrees=True)
yaw = euler[:,2]
neg = np.where(yaw<0)
yaw[neg] += 360
diff_angle = np.diff(yaw)
# find negative values
neg = np.where(diff_angle<-1)
diff_angle[neg] += 360

diff_time = np.diff(time_opti_new)

angular_vel = diff_angle/(diff_time/1e9)
check2 = np.where(angular_vel>500)
time_before_rotate1 = time_opti_new[check2[0][0]]-time_opti_new[0]

result = time_before_rotate1- time_before_rotate

if result==0:
    time_event = time_event - time_event[0]
    time_opti = time_opti_new - time_opti_new[0]

elif result>0:
    temp = np.where(time_opti_new> time_opti_new[0]+result)
    time_opti_new = time_opti_new[temp]
    quat = quat[temp]
    time_event = time_event - time_event[0]
    time_opti = time_opti_new - time_opti_new[0]
    
if result<0:
    temp = np.where(time_event[:,0]> time_event[0,0]+abs(result))
    time_event = time_event[temp]
    event = event[temp]
    time_event = time_event - time_event[0]
    time_opti = time_opti_new - time_opti_new[0]

# time_min = time_event[0]
# time_max = time_opti[-1]


# fig,(ax1,ax2) = plt.subplots(2)
# ax1.plot(time_binEdge[:-1],time_hist,color='red')
# ax1.grid()
# ax1.set_xlim(time_min,time_max)
# ax2.plot(time_opti,euler[:,2],color='blue')
# ax2.grid()
# ax2.set_xlim(time_min,time_max)
# plt.show()


# x=1
# data = np.loadtxt('6300ERPM_2000_Opti_Tf.txt',delimiter=',')
# time = data[:,0]
# #time = time/1e9
# quat = data[:,1:5]
# quat = quat[:-2]
#time_mod = np.arange(time_opti[0],time_opti[-1],1*1e9/180)

# time_sec = np.fromfile('6300ERPM_2000_Time_2.bin',dtype=np.int64)

check = np.where(time_opti<time_event[-1])

time_opti = time_opti[check[0]]
quat = quat[check[0]]

if time_event[-1]-time_opti[-1]>0:
    temp = time_event[-1]
    time_opti = time_opti.tolist()
    time_opti.append(temp.item())
    time_opti = np.array(time_opti)
    temp1 = [0,0,0,0]
    quat = quat.tolist()
    quat.append(temp1)
    quat = np.array(quat)
    quat[-1] = quat[-2]


time_opti = time_opti.reshape(-1,1)
time_event = time_event.reshape(-1,1)

time_opti.tofile('data_set1/6300ERPM_Max_Opti_TimeMod.bin')
time_event.tofile('data_set1/6300ERPM_Max_TimeMod.bin')
event.tofile('data_set1/6300ERPM_Max_EventMod.bin')
quat.tofile('data_set1/6300ERPM_Max_Opti_QuatMod.bin')
# result = np.append(time_mod,quat,axis=1)
# result.tofile('6300ERPM_2000_Opti_New.bin')
