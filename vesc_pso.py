import numpy as np
import matplotlib.pyplot as plt


path = '../pso_code/vesc_event_syn_data'
path1 = '../vesc_event_syn_data'
path2 ='../CMBNB-demo'

pso= np.fromfile(path+'/6300ERPM_results.bin',dtype = 'float64')

pso_angVel = pso[2:]

vesc_ERPM = np.fromfile(path1+'/6300ERPM_VescMod_14102020.bin',dtype='int64')

cmbnb_angVel = np.fromfile(path2+'/6300ERPM.bin',dtype='float64')


cmbnb_angVel = cmbnb_angVel.reshape(3,int(cmbnb_angVel.shape[0]/3))
cmbnb_angVel = cmbnb_angVel.T
pso_angVel = pso_angVel.reshape(3,int(pso_angVel.shape[0]/3))
pso_angVel = pso_angVel.T
vesc_ERPM =vesc_ERPM.reshape(int(vesc_ERPM.shape[0]/2),2)
vesc_angVel = vesc_ERPM[:,1]*360/21/60
vesc_Time = vesc_ERPM[:,0]
vesc_Time = vesc_Time/1e9

pso_Time = np.arange(0,pso_angVel.shape[0]*0.01,0.01)
pso_yawRot = pso_angVel[:,1]*180*100/np.pi
pso_yawRot = -pso_yawRot
pso_Time = pso_Time[:1521]
pso_yawRot = pso_yawRot[:1521]
cmbnb_Time = np.arange(0,cmbnb_angVel.shape[0]*0.01,0.01)
cmbnb_yawRot = cmbnb_angVel[:,1]*180*100/np.pi
cmbnb_yawRot = -cmbnb_yawRot

# check = np.where((pso_Time>=0.3) & (pso_Time<=15))[0]
# check1=np.where(pso_yawRot[check]<100)[0]
# check1 =check1+check[0]
# pso_yawRot = np.delete(pso_yawRot,check1)
# pso_Time = np.delete(pso_Time,check1)

plt.scatter(pso_Time,pso_yawRot,color='red')
#plt.scatter(vesc_Time,vesc_angVel,color='blue')
plt.scatter(cmbnb_Time,cmbnb_yawRot,color='green')
plt.grid()
plt.show()