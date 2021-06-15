from pickle import STACK_GLOBAL
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import cv2
import os

class backPropagatedEvents:
    def __init__(self,path_to_event_file,path_to_time_file,path_to_imu_file):

        # Initialize image dimension
        self.width =320
        self.height=240

        # Initialize K matrix
        # self.K= np.matrix([[326.1387442, 0., 158.28624894], [0., 326.84650981, 112.96798711],[ 0., 0., 1.]])
        self.K= np.matrix([[322.27115293, 0., 156.60442714], [0., 323.8544765, 116.09022504],[ 0., 0., 1.]]) # default
        self.K_arr = np.array(self.K)
        self.K_I_arr = np.array(self.K.I)

        # Load events data from bin file
        self.event = np.fromfile(path_to_event_file,dtype=np.uint16)
        self.event = self.event.reshape(-1,3)
        self.event = self.event.astype(np.float32)

        # Load time data (event) from bin file
        self.time_sec = np.fromfile(path_to_time_file,dtype=np.float64)

        self.time_sec = self.time_sec.reshape(-1,1) 
        time_interval = 1/200

        # Load imu data
        imu = np.fromfile(path_to_imu_file,dtype=np.float64)
        imu = imu.reshape(-1,4)
        self.imu_time = imu[:,0]
        self.imu_time = self.imu_time.reshape(-1,1)
        imu_ang = imu[:,1:]
        imu_ang = imu_ang*180/np.pi

        #Calcaulate angle rotated
        diff_imu_time = np.diff(self.imu_time,axis=0)
        diff_angle = diff_imu_time*imu_ang[0:-1,:]

        self.angle = np.zeros(imu_ang.shape)

        for i in range(1,self.angle.shape[0]):
            self.angle[i] = diff_angle[i-1] + self.angle[i-1]

        self.angle[:,1] = -self.angle[:,1]
        self.angle[:,2] = -self.angle[:,2]

        # Find the start time and end time for opti track data to sync with event data time
        findFirst = np.where(self.time_sec[:,0]<self.imu_time[0,0])[0][-1] + 1
        findLast = np.where(self.time_sec[:,0]>self.imu_time[-1,0])[0][0]


        # Slice the time_sec and event from findFirst to findLast
        self.time_sec = self.time_sec[findFirst:findLast,:]
        self.event = self.event[findFirst:findLast,:]


        self.time_sec = self.time_sec - self.imu_time[0,0]
        self.imu_time = self.imu_time - self.imu_time[0,0]

        self.time_sec = self.time_sec.astype(np.float32)
        self.imu_time = self.imu_time.astype(np.float32)




        # Construct time histogram with respect to the opti_time (findFirst and findLast)
        time_init = self.imu_time[0,:]
        time_end = self.imu_time[-1,:]
        time_range = np.arange(time_init,time_end+time_interval,time_interval)
        time_hist,time_binEdge = np.histogram(self.time_sec,bins=time_range)
        self.time_hist_cum = np.cumsum(time_hist)

        # Maximum frame and initialize first time interval event frame
        self.max_frame= time_hist.shape[0]
        self.i = 0
        
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.i < self.max_frame:
            if self.i == 0:
                prev = 0
                current = self.time_hist_cum[self.i]
            
            else:
                prev = self.time_hist_cum[self.i-1]
                current = self.time_hist_cum[self.i]
            while prev == current:
                self.i += 1
                prev = self.time_hist_cum[self.i-1]
                current = self.time_hist_cum[self.i]
                
            # Declare a specific_event and specific_time for specific frame requested
            specific_event = self.event[prev:current,:]
            specific_time = self.time_sec[prev:current,:]

            # Find the initial opti time index(temp_init) that is lesser than first specific time and
            # last opti time index(temp_last) that is more that last specific time
            temp_init = np.where(self.imu_time[:,0]<=specific_time[0,0])[0][-1]
            temp_last = np.where(self.imu_time[:,0]>=specific_time[-1,0])[0][0]
            # temp_init = self.imu_time[:,0]<=specific_time[0,0]
            # temp_last = self.imu_time[:,0]>=specific_time[-1,0]
        

            # Define the rotation ratio 
            diff_time_numerator = specific_time-self.imu_time[temp_init,0]
            diff_time_denominator = self.imu_time[temp_last,0]-self.imu_time[temp_init,0]
            # diff_time_numerator = specific_time-self.imu_time[temp_init][-1]
            # diff_time_denominator = self.imu_time[temp_last][0]-self.imu_time[temp_init][-1]
            ratio = diff_time_numerator/diff_time_denominator

            # Declare the quaternions with respect to the initial and last opti time index
            a1 = self.angle[temp_init]
            a2 = self.angle[temp_last]
            # a1 = self.angle[temp_init][-1]
            # a2 = self.angle[temp_last][0]

            # Get rotated angle between 2 pose
            euler = a2-a1

            # q3 = q2*q1.conjugate

            # Convert quaternion to euler with the sequence of ZYX
            euler[0],euler[2] = euler[2],euler[0]
            euler = euler.reshape(1,-1)

            # Compute euler with rotation ratio
            # to obtain euler_arr (rotation with respect to each specific event time frame))
            euler_arr = np.dot(ratio,euler)
            euler_arr = euler_arr-euler_arr[0,:]

            # Convert the euler arr to dcm
            r1 = R.from_euler('ZYX',euler_arr,degrees=True)
            dcm = r1.as_dcm()
            dcm_T = np.einsum('iab->iba',dcm)

            # Compute a 3 by N dimension array with respect to the specific events
            x_arr = specific_event[:,0]
            y_arr = specific_event[:,1]
            z_arr = np.ones(specific_event.shape[0],dtype=np.uint16)
            specific_pos_pixel = np.reshape((x_arr,y_arr,z_arr),(3,-1)) # Points in pixel frame

            # Compute points in camera frame
            specific_pos_camera = self.K_I_arr@specific_pos_pixel

            # Back propagate points in camera frame
            BxC = np.einsum('iab,bi->ai',dcm_T,specific_pos_camera)

            final_pos_pixel = self.K_arr@BxC
            final_pos_pixel = final_pos_pixel.T
            final_pos_pixel[:,0], final_pos_pixel[:,1],final_pos_pixel[:,2]=final_pos_pixel[:,0]/final_pos_pixel[:,2],final_pos_pixel[:,1]/final_pos_pixel[:,2],final_pos_pixel[:,2]/final_pos_pixel[:,2]
            final_pos_pixel = np.round(final_pos_pixel)
            final_pos_pixel[:,2] = self.event[prev:current,2]

            current_event = final_pos_pixel
            boolean = (current_event[:,0]<self.width) & (current_event[:,1]<self.height) & (current_event[:,1]>=0) & (current_event[:,0]>=0)
            current_event  = current_event[boolean,:]
            current_time = specific_time[boolean,:]

            res = np.hstack([current_time,current_event])

            self.i += 1
            return res

        raise StopIteration