from pickle import STACK_GLOBAL
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import cv2
import os

class Events_Opti:
    def __init__(self,path_to_event_file,path_to_time_file,path_to_opti_file):

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
        time_interval = 1/100

        # Load opti data
        opti = np.fromfile(path_to_opti_file,dtype=np.float64)
        opti = opti.reshape(-1,5)
        self.opti_time = opti[:,0]
        self.opti_time = self.opti_time.reshape(-1,1)
        opti_quat = opti[:,1:]
        

        # Find the start time and end time for opti track data to sync with event data time
        findFirst = np.where(self.opti_time[:,0]<self.time_sec[0,0])[0][-1] 
        findLast = np.where(self.opti_time[:,0]>self.time_sec[-1,0])[0][0]+1

        self.time_sec = self.time_sec - self.opti_time[findFirst,0]
        self.opti_time = self.opti_time -  self.opti_time[findFirst,0]

        self.time_sec = self.time_sec.astype(np.float32)
        self.opti_time = self.opti_time.astype(np.float32)


        # Construct time histogram with respect to the opti_time (findFirst and findLast)
        time_init = self.opti_time[findFirst,:]
        time_end = self.opti_time[findLast,:]
        # print(time_init,time_end)
        # print(self.time_sec[0],self.time_sec[-1])
        time_range = np.arange(time_init,time_end,time_interval)
        time_hist,time_binEdge = np.histogram(self.time_sec,bins=time_range)
        self.time_hist_cum = np.cumsum(time_hist)

        quat = np.roll(opti_quat,1,axis=1)
        self.quat_world = []


        for i in range(quat.shape[0]):
            self.quat_world.append(Quaternion(quat[i,:]))


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

            res = np.hstack([specific_time,specific_event])

            self.i += 1
            return res

        raise StopIteration