from __future__ import print_function, absolute_import
from utils.loading_utils import load_model, get_device
from utils.event_readers import FixedSizeEventReader, FixedDurationEventReader
from utils.inference_utils import events_to_voxel_grid, events_to_voxel_grid_pytorch
from utils.timers import Timer
from image_reconstructor import ImageReconstructor
from options.inference_options import set_inference_options

import torch
import argparse
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import cv2

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion
import cv2
import os

# Initialize image dimension
width =320
height=240
    
# Initialize K matrix
K= np.matrix([[330.1570953377, 0., 161.9624665569], [0., 329.536232838, 110.80414596744],[ 0., 0., 1.]])
K_arr = np.array(K)
K_I_arr = np.array(K.I)

# Load events data from bin file
event = np.fromfile('../event_data_06022021/bin_file/kratos_eventOnly_06022021.bin',dtype=np.uint16)
event = event.reshape(-1,3)

# Load time data (event) from bin file
time_sec = np.fromfile('../event_data_06022021/bin_file/kratos_eventTime_06022021.bin',dtype=np.float64)
time_sec = time_sec.reshape(-1,1) 
time_interval = 1/100

# Load opti track data
imu = np.fromfile('../event_data_06022021/bin_file/kratos_IMU_06022021.bin',dtype=np.float64)
imu = imu.reshape(-1,4)
imu_time = imu[:,0]
imu_time = imu_time.reshape(-1,1)
imu_ang = imu[:,1:]
imu_ang = imu_ang*180/np.pi

#Calcaulate angle rotated
diff_imu_time = np.diff(imu_time,axis=0)
diff_angle = diff_imu_time*imu_ang[0:-1,:]

angle = np.zeros(imu_ang.shape)

for i in range(1,angle.shape[0]):
    angle[i] = diff_angle[i-1] + angle[i-1]

angle[:,1] = -angle[:,1]
angle[:,2] = -angle[:,2]

# Find the start time and end time for opti track data to sync with event data time
findFirst = np.where(time_sec[:,0]<imu_time[0,0])[0][-1] + 1
findLast = np.where(time_sec[:,0]>imu_time[-1,0])[0][0]


# Slice the time_sec and event from findFirst to findLast
time_sec = time_sec[findFirst:findLast,:]
event = event[findFirst:findLast,:]

# Construct time histogram with respect to the opti_time (findFirst and findLast)
time_init = imu_time[0,:]
time_end = imu_time[-1,:]
time_range = np.arange(time_init,time_end+time_interval,time_interval)
time_hist,time_binEdge = np.histogram(time_sec,bins=time_range)
time_hist_cum = np.cumsum(time_hist)

# Maximum frame and initialize first time interval event frame
max_frame= time_hist.shape[0]
i = 0
prev = 0
current = time_hist_cum[i]

if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()

    width,height = 320,240
    print('Sensor size: {} x {}'.format(width, height))


    # Load model
    model = load_model(args.path_to_model)
    device = get_device(args.use_gpu)

    model = model.to(device)
    model.eval()

    # Load image reconstructor class object
    reconstructor = ImageReconstructor(model, height, width, model.num_bins, args)

    initial_offset = args.skipevents
    sub_offset = args.suboffset
    start_index = initial_offset + sub_offset
    print(initial_offset,sub_offset,start_index)



    # device1 = DAVIS(noise_filter=False)
    device1 = DVXPLORER()
    device1.start_data_stream()
    # load new config
    # device1.set_bias_from_json("./scripts/configs/davis240c_config.json")
    device1.set_bias_from_json("./scripts/configs/dvxplorer_config.json")
    
    while True:
        try:
            (pol_events, num_pol_event,
            special_events, num_special_event,
            imu_events, num_imu_event) = \
                device1.get_event("events")
            if num_pol_event != 0:


                pol_events = pol_events.astype(np.float32)
                event_window = pol_events
                event_window[:,0] = event_window[:,0]/1e6



                with Timer('Processing entire dataset'):
                    last_timestamp = event_window[-1, 0]



                    with Timer('Building event tensor'):
                        if args.compute_voxel_grid_on_cpu:
                            event_tensor = events_to_voxel_grid(event_window,
                                                                num_bins=model.num_bins,
                                                                width=width,
                                                                height=height)
                            event_tensor = torch.from_numpy(event_tensor)
                        else:
                            event_tensor = events_to_voxel_grid_pytorch(event_window,
                                                                        num_bins=model.num_bins,
                                                                        width=width,
                                                                        height=height,
                                                                        device=device)

                    num_events_in_window = num_pol_event
                    reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

                    start_index += num_events_in_window

                
                # cv2.waitKey(1)

        except KeyboardInterrupt:
            
            device1.shutdown()
            cv2.destroyAllWindows()
            break

