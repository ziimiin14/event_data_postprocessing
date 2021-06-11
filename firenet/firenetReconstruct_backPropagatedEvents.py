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
import cv2
import backPropagatedEvents as bp




if __name__ == "__main__":


    parser = argparse.ArgumentParser(
        description='Evaluating a trained network')
    parser.add_argument('-c', '--path_to_model', required=True, type=str,
                        help='path to model weights')
    parser.add_argument('--event_path',default = "ttt", type=str)
    parser.add_argument('--time_path',default = "aaa", type=str)
    parser.add_argument('--imu_path',default = " ccc", type=str)
    parser.add_argument('--skipevents', default=0, type=int)
    parser.add_argument('--suboffset', default=0, type=int)
    parser.add_argument('--compute_voxel_grid_on_cpu', dest='compute_voxel_grid_on_cpu', action='store_true')
    parser.set_defaults(compute_voxel_grid_on_cpu=False)

    set_inference_options(parser)

    args = parser.parse_args()
    print(args)

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


    bp_events_iterator = bp.backPropagatedEvents(args.event_path, args.time_path, args.imu_path)
    with Timer('Processing entire iterator'):
        for event_window in bp_events_iterator:
            try:

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

                    num_events_in_window = 10
                    reconstructor.update_reconstruction(event_tensor, start_index + num_events_in_window, last_timestamp)

                    start_index += num_events_in_window

                    
                    # cv2.waitKey(1)

            except KeyboardInterrupt:
                
                cv2.destroyAllWindows()
                break

