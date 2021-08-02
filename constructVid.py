import cv2
import glob
import numpy as np
import os

path_bp = './firenet/reconstruction/reconstructed_05122021_2_10ms_bp_opti/reconstruction/*.png'
path_bp1 = './firenet/reconstruction/reconstructed_05122021_2_10ms_bp_opti/reconstruction/events/*.png'
path = './firenet/reconstruction/reconstructed_05122021_2_10ms_nbp_opti/reconstruction/*.png'
path1 = './firenet/reconstruction/reconstructed_05122021_2_10ms_nbp_opti/reconstruction/events/*.png'


file_bp = sorted(glob.glob(path_bp),key=os.path.basename)
file_bp1 = sorted(glob.glob(path_bp1),key=os.path.basename)
file = sorted(glob.glob(path),key=os.path.basename)
file1 = sorted(glob.glob(path1),key=os.path.basename)


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('event_dvxplorer_05122021_fly_slow.avi',fourcc, 15.0, (640,480))

for a,b,c,d in zip(file_bp,file_bp1,file,file1):
    w = cv2.imread(a)
    x = cv2.imread(b)
    y = cv2.imread(c)
    z = cv2.imread(d)
    temp = np.vstack([x,w])
    temp1 = np.vstack([z,y])
    img = np.hstack([temp1,temp])
    out.write(img)

out.release()
    