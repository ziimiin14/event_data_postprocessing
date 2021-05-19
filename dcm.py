import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


def Rz(x):
    R = np.zeros((3,3))
    R[0,0] = np.cos(x)
    R[0,1] = -np.sin(x)
    R[1,0] = np.sin(x)
    R[1,1] = np.cos(x)
    R[2,2] = 1

    return R


def Ry(x):
    R = np.zeros((3,3))
    R[0,0] = np.cos(x)
    R[0,2] = np.sin(x)
    R[2,0] = -np.sin(x)
    R[2,2] = np.cos(x)
    R[1,1] = 1

    return R

def Rx(x):
    R = np.zeros((3,3))
    R[1,1] = np.cos(x)
    R[1,2] = -np.sin(x)
    R[2,1] = np.sin(x)
    R[2,2] = np.cos(x)
    R[0,0] = 1

    return R

z1 = 0
y1 = 0
x1 = np.pi/2

z2 = np.pi/4
y2 = 0
x2 = 0

z3 = np.pi/4
y3 = 0
x3 = np.pi/2


# Point rotation with respect to original frame (old frame) [for camera] . Sequence is ZYX
RR1 = Rz(z1)@Ry(y1)@Rx(x1)
RR2 = Rz(z2)@Ry(y2)@Rx(x2)
RR3 = Rz(z3)@Ry(y3)@Rx(x3) 
RRR_p = RR2@RR1
# print(RR3@RR1.T)
# print(RR2)
print(RRR_p)
# Check whether the point rotation is correct
print((RRR_p-RR3) < 0.0000000001)

# Frame rotatione (from frame world -> frame 1 ,then frame 1 -> frame 2)
RRR_f = RR1.T@RR2.T # Whenever do the transpose for frame rotation, it is important to note that the sequence will be changed to XYZ
print(RRR_f)
# Check whether the frame rotation is correct
print((RR3.T - RRR_f) < 0.00000001) 

# Quaternion Version
q1 = Quaternion(0.7071068,0.7071068,0,0) # Similar to RR1
q2 = Quaternion(0.9238795,0,0,0.3826834) # Similar to RR2
q3 = Quaternion(0.6532815,0.6532815,0.2705981,0.2705981) # Similar to RR3
qq_p = q2*q1
qq_f=q1.conjugate*q2.conjugate

print('qq_p=',qq_p)
print('qq_f=',qq_f)

