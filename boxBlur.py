from ast import walk
import numpy as np


def boxesForGauss(sigma,n):
    wIdeal = np.sqrt((12*sigma*sigma/n)+1)
    wl = np.floor(wIdeal)
    wl = wl-1 if wl%2==0 else wl
    wu = wl+2

    mIdeal = (12*sigma*sigma - n*wl*wl - 4*n*wl - 3*n)/(-4*wl - 4)

    m = np.round(mIdeal)

    sizes = [wl if i < m else wu for i in range(n)]

    return sizes

def gaussBlur_3(scl,tcl,w,h,r):
    bxs = boxesForGauss(r,3)
    boxBlur_3(scl,tcl,w,h,int((bxs[0]-1)/2))
    boxBlur_3(scl,tcl,w,h,int((bxs[1]-1)/2))
    boxBlur_3(scl,tcl,w,h,int((bxs[2]-1)/2))

def boxBlur_3(scl,tcl,w,h,r):
    for i in range(len(scl)):
        
        tcl[i] = scl[i]
        # if i <=2:
        #     print('scl:',scl)
        #     print('tcl:',tcl)
        boxBlurH_3(tcl,scl,w,h,r)
        # if i <=2:
        #     print('scl1:',scl)
        #     print('tcl1:',tcl)
        boxBlurT_3(scl,tcl,w,h,r)


def boxBlurH_3(scl,tcl,w,h,r):
    for i in range(h):
        for j in range(w):
            val = 0
            for ix in range(j-r,j+r+1,1):
                x = min(w-1,max(0,ix))
                val += scl[i*w+x]

            tcl[i*w+j] = val/(r+r+1)

def boxBlurT_3(scl,tcl,w,h,r):
    for i in range(h):
        for j in range(w):
            val = 0
            for iy in range(i-r,i+r+1,1):
                y = min(h-1,max(0,iy))
                val += scl[y*w+j]

            tcl[i*w+j] = val/(r+r+1)



