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
    boxBlur_3(tcl,scl,w,h,int((bxs[1]-1)/2))
    boxBlur_3(scl,tcl,w,h,int((bxs[2]-1)/2))

def boxBlur_3(scl,tcl,w,h,r):
  
    scl,tcl = tcl,scl
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

def gaussBlur_4(scl,tcl,w,h,r):
    bxs = boxesForGauss(r,3)
    boxBlur_4(scl,tcl,w,h,int((bxs[0]-1)/2))
    boxBlur_4(tcl,scl,w,h,int((bxs[1]-1)/2))
    boxBlur_4(scl,tcl,w,h,int((bxs[2]-1)/2))

def boxBlur_4(scl,tcl,w,h,r):
    scl,tcl=tcl,scl    
    boxBlurH_4(tcl,scl,w,h,r)
    boxBlurT_4(scl,tcl,w,h,r)

def boxBlurH_4(scl,tcl,w,h,r):
    iarr = 1/(r+r+1)

    for i in range(h):
        ti = i*w
        li = ti
        ri = ti+r
        fv = scl[ti]
        lv = scl[ti+w-1]
        val = (r+1)*fv

        for j in range(r):
            val += scl[ti+j]

        for j in range(r+1):
            val += scl[ri]-fv
            tcl[ti] = np.round(val*iarr)
            # tcl[ti] = val*iarr
            ri += 1
            ti += 1

        for j in range(r+1,w-r,1):
            
            val += scl[ri]-scl[li]
            tcl[ti] = np.round(val*iarr)
            # tcl[ti] = val*iarr
            ri += 1
            li += 1
            ti += 1

        for j in range(w-r,w,1):
            val += lv - scl[li]
            tcl[ti] = np.round(val*iarr)
            # tcl[ti] = val*iarr
            li += 1
            ti += 1

        
def boxBlurT_4(scl,tcl,w,h,r):
    iarr = 1/(r+r+1)

    for i in range(w):
        ti = i
        li = ti
        ri = ti+r*w
        fv = scl[ti]
        lv = scl[ti+w*(h-1)]
        val = (r+1)*fv

        for j in range(r):
            val += scl[ti+j*w]

        for j in range(r+1):
            val += scl[ri]-fv
            tcl[ti] = np.round(val*iarr)
            # tcl[ti] = val*iarr
            ri += w
            ti += w

        for j in range(r+1,h-r,1):
            
            val += scl[ri]-scl[li]
            tcl[ti] = np.round(val*iarr)
            # tcl[ti] = val*iarr
            ri += w
            li += w
            ti += w

        for j in range(h-r,h,1):
            val += lv - scl[li]
            tcl[ti] = np.round(val*iarr)
            # tcl[ti] = val*iarr
            li += w
            ti += w
