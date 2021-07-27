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