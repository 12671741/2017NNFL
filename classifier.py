import numpy as np
from scipy.signal import convolve2d

W0 = np.load('weights/W0.npy')
W1 = np.load('weights/W1.npy')
aug=np.array(-1).reshape(1,-1)

def bipola_logi(x, deriv=False):  #using the solftplus
    if(deriv==True):
        return 0.5*(1-x*x)
    return (1-np.exp(-x))/(1+np.exp(-x));

def classify(frame):
    frame=1-frame.astype(float)/128
    l0 = np.reshape(frame,(1,-1))
    l0 = np.hstack((l0,aug))
    l1 = bipola_logi(np.dot(l0, W0))
    l1 = np.hstack((l1,aug))
    l2 = bipola_logi(np.dot(l1, W1))
    return l2
