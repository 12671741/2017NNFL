import numpy as np
import matplotlib.pyplot as plt
cycleerrors = np.load('weights/cycleerrors.npy')
cycleerrorsval = np.load('weights/cycleerrorsval.npy')
np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=3)
W0 = np.load('weights/W0.npy')
W1 = np.load('weights/W1.npy')
print "W0"
print W0.shape
print W0
print "W1"
print W1.shape
print W1
plt.plot(cycleerrors,'black')
plt.plot(cycleerrorsval,'r')
plt.show()
