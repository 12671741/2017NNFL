import threading
from imutils.video import WebcamVideoStream
import numpy as np
import imutils
import cv2
import Queue
from camthread import camthread
import camprofile
import classifier
#mport cnnlassifyer
def nonlin(x, deriv=False):  # Note: there is a typo on this line in the video
    if(deriv==True):
        return (x*(1-x))
    return 1/(1+np.exp(-x))  # Note: there is a typo on this line in the video
def classify(frame):
    l1 = nonlin(np.dot(frame, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l3 = nonlin(np.dot(l2, syn2))
    return l3

qin = Queue.Queue()
qout= Queue.Queue()

width=camprofile.width
height=camprofile.height
hrange=np.array(range(height/2-camprofile.croph,height/2+camprofile.croph))
wrange=np.array(range(width/2-camprofile.cropw,width/2+camprofile.cropw))


kind=0
vs = WebcamVideoStream(src=0).start()
myThread = camthread(qin,qout,1)
myThread.start()

f=open('../data/d.txt','r')
f=f.read()
f=f.split('\n')
f=f[0:len(f)-1]
totallen=len(f)

f=open('../data/d.txt','a')
d=np.zeros(26,np.uint8)
k=totallen
while True:
    frame = vs.read()
    #frame = imutils.resize(frame, width=width,height=height)
    frame=frame[height/2:height+height/2,:][:,width/2:width+width/2]
    frame = cv2.flip(frame, camprofile.flip)
    if qin.empty():
        qin.put(frame)

    if not qout.empty():
        outf = qout.get()
        key=cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            vs.stop()
            qin.put("q")
            break
        if 64<key<91:
            cframe=outf[hrange,:][:,wrange]
            fname='../data/img'+str(k)+'.jpg'
            cv2.imwrite(fname,cframe)
            k=k+1
            d[key-65]=1
            strd=str(d)+'\n'
            f.write(strd)
            d[key-65]=0
            print "captured to: ",fname," discrptor:",strd
        #outf=cv2.putText(outf,"TYPE: "+str(kind), (width/3,height*3/8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, 0)
        #frame=cv2.add(frame,outf)
        cv2.imshow("q: quit",outf)
