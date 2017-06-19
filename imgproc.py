import numpy as np
import cv2
import camprofile
import classifier

kernel = camprofile.kernal
width=camprofile.width
height=camprofile.height
hrange=np.array(range(height/2-camprofile.croph,height/2+camprofile.croph))
wrange=np.array(range(width/2-camprofile.cropw,width/2+camprofile.cropw))
np.set_printoptions(precision=2)
str=camprofile.str10
acu=[]
def imgproc(frame,COLLECT=False):
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = cv2.medianBlur(frame,5)
    frame=cv2.bilateralFilter(frame,10,30,30)

    frame=cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    out = np.zeros((height,width,3),np.uint8)
    cv2.rectangle(out,  (width/2-18,height/2-18), (width/2+18,height/2+18), (0,255,0),2)
    for i in range(-3,3):
        for j in range(-3,3):
            l3c=classifier.classify(frame[hrange+2*i,:][:,wrange+2*j])
            acu.append(l3c)
            if len(acu)>200:
                acu.pop(0)
    #print np.sum(np.array(acu),axis=0)
    out=cv2.putText(out,str[np.argmax(np.sum(np.array(acu),axis=0))], (width/3,height*3/8), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,255,255))
    print np.sum(np.array(acu),axis=0)
    if COLLECT:
        frame=cv2.rectangle(frame,  (width/2-18,height/2-18), (width/2+18,height/2+18),0,2)
        return frame
    return out
