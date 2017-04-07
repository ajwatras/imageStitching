import cv2
import urllib
import numpy as np
import stitcher2


stream1=urllib.urlopen('http://10.42.0.105:8060/?action=stream')


CALIBRATION = True
flag1 = False

bytes1=''



image1 = np.zeros((960,1280,3)).astype(np.uint8)
while True:
    result = np.zeros((1900,1900,3)).astype(np.uint8)

    bytes1+=stream1.read(10000)


    a1 = bytes1.find('\xff\xd8')
    b1 = bytes1.find('\xff\xd9')



    if a1!=-1 and b1!=-1:
        flag1 = True
        jpg1 = bytes1[a1:b1+2]
        bytes1= bytes1[b1+2:]

        image1 = cv2.imdecode(np.fromstring(jpg1, dtype=np.uint8),cv2.IMREAD_COLOR)
        #cv2.imshow('stream1',image1)
        
	#if cv2.waitKey(1) == ord('q'):
    #       exit(0)

    if flag1:
        flag1 = False




        cv2.imshow("Result",image1)
        if cv2.waitKey(1) == ord('q'):
            exit(0)


