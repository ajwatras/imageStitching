import cv2
import urllib
import numpy as np
import stitcher2


VIDEOWRITER_OUTPUT_PATH = '../data/pi_writer/'

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out1 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output1.avi',fourcc, 20.0, (640,480))
out2 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output2.avi',fourcc, 20.0, (640,480))
out3 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output3.avi',fourcc, 20.0, (640,480))
out4 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output4.avi',fourcc, 20.0, (640,480))


stream1=urllib.urlopen('http://10.42.0.105:8060/?action=stream')
stream2=urllib.urlopen('http://10.42.0.124:8070/?action=stream')
stream3=urllib.urlopen('http://10.42.0.104:8080/?action=stream')
stream4=urllib.urlopen('http://10.42.0.102:8090/?action=stream')

CALIBRATION = True
flag1 = False
flag2 = False
flag3 = False
flag4 = False

bytes1=''
bytes2=''
bytes3=''
bytes4=''


image1 = np.zeros((960,1280,3)).astype(np.uint8)
image2 = np.zeros((960,1280,3)).astype(np.uint8)
image3 = np.zeros((960,1280,3)).astype(np.uint8)
image4 = np.zeros((960,1280,3)).astype(np.uint8)
while True:
    result = np.zeros((1900,1900,3)).astype(np.uint8)

    bytes1+=stream1.read(10000)
    bytes2+=stream2.read(10000)
    bytes3+=stream3.read(10000)
    bytes4+=stream4.read(10000)

    a1 = bytes1.find('\xff\xd8')
    b1 = bytes1.find('\xff\xd9')
    a2 = bytes2.find('\xff\xd8')
    b2 = bytes2.find('\xff\xd9')
    a3 = bytes3.find('\xff\xd8')
    b3 = bytes3.find('\xff\xd9')
    a4 = bytes4.find('\xff\xd8')
    b4 = bytes4.find('\xff\xd9')


    if a1!=-1 and b1!=-1:
        flag1 = True
        jpg1 = bytes1[a1:b1+2]
        bytes1= bytes1[b1+2:]

        image1 = cv2.imdecode(np.fromstring(jpg1, dtype=np.uint8),cv2.IMREAD_COLOR)
        #cv2.imshow('stream1',image1)
        
	if cv2.waitKey(1) == ord('q'):
            exit(0)

    if a2!=-1 and b2!=-1:
        flag2 = True        
        jpg2 = bytes2[a2:b2+2]
        bytes2= bytes2[b2+2:]
        image2 = cv2.imdecode(np.fromstring(jpg2, dtype=np.uint8),cv2.IMREAD_COLOR)
        #cv2.imshow('stream2',image2)
	if cv2.waitKey(1) == ord('q'):
            exit(0)

    if a3!=-1 and b3!=-1:
        flag3 = True
        jpg3 = bytes3[a3:b3+2]
        bytes3= bytes3[b3+2:]
        image3 = cv2.imdecode(np.fromstring(jpg3, dtype=np.uint8),cv2.IMREAD_COLOR)
        #cv2.imshow('stream3',image3)
	if cv2.waitKey(1) == ord('q'):
            exit(0)

    if a4!=-1 and b4!=-1:
        flag4 = True
        jpg4 = bytes4[a4:b4+2]
        bytes4= bytes4[b4+2:]
        image4 = cv2.imdecode(np.fromstring(jpg4, dtype=np.uint8),cv2.IMREAD_COLOR)
        #cv2.imshow('stream4',image4)
	if cv2.waitKey(1) == ord('q'):
            exit(0)

    if flag1 and flag2 and flag3 and flag4:
        flag1 = False
        flag2 = False
        flag3 = False
        flag4 = False


        out1.write(image1)
        out2.write(image2)
        out3.write(image3)
        out4.write(image4)


        cv2.imshow("Result",result)
        if cv2.waitKey(1) == ord('q'):
            exit(0)


out1.release()
out2.release()
out3.release()
out4.release()