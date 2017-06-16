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
out5 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output5.avi',fourcc, 20.0, (640,480))

print 0
cap1 = cv2.VideoCapture('http://10.42.0.124:8070/?action=stream')
cap2 = cv2.VideoCapture('http://10.42.0.105:8080/?action=stream')
cap3 = cv2.VideoCapture('http://10.42.0.106:8060/?action=stream')
cap4 = cv2.VideoCapture('http://10.42.0.102:8090/?action=stream')
cap5 = cv2.VideoCapture('http://10.42.0.104:8050/?action=stream')

print 1
while True:
	print 2
        ret1, image1 = cap1.read()
        ret2, image2 = cap2.read()
        ret3, image3 = cap3.read()
        ret4, image4 = cap4.read()
	ret5, image5 = cap5.read()

        out1.write(image1)
        out2.write(image2)
        out3.write(image3)
        out4.write(image4)
	out5.write(image5)
	print "reavhed"

        cv2.imshow("Result",image1)
        if cv2.waitKey(1) == ord('q'):
            exit(0)


out1.release()
out2.release()
out3.release()
out4.release()
out5.release()

cap1.release()
cap2.release()
cap3.release()
cap4.release()
cap5.release()
