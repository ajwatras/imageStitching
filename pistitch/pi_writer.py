import cv2
import urllib
import numpy as np

#pipeline = 'gst-launch-1.0 udpsrc port=5000 ! gdpdepay ! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false'
#pipeline = "udpsrc port=5000     ! gdpdepay     ! rtph264depay     ! avdec_h264     ! videoconvert ! appsink sync=false"
#pipeline = "gst-launch-1.0 udpsrc port=5000     ! gdpdepay     ! rtph264depay     ! avdec_h264     ! videoconvert     ! appsink sync=false"
#pipeline = "tcpclientsrc host=10.42.0.101 port=5000 ! gdpdepay ! rtph264depay ! video/x-h264, width=640, height=480, format=YUY2, framerate=49/1 ! ffdec_h264 ! autoconvert ! appsink sync=false"
#VIDEOWRITER_OUTPUT_PATH = '../data/pi_writer/'

#fourcc = cv2.VideoWriter_fourcc(*'XVID')

#out1 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output1.avi',fourcc, 20.0, (640,480))
#out2 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output2.avi',fourcc, 20.0, (640,480))
#out3 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output3.avi',fourcc, 20.0, (640,480))
#out4 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output4.avi',fourcc, 20.0, (640,480))

cap1 = cv2.VideoCapture('http://10.42.0.101:8010/?action=stream')
cap2 = cv2.VideoCapture('http://10.42.0.102:8020/?action=stream')
cap3 = cv2.VideoCapture('http://10.42.0.103:8030/?action=stream')
cap4 = cv2.VideoCapture('http://10.42.0.104:8040/?action=stream')
#cap1 = cv2.VideoCapture(,cv2.CAP_GSTREAMER)
#cap1 = cv2.VideoCapture("tcpclientsrc host=10.42.0.101 port=5000 ! gdpdepay ! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false")
#cap2 = cv2.VideoCapture("tcpclientsrc host=10.42.0.102 port=6000 ! gdpdepay ! rtph264depay ! avdec_h264 ! videoconvert ! appsink sync=false")
#cap1 = cv2.VideoCapture('udpsrc uri=udp://10.42.0.1:5000 auto-multicast=true caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264" ! rtph264depay ! decodebin ! appsink', cv2.CAP_GSTREAMER)

#cap1 = cv2.VideoCapture(pipeline,cv2.CAP_GSTREAMER)
print "before while loop"
while cap1.isOpened():
	print "in while loop"
        ret1, image1 = cap1.read()
        ret2, image2 = cap2.read()
        ret3, image3 = cap3.read()
        ret4, image4 = cap4.read()

#        out1.write(image1)
#        out2.write(image2)
#        out3.write(image3)
#        out4.write(image4)

        cv2.imshow("Result1",image1)
        cv2.imshow("Result2",image2)
        cv2.imshow("Result3",image3)
        cv2.imshow("Result4",image4)

        if cv2.waitKey(1) == ord('q'):
            exit(0)

print "after while loop"

#out1.release()
#out2.release()
#out3.release()
#out4.release()

cap1.release()
cap2.release()
cap3.release()
cap4.release()
