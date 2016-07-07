import numpy as np
import cv2
import time

CALIBRATION_FRAMES = 10
THRESHOLD = 20
DILATION_KERNEL = np.ones([3,3])
EROSION_LOOPS = 3
DILATION_LOOPS = 5

background_frame = np.zeros([480,640,3]) 
radial_dst = np.array([-0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628])
#radial_dst = [-0.37801445,  0.38328306, -0.01922729, -0.01341008, -0.60047771]
mtx = np.array([[ 444.64628787,    0.,          309.40196271],[   0.,          501.63984347,  255.86111216],[   0.,            0.,            1.        ]])


vid = cv2.VideoCapture('../data/testVideo/FGsep2/output1.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()

success,frame = vid.read()

#We assume that only background is in the frame initially. So an average of the initial frames gives us an estimate of the background image. 
while success:
	fgmask = fgbg.apply(frame)
	out_frame = np.zeros(fgmask.shape)

	fgmask = cv2.erode(fgmask,DILATION_KERNEL,iterations=EROSION_LOOPS)
	fgmask = cv2.dilate(fgmask,DILATION_KERNEL,iterations=EROSION_LOOPS)
	fgmask = cv2.dilate(fgmask,DILATION_KERNEL,iterations=DILATION_LOOPS)

	im2, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for i in range(0,len(contours)):
		if (i % 1 == 0):
			cnt = contours[i]

			x,y,w,h = cv2.boundingRect(cnt)
			cv2.drawContours(fgmask,contours,-1,(255,255,0),3)
			cv2.rectangle(fgmask,(x,y),(x+w,y+h),(255,0,0),2)
			out_frame[y:y+h,x:x+w] = np.ones([h,w])
			 

	#cv2.drawContours(im2, contours, -1, (0,255,0), 3)


	cv2.imshow('Video Feed',out_frame)
	cv2.waitKey(40)
	success,frame = vid.read()


cv2.imshow('frame',background_gray)
cv2.waitKey(0)

vid.release()
cv2.destroyAllWindows()





