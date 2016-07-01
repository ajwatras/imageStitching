import numpy as np
import cv2
import time

CALIBRATION_FRAMES = 10
THRESHOLD = 10

background_frame = np.zeros([480,640,3]) 
radial_dst = np.array([-0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628])
#radial_dst = [-0.37801445,  0.38328306, -0.01922729, -0.01341008, -0.60047771]
mtx = np.array([[ 444.64628787,    0.,          309.40196271],[   0.,          501.63984347,  255.86111216],[   0.,            0.,            1.        ]])


vid = cv2.VideoCapture('../data/testVideo/office/output1.avi')
#Throw out opening frames, they are usually garbage.
for k in range(0,5):
	success,frame = vid.read()

#We assume that only background is in the frame initially. So an average of the initial frames gives us an estimate of the background image. 
for x in range(0,CALIBRATION_FRAMES):
	success,frame = vid.read()
	background_frame = background_frame + frame

background_frame = (background_frame/CALIBRATION_FRAMES).astype('uint8')
background_gray = cv2.cvtColor(background_frame,cv2.COLOR_BGR2GRAY)

while success:
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	diff_img = np.abs(gray_frame - background_gray)

	fgMask = (diff_img > THRESHOLD).astype('int')
	print fgMask
	cv2.imshow('video',256*fgMask)
	cv2.waitKey(100)


	success,frame = vid.read()
	
cv2.imshow('frame',background_gray)
cv2.waitKey(0)

vid.release()
cv2.destroyAllWindows()





