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

	diff_img = (np.sum(np.abs(frame.astype('float') - background_frame.astype('float')),axis=2)/3).astype('uint8')
	
	T,fgMask = cv2.threshold(diff_img,THRESHOLD,255,cv2.THRESH_BINARY)


# We will attempt to denoise our binary image using dilation and erosion. 
	# In order to attempt to fill in holes we dilate the object followed by an equivalent erosion. 
	for k in range(0,DILATION_LOOPS):
		fgMask = cv2.dilate(fgMask,DILATION_KERNEL)
	for k in range(0,DILATION_LOOPS):
		fgMask = cv2.erode(fgMask,DILATION_KERNEL)

	# In order to remove noisy points in the image, we erode away the objects, then restore any remaining objects to their original size. Unfortunately, the dilation algorithm does not exactly undo the erosion algorithm. So some artifacts are introduced. 
	for k in range(0,EROSION_LOOPS):
    		fgMask = cv2.erode(fgMask,DILATION_KERNEL)
	for k in range(0,EROSION_LOOPS):
		fgMask = cv2.dilate(fgMask,DILATION_KERNEL)



	print diff_img
	cv2.imshow('video',fgMask)
	cv2.waitKey(100)


	success,frame = vid.read()
	
cv2.imshow('frame',background_gray)
cv2.waitKey(0)

vid.release()
cv2.destroyAllWindows()





