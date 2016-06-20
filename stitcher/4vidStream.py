# import the necessary packages
import stitcher
import argparse
import imutils
import cv2
import numpy as np
import time
 
stitch = stitcher.Stitcher()

#vidcap1 = cv2.VideoCapture('../data/testVideo/output1.avi')
#vidcap2 = cv2.VideoCapture('../data/testVideo/output2.avi')
#vidcap3 = cv2.VideoCapture('../data/testVideo/output3.avi')
#vidcap4 = cv2.VideoCapture('../data/testVideo/output4.avi')

vidcap1 = cv2.VideoCapture(1)
vidcap2 = cv2.VideoCapture(2)
vidcap3 = cv2.VideoCapture(3)
vidcap4 = cv2.VideoCapture(4)


# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
for k in range(0,5):
	success1,image1 = vidcap1.read()
	success2,image2 = vidcap2.read()
	success3,image3 = vidcap3.read()
	success4,image4 = vidcap4.read()

while ((success1 & success2) & (success3 & success4)):
	print "success \n"
	

 
	# stitch the images together to create a panorama
	result1 = np.concatenate((image1,image2),axis=1)
	result2 =np.concatenate((image3,image4),axis=1)
	result = np.concatenate((result1,result2),axis=0)
	#descriptor = cv2.xfeatures2d.SURF_create()
	#(kps, features) = descriptor.detectAndCompute(result, None)
	
	
	#cv2.drawKeypoints(result,kps,result)
	
	# show the images
	cv2.imshow("Result", result)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
	#cv2.waitKey(0)

	success1,image1 = vidcap1.read()
	success2,image2 = vidcap2.read()
	success3,image3 = vidcap3.read()
	success4,image4 = vidcap4.read()


