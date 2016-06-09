# Use this method to try and call the stitch function to stitch together all four images. (s1-s4)

# import the necessary packages
import stitcher
import argparse
import imutils
import cv2
import numpy as np
import time
 
vidcap1 = cv2.VideoCapture('../data/testVideo/output1.avi')
vidcap2 = cv2.VideoCapture('../data/testVideo/output2.avi')
vidcap3 = cv2.VideoCapture('../data/testVideo/output3.avi')
vidcap4 = cv2.VideoCapture('../data/testVideo/output4.avi')


# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
for k in range(0,5):
	success1, image1 = vidcap1.read()
	success2, image2 = vidcap2.read()
	success3, image3 = vidcap3.read()
	success4, image4 = vidcap4.read()

while ((success1 & success2) & (success3 & success4)):
 	
	#Attempt to denoise the images a bit.
	#kernel = np.ones((5,5),np.float32)/25
        #image1 = cv2.filter2D(image1,-1,kernel)
	#image2 = cv2.filter2D(image2,-1,kernel)
	#image3 = cv2.filter2D(image3,-1,kernel)
	#image4 = cv2.filter2D(image4,-1,kernel)

	#image1 = cv2.GaussianBlur(image1,(5,5),0)
	#image2 = cv2.GaussianBlur(image2,(5,5),0)
	#image3 = cv2.GaussianBlur(image3,(5,5),0)
	#image4 = cv2.GaussianBlur(image4,(5,5),0)


	# stitch the images together to create a panorama
	stitch = stitcher.Stitcher()

	total = time.time()
	im1 = np.concatenate((image1,image2),axis=1)
	im2 =np.concatenate((image3,image4),axis=1)
	im3 = np.concatenate((im1,im2),axis=0)
	cv2.imshow("Frame",im3)
	

	print "\n1:"
	(result1, vis1,H) = stitch.stitch([image1, image2], showMatches=True)
	print "\n2:"
	(result2, vis2,H) = stitch.stitch([result1, image3], showMatches=True)
	print "\n3:"
	(result3, vis3,H) = stitch.stitch([result2, image4], showMatches=True)


	elapsed = time.time() - total
	print "\n Total Time Elapsed: %f Seconds" % elapsed

	# show the images
	cv2.imshow("Result", result3)

	success1,image1 = vidcap1.read()
	success2,image2 = vidcap2.read()
	success3,image3 = vidcap3.read()
	success4,image4 = vidcap4.read()

cv2.waitKey(0)

cv2.imwrite('vidStitched.jpg',result3);
