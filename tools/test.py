## Import libraries for test.
import numpy as np
import cv2
import time

## Constants often used in testing
radial_dst = np.array([-0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628]) # Estimated Mobius camera Distortion.
#radial_dst = [-0.37801445,  0.38328306, -0.01922729, -0.01341008, -0.60047771]
mtx = np.array([[ 444.64628787,    0.,          309.40196271],[   0.,          501.63984347,  255.86111216],[   0.,            0.,            1.        ]]) # Estimated Mobius Camera matrix

## Introduce any GLOBAL constants

## Begin test

# Testing Background subtractor.
vid = cv2.VideoCapture('../data/testVideo/office/output1.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()

success,frame = vid.read()

while success:
	fgmask = fgbg.apply(frame)
	contour = cv2.findCountours(fgmask,cv2.CV_RETR_EXTERNAL,cv2.CV_CHAIN_APPROX_NONE)

	cv2.imshow('Video Feed',fgmask)
	cv2.waitKey(40)
	success,frame = vid.read()

cv2.waitKey(0)

cv2.destroyAllWindows()
vid.release()

# testing blob detector
#vid = cv2.VideoCapture('../data/testVideo/office/output1.avi')
#detector = cv2.SimpleBlobDetector()
#
#success,frame = vid.read()
#while success:
#	keypoints = detector.detect(frame)
#
#	im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#	cv2.imshow("keypoints", im_with_keypoints)
#	cv2.waitKey(40)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()
