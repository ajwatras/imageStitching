## Import libraries for test.
import numpy as np
import cv2
import time

## Constants often used in testing
radial_dst = np.array([-0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628]) # Estimated Mobius camera Distortion.
#radial_dst = [-0.37801445,  0.38328306, -0.01922729, -0.01341008, -0.60047771]
mtx = np.array([[ 444.64628787,    0.,          309.40196271],[   0.,          501.63984347,  255.86111216],[   0.,            0.,            1.        ]]) # Estimated Mobius Camera matrix

## Introduce any GLOBAL constants
IMAGE_SIZE = [480,640]
DILATION_KERNEL = np.ones([3,3])
EDGE_WIN_SIZE = 40

maskA = np.zeros(IMAGE_SIZE)
maskB = np.zeros(IMAGE_SIZE)
out_image = np.zeros(IMAGE_SIZE)



maskA[100:200,100:200] = np.ones([100,100])
maskB[150:250,150:250] = np.ones([100,100])


cv2.imshow("frame",maskA)
cv2.waitKey(0)

## Begin test

contour_copy = np.zeros(maskA.shape).astype('uint8')
contour_copy[:] = maskA[:]
im2, contours, hierarchy = cv2.findContours(contour_copy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(out_image,contours,-1,(255,255,0),1)
out_image = np.logical_and(out_image,maskB).astype('float')
out_image = cv2.dilate(out_image,DILATION_KERNEL,iterations=EDGE_WIN_SIZE)


cv2.imshow('original',maskA)
cv2.imshow('frame',out_image - maskB)
cv2.waitKey(0)
