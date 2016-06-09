# Use this method to try and call the stitch function to stitch together all four images. (s1-s4)

# import the necessary packages
import stitcher
import argparse
import time
import imutils
import cv2
import numpy as np
 
# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
#imageA = cv2.imread("../data/stitching_img2/image002.png") # top left
#imageB = cv2.imread("../data/stitching_img2/image003.png") # top right
#imageC = cv2.imread("../data/stitching_img2/image009.png") # bottom left
#imageD = cv2.imread("../data/stitching_img2/image005.png") # bottom right
imageA = cv2.imread("../data/MeasTape/S1.jpg") # top right
imageB = cv2.imread("../data/MeasTape/S2.jpg") # top Left
imageC = cv2.imread("../data/MeasTape/S3.jpg") # bottom right
imageD = cv2.imread("../data/MeasTape/S4.jpg") # bottom left
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)
imageC = imutils.resize(imageC, width=400)
imageD = imutils.resize(imageD, width=400)


 
# stitch the images together to create a panorama
stitch = stitcher.Stitcher()
t = time.time()
print "\nAB"
(result1, vis1, H1) = stitch.stitch([imageB, imageA], showMatches=True)
print " \nCD"
(result2, vis2,H2) = stitch.stitch([imageD, imageC], showMatches=True)
print " \nAB,C"
(result3, vis3,H3) = stitch.stitch([imageC,result1], showMatches=True)
print " \nABC,D"
(result4,vis4,H4) = stitch.stitch([imageD, result3], showMatches =True)
print " \nAB,CD"
(result5,vis5,H4) = stitch.stitch([result1, result2], showMatches =True) 
print" \n"
elapsed = time.time() - t
print "Total time elapsed: %f Seconds" %elapsed

# Warp Viewpoint
t = np.array([[50],[25],[.05]])
# We can choose any orthonormal basis of r3 for x,y, and z. but by fixing y, we can easily choose x and computethe corresponding z. 
x = np.array([700,0,-1])
y = np.array([0,1,0])
z = np.cross(x,y)

x = np.divide(x,np.linalg.norm(x))
y = np.divide(y,np.linalg.norm(y))
z = np.divide(z,np.linalg.norm(z))

R = np.array([x,y,z])

result6 = stitch.changeView(R,t,result5)


# show the images
#cv2.imshow("Image A", imageA)
#cv2.imshow("Image B", imageB)
#cv2.imshow("Image C", imageC)
#cv2.imshow("Image D", imageD)
#cv2.imshow("Keypoint Matches:A,B", vis1)
#cv2.imshow("Keypoint Matches:C,D", vis2)
#cv2.imshow("Keypoint Matches:AB,C", vis3)
#cv2.imshow("Keypoint Matches:ABC,D", vis4)
#cv2.imshow("Keypoint Matches:AB,CD", vis5)
#cv2.imshow("Keypoint Matches:[A,B][C]", vis3)
#cv2.imshow("Keypoint Matches:[A,B,C][D]", vis4)
#cv2.imshow("Keypoint Matches:[A,B][C,D]", vis5)
#cv2.imshow("Result1", result1)
#cv2.imshow("Result2", result2)
#cv2.imshow("Result3", result3)
#cv2.imshow("Result4", result4)
cv2.imshow("Result5", result5)
#cv2.imshow("Result6", result6)
cv2.waitKey(0)

cv2.imwrite('4Stitched.jpg',result5);
cv2.imwrite('4Stitched-reproj.jpg',result6);
