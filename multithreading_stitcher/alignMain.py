import numpy as np
import cv2
import stitcher2
import line_align as la


edgeThresh = 20



stitch = stitcher2.Stitcher();
calIm1 = cv2.imread("./Line_Alignment/frame00005.jpg")
calIm2 = cv2.imread("./Line_Alignment/secondframe00005.jpg")

image1 = cv2.imread("./Line_Alignment/frame00204.jpg")
image2 = cv2.imread("./Line_Alignment/secondframe204.jpg")

#Find Fundamental Matrix
F = la.calcF(calIm1,calIm2)
# Calculate Needed Homography Transformation
(result1, vis1,H,mask11,mask12,coord_shift1) = stitch.stitch([calIm1, calIm2], showMatches=True)
H1 = H	

#print image1.shape


H2 = la.lineAlign(np.mat([[77,400],[46,400]]),image1,np.mat([[580,359],[580,334]]),image2,F)
H3 = la.lineAlign(np.mat([[77,400]]),image1,np.mat([[580,359]]),image2,F)


result1,result2,mask1,mask2 = stitch.applyHomography(image1,image2,H1)
resultA = (result2*np.logical_not(mask1) + result1).astype('uint8')
result1,result2,mask1,mask2 = stitch.applyHomography(image1,image2,H2)
resultB = (result2*np.logical_not(mask1) + result1).astype('uint8')
result1,result2,mask1,mask2 = stitch.applyHomography(image1,image2,H3)
resultC = (result2*np.logical_not(mask1) + result1).astype('uint8')


cv2.imwrite("output1.jpg",resultA)
cv2.imwrite("output2.jpg",resultB)
cv2.imwrite("output3.jpg",resultC)

cv2.imshow("Result A", resultA)
cv2.imshow("Result B", resultB)
cv2.imshow("Result C", resultC)
cv2.imshow("image1", image1)
cv2.imshow("image2", image2)
cv2.waitKey(0)