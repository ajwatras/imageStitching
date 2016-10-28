# Use this method to try and call the stitch function to stitch together all four images. (s1-s4)

# import the necessary packages
import stitcher
import argparse
import imutils
import cv2
import numpy as np
import time



 
# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
image1 = cv2.imread("../data/6-28-16/convex/image009.png") # top left
image2 = cv2.imread("../data/6-28-16/convex/image008.png") # top right
image3 = cv2.imread("../data/6-28-16/convex/image007.png") # bottom left
image4 = cv2.imread("../data/6-28-16/convex/image006.png") # bottom right
image5 = cv2.imread("../data/6-28-16/convex/image005.png") # bottom right
image6 = cv2.imread("../data/6-28-16/convex/image004.png") # bottom right
image7 = cv2.imread("../data/6-28-16/convex/image001.png") # bottom right
image8 = cv2.imread("../data/6-28-16/convex/image002.png") # bottom right
image9 = cv2.imread("../data/6-28-16/convex/image003.png") # bottom right
#image1 = imutils.resize(image1, width=400)
#image2 = imutils.resize(image2, width=400)
#image3 = imutils.resize(image3, width=400)
#image4 = imutils.resize(image4, width=400)
#image5 = imutils.resize(image5, width=400)
#image6 = imutils.resize(image6, width=400)
#image7 = imutils.resize(image7, width=400)
#image8 = imutils.resize(image8, width=400)
#image9 = imutils.resize(image9, width=400)


 
# stitch the images together to create a panorama
stitch = stitcher.Stitcher()

total = time.time()
print "\n1:"
(result1, vis1,H1) = stitch.stitch([image5, image1], showMatches=True)
print "\n2:"
(result2, vis2,H2) = stitch.stitch([result1, image2], showMatches=True)
print "\n3:"
(result3, vis3,H3) = stitch.stitch([result2, image3], showMatches=True)
print "\n4:"
(result4, vis4,H4) = stitch.stitch([result3, image4], showMatches=True)
print "\n5:"
(result5, vis5,H5) = stitch.stitch([result4, image6], showMatches=True)
print "\n6:"
(result6, vis6,H6) = stitch.stitch([result5, image7], showMatches=True)
print "\n7:"
(result7, vis7,H7) = stitch.stitch([result6, image8], showMatches=True)
print "\n8:"
(result8, vis8,H8) = stitch.stitch([result7, image9], showMatches=True)

elapsed = time.time() - total
print "\n Total Time Elapsed: %f Seconds" % elapsed

# show the images
cv2.imshow("Result", result8)
cv2.waitKey(0)

cv2.imwrite('9Stitched.jpg',result8);
