# Use this method to try and call the stitch function to stitch together all four images. (s1-s4)
# import the necessary packages
import stitcher2
import argparse
import imutils
import cv2
import numpy as np
import time
 
#vidcap1 = cv2.VideoCapture('../data/testVideo/output1.avi')
#vidcap2 = cv2.VideoCapture('../data/testVideo/output2.avi')
#vidcap3 = cv2.VideoCapture('../data/testVideo/output3.avi')
#vidcap4 = cv2.VideoCapture('../data/testVideo/output4.avi')

vidcap1 = cv2.VideoCapture('../data/vidwriter/output1.avi')
vidcap2 = cv2.VideoCapture('../data/vidwriter/output2.avi')
vidcap3 = cv2.VideoCapture('../data/vidwriter/output3.avi')
vidcap4 = cv2.VideoCapture('../data/vidwriter/output4.avi')

#vidcap1 = cv2.VideoCapture(1)
#vidcap2 = cv2.VideoCapture(2)
#vidcap3 = cv2.VideoCapture(3)
#vidcap4 = cv2.VideoCapture(4)

OUTPUT_SIZE = [1080,1920,3]



result1 = None
result2 = None
result = np.zeros(OUTPUT_SIZE).astype('uint8')


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
	stitch = stitcher2.Stitcher()
	#stitch = stitcher.Stitcher()
	

	total = time.time()
	#im1 = np.concatenate((image1,image2),axis=1)
	#im2 =np.concatenate((image3,image4),axis=1)
	#im3 = np.concatenate((im1,im2),axis=0)
	#cv2.imshow("Frame",im3)
	
	
	# Perform first stitch. Stitching together image 1 and image 2.
	print "\n1:"
	(result1, vis1,H,mask11,mask12) = stitch.stitch([image1, image2], showMatches=True)
	#H1 = H							# Store first Homography
	#seam1 = stitch.locateSeam(mask11[:,:,0],mask12[:,:,0]) 	# Locate the seam between the two images.
	
        # Uncomment to Check quality of first calibration stitch
        #cv2.imshow("result1", result1)
        #cv2.imshow("image3",image3)
        #cv2.waitKey(0)

	print "\n2:"
	(result2, vis2,H,mask21,mask22) = stitch.stitch([resul1, image3], showMatches=True)
	#H2 = H							# Store the second Homography
	#seam2 = stitch.locateSeam(mask21[:,:,0],mask22[:,:,0])	# Locate the seam between the two images.
        if (result1 is not 0) and (result2 is not 0):
            print "\n3:"
            (result3, vis3,H,mask31,mask32) = stitch.stitch([result2,image4], showMatches=True)
            #H3 = H3+H/CALWINDOW
            #H3 = H							# Store the third homography
            #seam3 = stitch.locateSeam(mask31[:,:,0],mask32[:,:,0])	# Locate the seam between the two images.


	#print "\n1:"	# Perform first stitch. Stitching together image 1 and image 2.
	print "\n1:"
	(result1, vis1,H,mask11,mask12) = stitch.stitch([image1, image2], showMatches=True)
	H1 = H							# Store first Homography
	seam1 = stitch.locateSeam(mask11[:,:,0],mask12[:,:,0]) 	# Locate the seam between the two images.
	
        # Uncomment to Check quality of first calibration stitch
        #cv2.imshow("result1", result1)
        #cv2.imshow("image3",image3)
        #cv2.waitKey(0)

	print "\n2:"
	(result2, vis2,H,mask21,mask22) = stitch.stitch([image4, image3], showMatches=True)
	H2 = H							# Store the second Homography
	seam2 = stitch.locateSeam(mask21[:,:,0],mask22[:,:,0])	# Locate the seam between the two images.
	print "\n3:"
	(result3, vis3,H,mask31,mask32) = stitch.stitch([result1,result2], showMatches=True)
	#H3 = H3+H/CALWINDOW
	H3 = H							# Store the third homography
	seam3 = stitch.locateSeam(mask31[:,:,0],mask32[:,:,0])	# Locate the seam between the two images.

	#(result1, vis1,H) = stitch.stitch([image1, image2], showMatches=True)
	#if (result1 is not None):
        #    print "\n2:"
        #    print "Result1",result1
        #    (result2, vis2,H) = stitch.stitch([result1, image3], showMatches=True)
	#if result2 is not None:
        #    print result2
        #    print "\n3:"
        #    (result3, vis3,H) = stitch.stitch([result2, image4], showMatches=True)

        if result3 is not None:
            if result3 is not 0:
                result = np.zeros(OUTPUT_SIZE).astype('uint8')
                result[0:result3.shape[0],0:result3.shape[1],:] = result3[0:result.shape[0],0:result.shape[1]].astype('uint8')
            
	elapsed = time.time() - total
	print "\n Total Time Elapsed: %f Seconds" % elapsed

	# show the images
	cv2.imshow("Result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

	success1,image1 = vidcap1.read()
	success2,image2 = vidcap2.read()
	success3,image3 = vidcap3.read()
	success4,image4 = vidcap4.read()

print "Video Stream Complete, Press any Key"
cv2.waitKey(0)

cv2.imwrite('vidStitched.jpg',result);
