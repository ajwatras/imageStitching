# Use this method to try and call the stitch function to stitch together all four images. (s1-s4). This 3rd iteration will implement foreground re-stitching to try and remove discontinuities near image boundaries. 

# Import the necessary packages
import stitcher2
import argparse
import imutils
import cv2
import numpy as np
import time

# Global Constants
CALWINDOW = 1
DILATION_KERNEL = np.ones([3,3])
EROSION_LOOPS = 2
DILATION_LOOPS = 4
FOV = 45

#Camera Coefficients
radial_dst = np.array([-0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628])
#radial_dst = [-0.37801445,  0.38328306, -0.01922729, -0.01341008, -0.60047771]
mtx = np.array([[ 444.64628787,    0.,          309.40196271],[   0.,          501.63984347,  255.86111216],[   0.,            0.,            1.        ]])
#mtx = [[ 457.59059782,    0.,          324.59076057],[   0.,          521.77053812,  294.85975196],[   0.,            0.,            1.,        ]]

#FOV bounds
pix2Ang = np.linspace(-45,45,480)
top_edge = (np.abs(pix2Ang-FOV)).argmin()
bot_edge = (np.abs(pix2Ang+FOV)).argmin()
print [top_edge,bot_edge]

#Initializing Needed Processes and Variables
stitch = stitcher2.Stitcher()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fgbg1A = cv2.createBackgroundSubtractorMOG2()
fgbg1B = cv2.createBackgroundSubtractorMOG2()
fgbg2A = cv2.createBackgroundSubtractorMOG2()
fgbg2B = cv2.createBackgroundSubtractorMOG2()
fgbg3A = cv2.createBackgroundSubtractorMOG2()
fgbg3B= cv2.createBackgroundSubtractorMOG2()
H1 = np.zeros([3,3])
H2 = np.zeros([3,3])
H3 = np.zeros([3,3])

# Video Sources
vidcap1 = cv2.VideoCapture('../data/testVideo/July2/output1.avi')
vidcap2 = cv2.VideoCapture('../data/testVideo/July2/output2.avi')
vidcap3 = cv2.VideoCapture('../data/testVideo/July2/output3.avi')
vidcap4 = cv2.VideoCapture('../data/testVideo/July2/output4.avi')

#vidcap1 = cv2.VideoCapture('../data/vidwriter/output1.avi')
#vidcap2 = cv2.VideoCapture('../data/vidwriter/output2.avi')
#vidcap3 = cv2.VideoCapture('../data/vidwriter/output3.avi')
#vidcap4 = cv2.VideoCapture('../data/vidwriter/output4.avi')

#vidcap1 = cv2.VideoCapture(1)
#vidcap2 = cv2.VideoCapture(2)
#vidcap3 = cv2.VideoCapture(3)
#vidcap4 = cv2.VideoCapture(4)




# Throw out first several video frames as they often contain bad data.
for k in range(0,5):
	success1, image1 = vidcap1.read()
	success2, image2 = vidcap2.read()
	success3, image3 = vidcap3.read()
	success4, image4 = vidcap4.read()

# Calibration step
for k in range(0,CALWINDOW):
	#Read incoming video feeds
	success1, image1 = vidcap1.read()
	success2, image2 = vidcap2.read()
	success3, image3 = vidcap3.read()
	success4, image4 = vidcap4.read()

	#Resize the images so that we have the correct field of view
	image1 = image1[bot_edge:top_edge,bot_edge:top_edge]
	image2 = image2[bot_edge:top_edge,bot_edge:top_edge]
	image3 = image3[bot_edge:top_edge,bot_edge:top_edge]
	image4 = image4[bot_edge:top_edge,bot_edge:top_edge]

	# Remove radial distortion based on pre-calibrated camera values
	image1 = cv2.undistort(image1,mtx,radial_dst,None,mtx)
	image2 = cv2.undistort(image2,mtx,radial_dst,None,mtx)
	image3 = cv2.undistort(image3,mtx,radial_dst,None,mtx)
	image4 = cv2.undistort(image4,mtx,radial_dst,None,mtx)

	# Perform first stitch. Stitching together image 1 and image 2.
	print "\n1:"
	(result1, vis1,H,mask11,mask12) = stitch.stitch([image1, image2], showMatches=True)
	H1 = H							# Store first Homography
	seam1 = stitch.locateSeam(mask11[:,:,0],mask12[:,:,0]) 	# Locate the seam between the two images.
	seam1_points = np.nonzero(seam1)			# Determine the point locations of the seam
	seam1_bounds = np.min(seam1_points[0]),np.max(seam1_points[0]),np.min(seam1_points[1]),np.max(seam1_points[1])
	
	print "\n2:"
	(result2, vis2,H,mask21,mask22) = stitch.stitch([result1, image3], showMatches=True)
	H2 = H							# Store the second Homography
	seam2 = stitch.locateSeam(mask21[:,:,0],mask22[:,:,0])	# Locate the seam between the two images.
	print "\n3:"
	(result3, vis3,H,mask31,mask32) = stitch.stitch([result2, image4], showMatches=True)
	#H3 = H3+H/CALWINDOW
	H3 = H							# Store the third homography
	seam3 = stitch.locateSeam(mask31[:,:,0],mask32[:,:,0])	# Locate the seam between the two images.
	height = result3.shape[0]
	width = result3.shape[1]
	out = cv2.VideoWriter('stitched.avi',fourcc, 20.0, (width,height))


# Streaming Step
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


	total = time.time()
	#im1 = np.concatenate((image1,image2),axis=1)
	#im2 =np.concatenate((image3,image4),axis=1)
	#im3 = np.concatenate((im1,im2),axis=0)
	#cv2.imshow("Frame",im3)
	#cv2.waitKey(0)
	

	result1,result2,mask1,mask2 = stitch.applyHomography(image1,image2,H1)
	resultA = (result2*np.logical_not(mask1) + result1).astype('uint8')
	#Foreground Re-Stitching
	fgmask = fgbg1B.apply(result2)			# Apply Background Subtractor
	out_frame = np.zeros(fgmask.shape)		# Generate correctly sized output_frame for foreground mask

	# Denoise by erosion, then use dilation to fill in holes
	fgmask = cv2.erode(fgmask,DILATION_KERNEL,iterations=EROSION_LOOPS)		
	fgmask = cv2.dilate(fgmask,DILATION_KERNEL,iterations=EROSION_LOOPS+DILATION_LOOPS) 

	# Find bounding rectangle of all moving contours.
	im2, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


	x = np.zeros(len(contours))
	y = np.zeros(len(contours))
	w = np.zeros(len(contours))
	h = np.zeros(len(contours)) 
	for i in range(0,len(contours)):
		if (i % 1 == 0):
			cnt = contours[i]

			x[i],y[i],w[i],h[i] = cv2.boundingRect(cnt)
			out_frame[y[i]:y[i]+h[i],x[i]:x[i]+w[i]] = (i+1)*np.ones([h[i],w[i]])

	#Create list of moving objects that cross the seam line.
	moving_objects = np.unique(out_frame*seam1)

	# If there are objects that cross the seam line, we attempt to re-stitch those objects.
	if len(moving_objects) > 1:
		
		for i in range(1,len(moving_objects)):
			
			print "object %d in seam" % moving_objects[i]
			print x,y,w,h
                        print result1[seam1_bounds[0]:seam1_bounds[1],seam1_bounds[2]:seam1_bounds[3]].shape
                        print result2[y[i-1]:y[i-1]+h[i-1],x[i-1]:x[i-1]+w[i-1]].shape
			
			if (w[i-1] > 50) & (h[i-1] > 50):
                            r1,vis_tmp,Htemp,m1,m2 = stitch.stitch([result1[seam1_bounds[0]:seam1_bounds[1],seam1_bounds[2]:seam1_bounds[3]],result2[y[i-1]:y[i-1]+h[i-1],x[i-1]:x[i-1]+w[i-1]]],showMatches=True)
                            
                            if r1 is not 0:
                                cv2.imshow('Stitched small', r1)
                                cv2.waitKey(0)
				
	
	
	

	result1,result2,mask1,mask2 = stitch.applyHomography(resultA,image3,H2)
	resultB = (result2*np.logical_not(mask1) + result1).astype('uint8')

	result1,result2,mask1,mask2 = stitch.applyHomography(resultB,image4,H3)
	result3 = (result2*np.logical_not(mask1) + result1).astype('uint8')


	elapsed = time.time() - total
	print "\n Total Time Elapsed: %f Seconds" % elapsed


	out.write(result3)
	# show the images
	cv2.imshow("Result", result3)
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break

	success1,image1 = vidcap1.read()
	success2,image2 = vidcap2.read()
	success3,image3 = vidcap3.read()
	success4,image4 = vidcap4.read()
	
	


	if ((success1 & success2) & (success3 & success4)):
                image1 = image1[bot_edge:top_edge,bot_edge:top_edge]
                image2 = image2[bot_edge:top_edge,bot_edge:top_edge]
                image3 = image3[bot_edge:top_edge,bot_edge:top_edge]
                image4 = image4[bot_edge:top_edge,bot_edge:top_edge]
	
		image1 = cv2.undistort(image1,mtx,radial_dst,None,mtx)
		image2 = cv2.undistort(image2,mtx,radial_dst,None,mtx)
		image3 = cv2.undistort(image3,mtx,radial_dst,None,mtx)
		image4 = cv2.undistort(image4,mtx,radial_dst,None,mtx)

cv2.waitKey(0)
cv2.imwrite('vidStitched.jpg',result3);

out.release()

vidcap1.release()
vidcap2.release()
vidcap3.release()
vidcap4.release()
