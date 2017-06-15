# Use this method to try and call the stitch function to stitch together all four images. (s1-s4)

# import the necessary packages
import stitcher
import argparse
import imutils
import cv2
import numpy as np
import time

stitch = stitcher.Stitcher()
CALWINDOW = 5
FOV = 45
radial_dst = np.array([-0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628])
#radial_dst = [-0.37801445,  0.38328306, -0.01922729, -0.01341008, -0.60047771]
mtx = np.array([[ 444.64628787,    0.,          309.40196271],[   0.,          501.63984347,  255.86111216],[   0.,            0.,            1.        ]])


#mtx = [[ 457.59059782,    0.,          324.59076057],[   0.,          521.77053812,  294.85975196],[   0.,            0.,            1.,        ]]

pix2Ang = np.linspace(-45,45,480)
top_edge = (np.abs(pix2Ang-FOV)).argmin()
bot_edge = (np.abs(pix2Ang+FOV)).argmin()
print [top_edge,bot_edge]



H1 = np.zeros([3,3])
H2 = np.zeros([3,3])
H3 = np.zeros([3,3])

#vidcap1 = cv2.VideoCapture('../data/testVideo/July2/output2.avi')
#vidcap2 = cv2.VideoCapture('../data/testVideo/July2/output3.avi')
#vidcap3 = cv2.VideoCapture('../data/testVideo/July2/output4.avi')
#vidcap4 = cv2.VideoCapture('../data/testVideo/July2/output1.avi')

vidcap4 = cv2.VideoCapture('../data/vidwriter/output1.avi')
vidcap3 = cv2.VideoCapture('../data/vidwriter/output2.avi')
vidcap2 = cv2.VideoCapture('../data/vidwriter/output3.avi')
vidcap1 = cv2.VideoCapture('../data/vidwriter/output4.avi')

#vidcap1 = cv2.VideoCapture(1)
#vidcap2 = cv2.VideoCapture(2)
#vidcap3 = cv2.VideoCapture(3)
#vidcap4 = cv2.VideoCapture(4)

fourcc = cv2.VideoWriter_fourcc(*'XVID')


# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
for k in range(0,5):
	success1, image1 = vidcap1.read()
	success2, image2 = vidcap2.read()
	success3, image3 = vidcap3.read()
	success4, image4 = vidcap4.read()


for k in range(0,CALWINDOW):
	success1, image1 = vidcap1.read()
	success2, image2 = vidcap2.read()
	success3, image3 = vidcap3.read()
	success4, image4 = vidcap4.read()

	image1 = image1[bot_edge:top_edge,bot_edge:top_edge]
	image2 = image2[bot_edge:top_edge,bot_edge:top_edge]
	image3 = image3[bot_edge:top_edge,bot_edge:top_edge]
	image4 = image4[bot_edge:top_edge,bot_edge:top_edge]

	image1 = cv2.undistort(image1,mtx,radial_dst,None,mtx)
	image2 = cv2.undistort(image2,mtx,radial_dst,None,mtx)
	image3 = cv2.undistort(image3,mtx,radial_dst,None,mtx)
	image4 = cv2.undistort(image4,mtx,radial_dst,None,mtx)

	print "\n1:"
	(result1, vis1,H,mask11,mask12,coord_shift1) = stitch.stitch([image1, image2], showMatches=True)
	H1 = H							# Store first Homography
	seam1 = stitch.locateSeam(mask11[:,:,0],mask12[:,:,0]) 	# Locate the seam between the two images.
	
        # Uncomment to Check quality of first calibration stitch
        #cv2.imshow("result1", result1)
        #cv2.imshow("vis1",vis1)
        #cv2.imshow("image3",image3)
        #cv2.waitKey(0)

	print "\n2:"
	(result2, vis2,H,mask21,mask22,coord_shift2) = stitch.stitch([image1, image3], showMatches=True)
	H2 = H							# Store the second Homography
	seam2 = stitch.locateSeam(mask21[:,:,0],mask22[:,:,0])	# Locate the seam between the two images.
	
	#cv2.imshow("result2",result2)
	#cv2.imshow("vis2",vis2)
	#cv2.waitKey(0)
	
	print "\n3:"
	(result3, vis3,H,mask31,mask32,coord_shift3) = stitch.stitch([image1,image4], showMatches=True)
	#H3 = H3+H/CALWINDOW
	H3 = H							# Store the third homography
        #print mask31,mask32
	seam3 = stitch.locateSeam(mask31[:,:,0],mask32[:,:,0])	# Locate the seam between the two images.
	
	#cv2.imshow("result3",result3)
	#cv2.imshow("vis3",vis3)
	#cv2.waitKey(0)

	height = result3.shape[0]
	width = result3.shape[1]
	out = cv2.VideoWriter('stitched.avi',fourcc, 20.0, (width,height))

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
	im1 = np.concatenate((image1,image2),axis=1)
	im2 =np.concatenate((image3,image4),axis=1)
	im3 = np.concatenate((im1,im2),axis=0)
	#cv2.imshow("Frame",im3)
	#cv2.waitKey(0)
	

	result1 = stitch.applyHomography(image1,image2,H1)
	result2 = stitch.applyHomography(result1,image3,H2)
	result3 = stitch.applyHomography(result2,image4,H3)
	print result3.shape


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

	image1 = image1[bot_edge:top_edge,bot_edge:top_edge]
	image2 = image2[bot_edge:top_edge,bot_edge:top_edge]
	image3 = image3[bot_edge:top_edge,bot_edge:top_edge]
	image4 = image4[bot_edge:top_edge,bot_edge:top_edge]
	if ((success1 & success2) & (success3 & success4)):
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
