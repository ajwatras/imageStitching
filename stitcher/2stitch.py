# Use this method to try and call the stitch function to stitch together all four images. (s1-s4). This 3rd iteration will implement foreground re-stitching to try and remove discontinuities near image boundaries. 

# Import the necessary packages
import stitcher2
import argparse
import imutils
import cv2
import numpy as np
import time

# Variables used to change stitcher settings. ie. Tunable Parameters
CALWINDOW = 1
DILATION_KERNEL = np.ones([3,3])
EROSION_LOOPS = 1
DEBUGGING=0
DILATION_LOOPS = 4
FOV = 45
OUTPUT_SIZE = [537,808,3]

#Camera Coefficients
radial_dst = np.array([-0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628])
#radial_dst = [-0.37801445,  0.38328306, -0.01922729, -0.01341008, -0.60047771]
mtx = np.array([[ 444.64628787,    0.,          309.40196271],[   0.,          501.63984347,  255.86111216],[   0.,            0.,            1.        ]])
#mtx = [[ 457.59059782,    0.,          324.59076057],[   0.,          521.77053812,  294.85975196],[   0.,            0.,            1.,        ]]

# Derived Parameters
#FOV bounds
pix2Ang = np.linspace(-45,45,480)
pix2Ang2 = np.linspace(-45,45,640)
top_edge = (np.abs(pix2Ang-FOV)).argmin()
bot_edge = (np.abs(pix2Ang+FOV)).argmin()
left_edge = (np.abs(pix2Ang2 + FOV)).argmin()
right_edge = (np.abs(pix2Ang2 - FOV)).argmin()
print [top_edge,bot_edge,left_edge,right_edge]


#Initializing Needed Processes and Variables
stitch = stitcher2.Stitcher()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('2stitchresult.avi',fourcc, 20.0, (OUTPUT_SIZE[1],OUTPUT_SIZE[0]))
fgbg1A = cv2.createBackgroundSubtractorMOG2()
fgbg1B = cv2.createBackgroundSubtractorMOG2()
fgbg2A = cv2.createBackgroundSubtractorMOG2()
fgbg2B = cv2.createBackgroundSubtractorMOG2()
fgbg3A = cv2.createBackgroundSubtractorMOG2()
fgbg3B= cv2.createBackgroundSubtractorMOG2()
H1 = np.zeros([3,3])
H2 = np.zeros([3,3])
H3 = np.zeros([3,3])
output_template = np.zeros(OUTPUT_SIZE)
output_center = np.array([OUTPUT_SIZE[0]/2,OUTPUT_SIZE[1]/2]).astype('int')
if DEBUGGING:
    f = open('videoStitch3_debug.txt','w')
    f.write('VIDEO STITCH 3 DEBUGGING OUTPUT \n')


im_pad = ((50,0),(50,0),(0,0))                  # Amount to pad the image so that the image doesn't get shifted

#vidcap1 = cv2.VideoCapture('../data/testVideo/chopsticker1/output1.avi')
#vidcap2 = cv2.VideoCapture('../data/testVideo/chopsticker1/output2.avi')

vidcap1 = cv2.VideoCapture('../data/testVideo/WideView-BeanDrop1/output1.avi')
vidcap2 = cv2.VideoCapture('../data/testVideo/WideView-BeanDrop1//output2.avi')

for k in range(0,5):
	success1, image1 = vidcap1.read()
	success2, image2 = vidcap2.read()

for K in range(0,CALWINDOW):
		#Read incoming video feeds
	success1, image1 = vidcap1.read()
	success2, image2 = vidcap2.read()

	#Resize the images so that we have the correct field of view
	image1 = image1[bot_edge:top_edge,left_edge:right_edge]
	image2 = image2[bot_edge:top_edge,left_edge:right_edge]

	# Remove radial distortion based on pre-calibrated camera values
	image1 = cv2.undistort(image1,mtx,radial_dst,None,mtx)
	image2 = cv2.undistort(image2,mtx,radial_dst,None,mtx)

	# Perform first stitch. Stitching together image 1 and image 2.
	print "\n1:"
	(result1, vis1,H,mask11,mask12,coord_shift1) = stitch.stitch([image1, image2], showMatches=True)
	seam1 = stitch.locateSeam(mask11[:,:,0],mask12[:,:,0]) 	# Locate the seam between the two images.
	
        # Uncomment to Check quality of first calibration stitch
        #cv2.imshow("result1", result1)
        #cv2.imshow("vis1",vis1)
        #cv2.imshow("image3",image3)
        #cv2.waitKey(0)

	out_pos = output_center - np.array(image1.shape[0:2])/2

	#print np.nonzero(seam1)
	cv2.imshow("image",result1)
	cv2.waitKey(0)


cv2.imwrite("2stitchResult.jpg",result1)
cv2.imwrite("2stitchImage1.jpg", image1)
cv2.imwrite("2stitchImage2.jpg", image2)
#height = result3.shape[0]
#width = result3.shape[1]
while (success1 & success2):
	print "\n \n";
	t_tot = time.time();
	result = np.zeros(OUTPUT_SIZE)

	#result1,result2,mask1,mask2 = stitch.applyHomography(image1,image2,H)
	#resultA = (result2*np.logical_not(mask1) + result1).astype('uint8')
	(resultA, vis1,H,mask11,mask12,coord_shift1) = stitch.stitch([image1, image2], showMatches=True)
	
	#result_window = result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:]
	#result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:] = resultA*mask2+ result_window*np.logical_not(mask2)
        
	#out.write(resultA)
	cv2.imshow("Result", resultA)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	success1,image1 = vidcap1.read()
	success2,image2 = vidcap2.read()

	if (success1 & success2):
		t_small = time.time();
		image1 = image1[bot_edge:top_edge,left_edge:right_edge]
		image2 = image2[bot_edge:top_edge,left_edge:right_edge]

		image1 = cv2.undistort(image1,mtx,radial_dst,None,mtx)
		image2 = cv2.undistort(image2,mtx,radial_dst,None,mtx)
		print "Correcting for barrel distortion: ", (time.time() - t_small);
		t_tot = time.time() - t_tot;
		print "Total computation time:", t_tot;
		print "Frames per Second: ", 1/t_tot; 


out.release()

vidcap1.release()
vidcap2.release()