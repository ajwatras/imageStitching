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
FOV = 30
OUTPUT_SIZE = [1080,1920,3]

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
out = cv2.VideoWriter('stitched.avi',fourcc, 20.0, (OUTPUT_SIZE[1],OUTPUT_SIZE[0]))
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

## Video Sources
#vidcap1 = cv2.VideoCapture('../data/testVideo/office/output1.avi')
#vidcap2 = cv2.VideoCapture('../data/testVideo/office/output2.avi')
#vidcap3 = cv2.VideoCapture('../data/testVideo/office/output3.avi')
#vidcap4 = cv2.VideoCapture('../data/testVideo/office/output4.avi')

#vidcap1 = cv2.VideoCapture('../data/testVideo/July2/output1.avi')
#vidcap2 = cv2.VideoCapture('../data/testVideo/July2/output2.avi')
#vidcap3 = cv2.VideoCapture('../data/testVideo/July2/output3.avi')
#vidcap4 = cv2.VideoCapture('../data/testVideo/July2/output4.avi')

#vidcap1 = cv2.VideoCapture('../data/vidwriter/output1.avi')
#vidcap2 = cv2.VideoCapture('../data/vidwriter/output2.avi')
#vidcap3 = cv2.VideoCapture('../data/vidwriter/output3.avi')
#vidcap4 = cv2.VideoCapture('../data/vidwriter/output4.avi')

vidcap1 = cv2.VideoCapture(1)
vidcap2 = cv2.VideoCapture(2)
vidcap3 = cv2.VideoCapture(3)
vidcap4 = cv2.VideoCapture(4)




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
	image1 = image1[bot_edge:top_edge,left_edge:right_edge]
	image2 = image2[bot_edge:top_edge,left_edge:right_edge]
	image3 = image3[bot_edge:top_edge,left_edge:right_edge]
	image4 = image4[bot_edge:top_edge,left_edge:right_edge]

	# Remove radial distortion based on pre-calibrated camera values
	image1 = cv2.undistort(image1,mtx,radial_dst,None,mtx)
	image2 = cv2.undistort(image2,mtx,radial_dst,None,mtx)
	image3 = cv2.undistort(image3,mtx,radial_dst,None,mtx)
	image4 = cv2.undistort(image4,mtx,radial_dst,None,mtx)
	
	#flip left right
	#image1 = np.fliplr(image1)
	#image2 = np.fliplr(image2)
	#image3 = np.fliplr(image3)
	#image4 = np.fliplr(image4)
	
	# Pad the image to avoid image reshifting
	#image1 = np.pad(image1,pad_width = im_pad, mode='constant',constant_values=0)
	#image2 = np.pad(image2,pad_width = im_pad, mode='constant',constant_values=0)
	#image3 = np.pad(image3,pad_width = im_pad, mode='constant',constant_values=0)
	#image4 = np.pad(image4,pad_width = im_pad, mode='constant',constant_values=0)

	# Perform first stitch. Stitching together image 1 and image 2.
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
	
	out_pos = output_center - np.array(image1.shape[0:2])/2


#height = result3.shape[0]
#width = result3.shape[1]

# Streaming Step
cv2.imwrite('comparison.jpg',image1)
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
	
        result = np.zeros(OUTPUT_SIZE)
        print "\nA:"
	result1,result2,mask1,mask2 = stitch.applyHomography(image1,image2,H1)
	resultA = (result2*np.logical_not(mask1) + result1).astype('uint8')
	
	result_window = result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:]
        result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:] = resultA*mask2+ result_window*np.logical_not(mask2)
        
        #result,fgbg1B = stitch.reStitch(result1,result2,result,fgbg1B,seam1,[out_pos[0] - coord_shift1[0],out_pos[1]-coord_shift1[1]])

        
        print "\nB:"
	result1,result2,mask1,mask2 = stitch.applyHomography(image1,image3,H2)
	resultB = (result2*np.logical_not(mask1) + result1).astype('uint8')
	#seam2 = stitch.locateSeam(mask1[:,:,0],mask2[:,:,0])	# Locate the seam between the two images.
        
        if (out_pos[0]-coord_shift2[0] < 0):
            shift = -(out_pos[0]-coord_shift2[0])
            
            print "Shifting: ", shift
            
            result1 = result1[shift:,:,:]
            result2 = result2[shift:,:,:]
            resultB = resultB[shift:,:,:]
            mask1 = mask1[shift:,:,:]
            mask2 = mask2[shift:,:,:]
            
            coord_shift2[0] = coord_shift2[0]-shift
            
            

	print out_pos[0]-coord_shift2[0], out_pos[0]-coord_shift2[0]+result1.shape[0]
	result_window = result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:]
        result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:] =resultB*mask2 + result_window*np.logical_not(mask2)

	#result,fgbg2B = stitch.reStitch(result1,result2,result,fgbg2B,seam2,[out_pos[0] - coord_shift2[0],out_pos[1]-coord_shift2[1]])

        print "\nC:"
	result1,result2,mask1,mask2 = stitch.applyHomography(image1,image4,H3)
	result3 = (result2*np.logical_not(mask1) + result1).astype('uint8')
	#seam3 = stitch.locateSeam(mask1[:,:,0],mask2[:,:,0])	# Locate the seam between the two images.
	
        result_window = result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:]
        result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:] =result3*mask2 + result_window*np.logical_not(mask2)

	#result,fgbg3B = stitch.reStitch(result1,result2,result,fgbg3B,seam3,[out_pos[0] - coord_shift3[0],out_pos[1]-coord_shift3[1]])
        

        out_write_coord1 = [out_pos[0] - coord_shift1[0], out_pos[1] - coord_shift1[1]]
        
        #result[out_write_coord1[0]:out_write_coord1[0]+resultA.shape[0],out_write_coord1[0]:out_write_coord1[0]+resultA.shape[0],:] = resultA
        print coord_shift1

        result[out_pos[0]:out_pos[0]+image1.shape[0],out_pos[1]:out_pos[1]+image1.shape[1],:] =image1 
        result = result.astype('uint8')

	elapsed = time.time() - total
	print "\n Total Time Elapsed: %f Seconds" % elapsed


	out.write(result)
	# show the images
	cv2.imshow("Result", result)
	
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break

	success1,image1 = vidcap1.read()
	success2,image2 = vidcap2.read()
	success3,image3 = vidcap3.read()
	success4,image4 = vidcap4.read()
	
	


	if ((success1 & success2) & (success3 & success4)):
                image1 = image1[bot_edge:top_edge,left_edge:right_edge]
                image2 = image2[bot_edge:top_edge,left_edge:right_edge]
                image3 = image3[bot_edge:top_edge,left_edge:right_edge]
                image4 = image4[bot_edge:top_edge,left_edge:right_edge]
	
		image1 = cv2.undistort(image1,mtx,radial_dst,None,mtx)
		image2 = cv2.undistort(image2,mtx,radial_dst,None,mtx)
		image3 = cv2.undistort(image3,mtx,radial_dst,None,mtx)
		image4 = cv2.undistort(image4,mtx,radial_dst,None,mtx)
		
		#flip left right
                #image1 = np.fliplr(image1)
                #image2 = np.fliplr(image2)
                #image3 = np.fliplr(image3)
                #image4 = np.fliplr(image4)
		
                #image1 = np.pad(image1,pad_width = im_pad, mode='constant',constant_values=0)
                #image2 = np.pad(image2,pad_width = im_pad, mode='constant',constant_values=0)
                #image3 = np.pad(image3,pad_width = im_pad, mode='constant',constant_values=0)
                #image4 = np.pad(image4,pad_width = im_pad, mode='constant',constant_values=0)


print "Video Stream Complete, Press any Key"
cv2.waitKey(0)
cv2.imwrite('vidStitched.jpg',result);

out.release()

vidcap1.release()
vidcap2.release()
vidcap3.release()
vidcap4.release()

if DEBUGGING:
    f.release()