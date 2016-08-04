# import the necessary packages
import numpy as np
import imutils
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Variables used to change stitcher settings. ie. Tunable Parameters
BLEND_IMAGES = 0
EDGE_CORRECTION = 0
DILATION_KERNEL = np.ones([3,3])
EROSION_LOOPS = 1
DILATION_LOOPS = 6
EDGE_WIN_SIZE = 40
SEAM_PAD = 45
SIZE_BOUNDS = [1080,1920]
SEAM_ADJ = [-SEAM_PAD,SEAM_PAD,-SEAM_PAD,SEAM_PAD]
 
class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()
        def stitch(self, images, ratio=.75, reprojThresh=4.0,
		showMatches=True, reStitching = False):
                if reStitching:
                    print "Re stitching video"
                    
		result = None
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		t=time.time()
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
		elapsed = time.time() - t
		print "Detecting keypoints: %f Seconds" % elapsed 

		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			print 'Error: No Matching Features'
			
                        if reStitching:
                            return [0,0,0,0,0,(0,0)]
                    
			return [0,0,0,0,0,(0,0)]
                    
                    
		# otherwise, apply a perspective warp to stitch the images
		# together
		t=time.time()
		(matches, H, status) = M
		if H is None: 
                    print "ERROR: no valid Homography"
                    
                    if reStitching:
                        return [0,0,0,0,0,(0,0)]
                    
                    return [0,0,0,0,0,(0,0)]

		# Detect the appropriate size for the resulting image. 
		corners = np.array([[0,0,1],[0,imageA.shape[0],1],[imageA.shape[1],0,1],
			[imageA.shape[1],imageA.shape[0],1]]).T
		
                #print H, corners
		img_bounds = np.dot(H,corners)
		
                x_bound = np.divide(img_bounds[0,:],img_bounds[2,:])
		y_bound = np.divide(img_bounds[1,:],img_bounds[2,:])
		x_shift = 0
		y_shift = 0
		if min(x_bound) < 0:
			x_shift = -int(min(x_bound))
		if min(y_bound) < 0:
			y_shift = -int(min(y_bound))
		coord_shift = np.array([y_shift,x_shift])
		shift_H = np.array([[1,0,x_shift],[0,1,y_shift],[0,0,1]])
		x_bound = int(max(max(x_bound),imageB.shape[1]))
		y_bound = int(max(max(y_bound),imageB.shape[0]))
		
		if (x_bound > SIZE_BOUNDS[0]) or (y_bound > SIZE_BOUNDS[1]):
                    print x_bound, y_bound
                    print "ERROR: Image Too Large"
                    if reStitching:
                        return [0,0,0,0,0,(0,0)]
                    return [0,0,0,0,0,(0,0)]
		
		
		# Warp Image A and place it in frame.
		imageB2 = np.pad(imageB,((y_shift,0),(x_shift,0),(0,0)),'constant',constant_values = 0)
		result2 = cv2.warpPerspective(imageA, np.dot(shift_H,H),
			(x_bound+x_shift,y_bound+y_shift))
		
		result1 = np.pad(imageB2,((0,y_bound+y_shift - imageB2.shape[0]),(0,x_bound+x_shift - imageB2.shape[1]),(0,0)),'constant', constant_values=0) 

		mask1 = (result1 > 0).astype('int')
		mask2 = (result2 > 0).astype('int')

		if BLEND_IMAGES == 1:

			mask = mask1+mask2
			mask = mask+(mask == 0).astype('int')	
	
			result = np.divide(result1.astype(int)+result2.astype(int),mask).astype('uint8')
			
		else:
                        #print result1.shape
			mask = (result1 == 0).astype('int')
			result = (result2*mask + result1).astype('uint8')
		elapsed = time.time() - t

		print "Applying transformation: %f Seconds" % elapsed
                
		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)
 
			# return a tuple of the stitched image and the
			# visualization
			if reStitching:
                                return (result, vis, H, mask1, mask2, coord_shift)
			
			return (result, vis,H,mask1,mask2,coord_shift)
 
		# return the stitched image
		return (result,H,mask1,mask2,coord_shift)

        def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
		# detect and extract features from the image
		#descriptor = cv2.xfeatures2d.SIFT_create()
		descriptor = cv2.xfeatures2d.SURF_create()
		
		(kps, features) = descriptor.detectAndCompute(image, None)
 
		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])
 
		# return a tuple of keypoints and features
		return (kps, features)

        def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		t=time.time()
		matcher = cv2.DescriptorMatcher_create("BruteForce")
		if (featuresA is None) or (featuresB is None):
                    print "Need to provide two sets of features"
                    if (featuresA is None):
                        print "FeaturesA missing"
                    if (featuresB is None):
                        print "featureB missing"
                    #cv2.waitKey(0)
                    return None
		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
		elapsed = time.time() - t
		print "Matching Feature Points: %f" % elapsed
		matches = []
 
		# loop over the raw matches
		t=time.time()
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))
                print len(featuresA)
                print len(featuresB)
                
                print len(matches), "Matches found"
		# computing a homography requires at least 4 matches
		if len(matches) > 4:
                        
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
 
			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
			if H is None:
				print "Homography failed: %d matches \n" % len(matches) 
 
			# return the matches along with the homograpy matrix
			# and status of each matched point 
			elapsed = time.time() -t
			print "Computing Homography: %f Seconds" % elapsed
			return (matches, H, status)
 
		# otherwise, no homograpy could be computed
		print "No valid Homography \n"
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
 
		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s == 1:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
 
		# return the visualization
		return vis
	def changeView(unknown,R,t, in_image):
		ti = time.time()
		im_lim = in_image.shape
		# We assume that R is a rotation matrix with each row corresponding to the direction of that axis.   
		H = np.append(R,t,1)
		# find new image bounds
		corners = np.transpose(np.array([[0,0,1,1],[im_lim[1],0,1,1],[im_lim[1],im_lim[0],1,1],[0,im_lim[0],1,1]]))
		im_bounds = np.dot(H,corners)
		max_x = round(max(np.divide(im_bounds[0,:],im_bounds[2,:])))
		max_y = round(max(np.divide(im_bounds[1,:],im_bounds[2,:])))

		output = np.zeros([max_y+1,max_x+1,3], dtype="uint8")
		# Apply transformation in 3D space
		for x in range(0,in_image.shape[1]):
			for y in range(0,in_image.shape[0]):
				in_pt = np.array([[x],[y],[1],[1]])
				out_pt = np.dot(H,in_pt)
				if out_pt[2] != 0:
					out_x = int(round(out_pt[0]/out_pt[2]))
					out_y = int(round(out_pt[1]/out_pt[2]))
					
					if (out_x >= 0) & (out_y >= 0) & (out_x < max_x+1) & (out_y< max_x + 1):
						output[out_y,out_x,:] = in_image[y,x,:]

		
		elapsed = time.time() - ti
		print " \nWarping Viewpoint: %f Seconds \n " % elapsed
		return output
	def applyHomography(self,imageB,imageA,H):
		# Detect The appropriate size for the resulting image. 
		corners = np.array([[0,0,1],[0,imageA.shape[0],1],[imageA.shape[1],0,1],
			[imageA.shape[1],imageA.shape[0],1]]).T
		
		# print H, corners
		img_bounds = np.dot(H,corners)
		
                x_bound = np.divide(img_bounds[0,:],img_bounds[2,:])
		y_bound = np.divide(img_bounds[1,:],img_bounds[2,:])
		x_shift = 0
		y_shift = 0
		if min(x_bound) < 0:
			x_shift = -int(min(x_bound))
		if min(y_bound) < 0:
			y_shift = -int(min(y_bound))
		
		shift_H = np.array([[1,0,x_shift],[0,1,y_shift],[0,0,1]])
		x_bound = int(max(max(x_bound),imageB.shape[1]))
		y_bound = int(max(max(y_bound),imageB.shape[0]))
		
		# Warp Image A and place it in frame.
		imageB2 = np.pad(imageB,((y_shift,0),(x_shift,0),(0,0)),'constant',constant_values = 0)
		result2 = cv2.warpPerspective(imageA, np.dot(shift_H,H),
			(x_bound+x_shift,y_bound+y_shift))
		
		result1 = np.pad(imageB2,((0,y_bound+y_shift - imageB2.shape[0]),(0,x_bound+x_shift - imageB2.shape[1]),(0,0)),'constant', constant_values=0) 
		
		mask1 = (result1 > 0).astype('int')
		mask2 = (result2 > 0).astype('int')

		return result1,result2,mask1,mask2

	def locateSeam(self, maskA, maskB):	
		out_mask = np.zeros(maskA.shape).astype('uint8')
		contour_copy = np.zeros(maskA.shape).astype('uint8')
		contour_copy[:] = maskA[:].astype('uint8')
		im2, contours, hierarchy = cv2.findContours(contour_copy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
		cv2.drawContours(out_mask,contours,-1,(255,255,0),1)
		out_mask = np.logical_and(out_mask,maskB).astype('float')
		return out_mask


        def reStitch(self, image1, image2,background_image,fgbg,seam):
            
                #cv2.imshow("seam",np.seam)
                tmp_result = background_image
                seam_points = np.nonzero(seam)			# Determine the point locations of the seam
                #print seam_points
                seam_bounds = [np.min(seam_points[0]) ,np.max(seam_points[0]),np.min(seam_points[1]),np.max(seam_points[1])]
                
                seam_bounds[0] = np.max([(seam_bounds[0] -SEAM_PAD),0])
                seam_bounds[2] = np.max([seam_bounds[2] -SEAM_PAD,0])
                seam_bounds[1] = np.min([seam_bounds[1] +SEAM_PAD,image1.shape[0]])
                seam_bounds[3] = np.min([seam_bounds[3] +SEAM_PAD,image1.shape[1]])

                #Foreground Re-Stitching
                fgmask = fgbg.apply(image2)			# Apply Background Subtractor
                out_frame = np.zeros(fgmask.shape)		# Generate correctly sized output_frame for foreground mask
                
                # DEBUGGING: show fgmask before dilation
                #cv2.imshow('fgmask',fgmask)
                # Denoise by erosion, then use dilation to fill in holes
                #fgmask = cv2.erode(fgmask,DILATION_KERNEL,iterations=EROSION_LOOPS)		
                #fgmask = cv2.dilate(fgmask,DILATION_KERNEL,iterations=EROSION_LOOPS+DILATION_LOOPS) 
                
                # Find bounding rectangle of all moving contours.
                im2, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

                #DEBUGGING: Show drawn contours.
                #tmp_seam = 255*np.tile(seam[...,None],[1,1,3])
                #print tmp_seam.shape
                #tmp_show = cv2.drawContours(tmp_seam.astype('uint8'),contours,-1,(0,255,0),3)
                #cv2.imshow('contours',tmp_show)

                
                x = np.zeros(len(contours))
                y = np.zeros(len(contours))
                w = np.zeros(len(contours))
                h = np.zeros(len(contours)) 
                for i in range(0,len(contours)):
                        if (i % 1 == 0):
                                cnt = contours[i]

                                x[i],y[i],w[i],h[i] = cv2.boundingRect(cnt)
                                out_frame[y[i]:y[i]+h[i],x[i]:x[i]+w[i]] = (i+1)*np.ones([h[i],w[i]])


                #cv2.imshow("Moving Objects",out_frame)
                #Create list of moving objects that cross the seam line.
                moving_objects = np.unique(out_frame*seam)
                print "Moving Objects: ", moving_objects

                # If there are objects that cross the seam line, we attempt to re-stitch those objects.
                if len(moving_objects) > 1:
                        
                        moving_objects = moving_objects[1:].astype('int')
                        #For each object that crosses the seam:
                        for i in moving_objects:
                                #DEBUGGING: output to check which objects were detected
                                print "object ",i," detected"
                                #cv2.imshow("Object Detected",(out_frame == i).astype('float'))
                                
			
                                # Print statements to help with testing.
                                #print "object %d in seam" % moving_objects[i]
                                #print x[i-1],y[i-1],w[i-1],h[i-1]
                                #cv2.imshow('image1',image1[seam_bounds[0]:seam_bounds[1],seam_bounds[2]:seam_bounds[3]])
                                #cv2.imshow('image2',image2[y[i-1]:y[i-1]+h[i-1],x[i-1]:x[i-1]+w[i-1]])
			
                                # if the object is large enough for feature point detection to function appropriately.
                                if (w[i-1] > 25) & (h[i-1] > 25):
                                    #print x[i-1],y[i-1],w[i-1],h[i-1]
                                    #image2 = cv2.rectangle(image2,(x[i-1].astype('int'),y[i-1].astype('int')),(x[i-1].astype('int')+w[i-1].astype('int'),y[i-1].astype('int')+h[i-1].astype('int')),(0,255,0),2)
                                    
                                    #Stitch together seam area with object bounding box. 
                                    r1,vis_tmp,Htemp,m1,m2,coord_shift  = self.stitch([image1[seam_bounds[0]:seam_bounds[1],seam_bounds[2]:seam_bounds[3]],image2[y[i-1]:y[i-1]+h[i-1],x[i-1]:x[i-1]+w[i-1]]],showMatches=True,reStitching=True)
                                    y_shift = coord_shift[0]
                                    x_shift = coord_shift[1]
                                    #If the stitch was successful. 
                                    if r1 is not 0:
                                        print "Re-stitched"
                                        #Show the stitched section. 
                                        #cv2.imshow('Stitched small', r1)
                                        #cv2.waitKey(0)
                                
                                        # pad the image to the right shape and coordinate system. 
                                        npad = ((y_shift,0),(x_shift,0),(0,0))
                                        tmp_result = np.pad(tmp_result,pad_width = npad, mode='constant',constant_values=0)
                                        tmp_window = r1.shape
                                        after_pad = np.array([seam_bounds[0]+tmp_window[0] - tmp_result.shape[0],seam_bounds[2]+tmp_window[1] - tmp_result.shape[1]])
                                        
                                        #cv2.imshow('tmp_result',tmp_result)
                                        #cv2.imshow('r1',r1)
                                        
                                        
                                        print tmp_window,tmp_result.shape,after_pad,seam_bounds[2],seam_bounds[0],w[i-1],h[i-1]
                                        #print "After Padding:",  after_pad
                                        after_pad = after_pad.clip(min=0)
                                        after_pad = ((0,after_pad[0].astype('int')),(0,after_pad[1].astype('int')),(0,0))
                                        tmp_result = np.pad(tmp_result, pad_width = after_pad,mode='constant', constant_values=0) 
                                        
                                        print "NPAD: ", npad, after_pad
                                        
                                
                                        tmp_mask = (r1 == 0).astype('int')
                                        #tmp_result = tmp_result[seam_bounds[0]:(seam_bounds[0]+tmp_window[0]),seam_bounds[2]:(seam_bounds[2]+tmp_window[1]),:]
                                
                                        small_result = tmp_result[seam_bounds[0]:(seam_bounds[0]+tmp_window[0]),seam_bounds[2]:(seam_bounds[2]+tmp_window[1]),:]*tmp_mask+r1
                                        
                                        tmp_result[seam_bounds[0]:(seam_bounds[0]+tmp_window[0]),seam_bounds[2]:(seam_bounds[2]+tmp_window[1]),:] = small_result
                                
                                        #background_image = np.pad(background_image,pad_width = npad,    mode='constant',constant_values=0)
                                        #background_image = np.pad(background_image,pad_width = after_pad, mode='constant',constant_values=0)
                                                                                                
                                        #if tmp_result.shape == background_image[seam_bounds[0]:(seam_bounds[0]+tmp_window[0]),seam_bounds[2]:(seam_bounds[2]+tmp_window[1]),:].shape:
                                        #    background_image[seam_bounds[0]:(seam_bounds[0]+tmp_result.shape[0]),seam_bounds[2]:(seam_bounds[2]+
                                        #tmp_result.shape[1]),:] = tmp_result
                                            
                                        #background_image = background_image[y_shift:background_image.shape[0],x_shift:background_image.shape[1]]
                                        
                                        #cv2.imshow('re-stitched', small_result.astype('uint8'))
                                        #cv2.waitKey(0)
                                        
                                        #return tmp_result, fgbg
                                            
                                            
                                        
                #cv2.imshow('bounding boxes',image2)
                return tmp_result, fgbg


