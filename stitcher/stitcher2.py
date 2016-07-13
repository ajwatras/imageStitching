# import the necessary packages
import numpy as np
import imutils
import cv2
import time

BLEND_IMAGES = 0
EDGE_CORRECTION = 0
DILATION_KERNEL = np.ones([3,3])
EROSION_LOOPS = 1
DILATION_LOOPS = 4
EDGE_WIN_SIZE = 40
 
class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v3.X
		self.isv3 = imutils.is_cv3()
        def stitch(self, images, ratio=.75, reprojThresh=4.0,
		showMatches=True):

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
			return None
		# otherwise, apply a perspective warp to stitch the images
		# together
		t=time.time()
		(matches, H, status) = M

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

		if BLEND_IMAGES == 1:

			mask = mask1+mask2
			mask = mask+(mask == 0).astype('int')	
	
			result = np.divide(result1.astype(int)+result2.astype(int),mask).astype('uint8')
			
		else:
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
			
			return (result, vis,H,mask1,mask2)
 
		# return the stitched image
		return result

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
		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
 
			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
			if H == None:
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


