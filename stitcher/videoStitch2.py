diff --git a/data/output.avi b/data/output.avi
deleted file mode 100644
index 5797b0b..0000000
Binary files a/data/output.avi and /dev/null differ
diff --git a/data/output1.avi b/data/output1.avi
deleted file mode 100644
index 72f8a51..0000000
Binary files a/data/output1.avi and /dev/null differ
diff --git a/data/output2.avi b/data/output2.avi
deleted file mode 100644
index dcdd66f..0000000
Binary files a/data/output2.avi and /dev/null differ
diff --git a/data/output3.avi b/data/output3.avi
deleted file mode 100644
index 84e11db..0000000
Binary files a/data/output3.avi and /dev/null differ
diff --git a/data/output4.avi b/data/output4.avi
deleted file mode 100644
index 40cc38d..0000000
Binary files a/data/output4.avi and /dev/null differ
diff --git a/data/testVideo/output.avi b/data/testVideo/output.avi
new file mode 100644
index 0000000..5797b0b
Binary files /dev/null and b/data/testVideo/output.avi differ
diff --git a/data/testVideo/output1.avi b/data/testVideo/output1.avi
new file mode 100644
index 0000000..20478e9
Binary files /dev/null and b/data/testVideo/output1.avi differ
diff --git a/data/testVideo/output2.avi b/data/testVideo/output2.avi
new file mode 100644
index 0000000..92c5cbe
Binary files /dev/null and b/data/testVideo/output2.avi differ
diff --git a/data/testVideo/output3.avi b/data/testVideo/output3.avi
new file mode 100644
index 0000000..684371b
Binary files /dev/null and b/data/testVideo/output3.avi differ
diff --git a/data/testVideo/output4.avi b/data/testVideo/output4.avi
new file mode 100644
index 0000000..840c84f
Binary files /dev/null and b/data/testVideo/output4.avi differ
diff --git a/output1.avi b/output1.avi
new file mode 100644
index 0000000..d129eca
Binary files /dev/null and b/output1.avi differ
diff --git a/output2.avi b/output2.avi
new file mode 100644
index 0000000..d129eca
Binary files /dev/null and b/output2.avi differ
diff --git a/output3.avi b/output3.avi
new file mode 100644
index 0000000..d129eca
Binary files /dev/null and b/output3.avi differ
diff --git a/output4.avi b/output4.avi
new file mode 100644
index 0000000..d129eca
Binary files /dev/null and b/output4.avi differ
diff --git a/stitcher/4vidStream.py b/stitcher/4vidStream.py
index 242acf8..78fd501 100644
--- a/stitcher/4vidStream.py
+++ b/stitcher/4vidStream.py
@@ -8,10 +8,10 @@ import time
  
 stitch = stitcher.Stitcher()
 
-vidcap1 = cv2.VideoCapture('../testVideo/output1.avi')
-vidcap2 = cv2.VideoCapture('../testVideo/output2.avi')
-vidcap3 = cv2.VideoCapture('../testVideo/output3.avi')
-vidcap4 = cv2.VideoCapture('../testVideo/output4.avi')
+vidcap1 = cv2.VideoCapture('../data/testVideo/output1.avi')
+vidcap2 = cv2.VideoCapture('../data/testVideo/output2.avi')
+vidcap3 = cv2.VideoCapture('../data/testVideo/output3.avi')
+vidcap4 = cv2.VideoCapture('../data/testVideo/output4.avi')
 
 
 # load the two images and resize them to have a width of 400 pixels
@@ -23,7 +23,7 @@ for k in range(0,5):
 	success4,image4 = vidcap4.read()
 
 while ((success1 & success2) & (success3 & success4)):
-	print "success \n"
+	#print "success \n"
 	
 
  
diff --git a/stitcher/stitcher.py b/stitcher/stitcher.py
index 2e4e18f..3c53b1e 100644
--- a/stitcher/stitcher.py
+++ b/stitcher/stitcher.py
@@ -6,7 +6,8 @@ import time
  
 class Stitcher:
 	def __init__(self):
-
+		# determine if we are using OpenCV v3.X
+		self.isv3 = imutils.is_cv3()
         def stitch(self, images, ratio=.75, reprojThresh=4.0,
 		showMatches=True):
 
@@ -163,7 +164,9 @@ class Stitcher:
  
 		# return the visualization
 		return vis
-	def changeView(unknown,R,t, in_image):
+
+
+	def changeView(self,R,t, in_image):
 		ti = time.time()
 		im_lim = in_image.shape
 		# We assume that R is a rotation matrix with each row corresponding to the direction of that axis.   
@@ -191,3 +194,43 @@ class Stitcher:
 		elapsed = time.time() - ti
 		print " \nWarping Viewpoint: %f Seconds \n " % elapsed
 		return output
+
+
+	def applyHomography(self,imageB,imageA,H):
+		# Detect The appropriate size for the resulting image. 
+		corners = np.array([[0,0,1],[0,imageA.shape[0],1],[imageA.shape[1],0,1],
+			[imageA.shape[1],imageA.shape[0],1]]).T
+		
+		# print H, corners
+		img_bounds = np.dot(H,corners)
+		
+                x_bound = np.divide(img_bounds[0,:],img_bounds[2,:])
+		y_bound = np.divide(img_bounds[1,:],img_bounds[2,:])
+		x_shift = 0
+		y_shift = 0
+		if min(x_bound) < 0:
+			x_shift = -int(min(x_bound))
+		if min(y_bound) < 0:
+			y_shift = -int(min(y_bound))
+		
+		shift_H = np.array([[1,0,x_shift],[0,1,y_shift],[0,0,1]])
+		x_bound = int(max(max(x_bound),imageB.shape[1]))
+		y_bound = int(max(max(y_bound),imageB.shape[0]))
+		
+		
+		# Warp Image A and place it in frame.
+		imageB2 = np.pad(imageB,((y_shift,0),(x_shift,0),(0,0)),'constant',constant_values = 0)
+		result1 = cv2.warpPerspective(imageA, np.dot(shift_H,H),
+			(x_bound+x_shift,y_bound+y_shift))
+		
+		result2 = np.pad(imageB2,((0,y_bound+y_shift - imageB2.shape[0]),(0,x_bound+x_shift - imageB2.shape[1]),(0,0)),'constant', constant_values=0) 
+		
+		mask1 = (result1 > 0).astype('int')
+		mask2 = (result2 > 0).astype('int')
+		mask = mask1+mask2
+		mask = mask+(mask == 0).astype('int')
+			
+		result = np.divide(result1.astype(int)+result2.astype(int),mask).astype('uint8')
+
+		return result
+
diff --git a/stitcher/stitcher.pyc b/stitcher/stitcher.pyc
new file mode 100644
index 0000000..025cc45
Binary files /dev/null and b/stitcher/stitcher.pyc differ
diff --git a/stitcher/stitcher_broken.py b/stitcher/stitcher_broken.py
new file mode 100644
index 0000000..2e4e18f
--- /dev/null
+++ b/stitcher/stitcher_broken.py
@@ -0,0 +1,193 @@
+# import the necessary packages
+import numpy as np
+import imutils
+import cv2
+import time
+ 
+class Stitcher:
+	def __init__(self):
+
+        def stitch(self, images, ratio=.75, reprojThresh=4.0,
+		showMatches=True):
+
+		result = None
+		# unpack the images, then detect keypoints and extract
+		# local invariant descriptors from them
+		t=time.time()
+		(imageB, imageA) = images
+		(kpsA, featuresA) = self.detectAndDescribe(imageA)
+		(kpsB, featuresB) = self.detectAndDescribe(imageB)
+		elapsed = time.time() - t
+		print "Detecting keypoints: %f Seconds" % elapsed 
+
+		# match features between the two images
+		M = self.matchKeypoints(kpsA, kpsB,
+			featuresA, featuresB, ratio, reprojThresh)
+		# if the match is None, then there aren't enough matched
+		# keypoints to create a panorama
+		if M is None:
+			print 'Error: No Matching Features'
+			return None
+		# otherwise, apply a perspective warp to stitch the images
+		# together
+		t=time.time()
+		(matches, H, status) = M
+
+		# Detect The appropriate size for the resulting image. 
+		corners = np.array([[0,0,1],[0,imageA.shape[0],1],[imageA.shape[1],0,1],
+			[imageA.shape[1],imageA.shape[0],1]]).T
+		
+		# print H, corners
+		img_bounds = np.dot(H,corners)
+		
+                x_bound = np.divide(img_bounds[0,:],img_bounds[2,:])
+		y_bound = np.divide(img_bounds[1,:],img_bounds[2,:])
+		x_shift = 0
+		y_shift = 0
+		if min(x_bound) < 0:
+			x_shift = -int(min(x_bound))
+		if min(y_bound) < 0:
+			y_shift = -int(min(y_bound))
+		
+		shift_H = np.array([[1,0,x_shift],[0,1,y_shift],[0,0,1]])
+		x_bound = int(max(max(x_bound),imageB.shape[1]))
+		y_bound = int(max(max(y_bound),imageB.shape[0]))
+		
+		
+		# Warp Image A and place it in frame.
+		imageB2 = np.pad(imageB,((y_shift,0),(x_shift,0),(0,0)),'constant',constant_values = 0)
+		result1 = cv2.warpPerspective(imageA, np.dot(shift_H,H),
+			(x_bound+x_shift,y_bound+y_shift))
+		
+		result2 = np.pad(imageB2,((0,y_bound+y_shift - imageB2.shape[0]),(0,x_bound+x_shift - imageB2.shape[1]),(0,0)),'constant', constant_values=0) 
+		
+		mask1 = (result1 > 0).astype('int')
+		mask2 = (result2 > 0).astype('int')
+		mask = mask1+mask2
+		mask = mask+(mask == 0).astype('int')
+		
+		
+
+			
+		result = np.divide(result1.astype(int)+result2.astype(int),mask).astype('uint8')
+		elapsed = time.time() - t
+
+		print "Applying transformation: %f Seconds" % elapsed
+ 
+		# check to see if the keypoint matches should be visualized
+		if showMatches:
+			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
+				status)
+ 
+			# return a tuple of the stitched image and the
+			# visualization
+			return (result, vis,H)
+ 
+		# return the stitched image
+		return result
+
+        def detectAndDescribe(self, image):
+		# convert the image to grayscale
+		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
+ 
+		# detect and extract features from the image
+		#descriptor = cv2.xfeatures2d.SIFT_create()
+		descriptor = cv2.xfeatures2d.SURF_create()
+		
+		(kps, features) = descriptor.detectAndCompute(image, None)
+ 
+		# convert the keypoints from KeyPoint objects to NumPy
+		# arrays
+		kps = np.float32([kp.pt for kp in kps])
+ 
+		# return a tuple of keypoints and features
+		return (kps, features)
+
+        def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
+		ratio, reprojThresh):
+		# compute the raw matches and initialize the list of actual
+		# matches
+		t=time.time()
+		matcher = cv2.DescriptorMatcher_create("BruteForce")
+		rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
+		elapsed = time.time() - t
+		print "Matching Feature Points: %f" % elapsed
+		matches = []
+ 
+		# loop over the raw matches
+		t=time.time()
+		for m in rawMatches:
+			# ensure the distance is within a certain ratio of each
+			# other (i.e. Lowe's ratio test)
+			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
+				matches.append((m[0].trainIdx, m[0].queryIdx))
+		# computing a homography requires at least 4 matches
+		if len(matches) > 4:
+			# construct the two sets of points
+			ptsA = np.float32([kpsA[i] for (_, i) in matches])
+			ptsB = np.float32([kpsB[i] for (i, _) in matches])
+ 
+			# compute the homography between the two sets of points
+			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
+				reprojThresh)
+			if H == None:
+				print "Homography failed: %d matches \n" % len(matches) 
+ 
+			# return the matches along with the homograpy matrix
+			# and status of each matched point 
+			elapsed = time.time() -t
+			print "Computing Homography: %f Seconds" % elapsed
+			return (matches, H, status)
+ 
+		# otherwise, no homograpy could be computed
+		print "No valid Homography \n"
+		return None
+
+	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
+		# initialize the output visualization image
+		(hA, wA) = imageA.shape[:2]
+		(hB, wB) = imageB.shape[:2]
+		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
+		vis[0:hA, 0:wA] = imageA
+		vis[0:hB, wA:] = imageB
+ 
+		# loop over the matches
+		for ((trainIdx, queryIdx), s) in zip(matches, status):
+			# only process the match if the keypoint was successfully
+			# matched
+			if s == 1:
+				# draw the match
+				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
+				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
+				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
+ 
+		# return the visualization
+		return vis
+	def changeView(unknown,R,t, in_image):
+		ti = time.time()
+		im_lim = in_image.shape
+		# We assume that R is a rotation matrix with each row corresponding to the direction of that axis.   
+		H = np.append(R,t,1)
+		# find new image bounds
+		corners = np.transpose(np.array([[0,0,1,1],[im_lim[1],0,1,1],[im_lim[1],im_lim[0],1,1],[0,im_lim[0],1,1]]))
+		im_bounds = np.dot(H,corners)
+		max_x = round(max(np.divide(im_bounds[0,:],im_bounds[2,:])))
+		max_y = round(max(np.divide(im_bounds[1,:],im_bounds[2,:])))
+
+		output = np.zeros([max_y+1,max_x+1,3], dtype="uint8")
+		# Apply transformation in 3D space
+		for x in range(0,in_image.shape[1]):
+			for y in range(0,in_image.shape[0]):
+				in_pt = np.array([[x],[y],[1],[1]])
+				out_pt = np.dot(H,in_pt)
+				if out_pt[2] != 0:
+					out_x = int(round(out_pt[0]/out_pt[2]))
+					out_y = int(round(out_pt[1]/out_pt[2]))
+					
+					if (out_x >= 0) & (out_y >= 0) & (out_x < max_x+1) & (out_y< max_x + 1):
+						output[out_y,out_x,:] = in_image[y,x,:]
+
+		
+		elapsed = time.time() - ti
+		print " \nWarping Viewpoint: %f Seconds \n " % elapsed
+		return output
diff --git a/stitcher/vidStitched.jpg b/stitcher/vidStitched.jpg
new file mode 100644
index 0000000..6acb26c
Binary files /dev/null and b/stitcher/vidStitched.jpg differ
diff --git a/stitcher/videoStitch.py b/stitcher/videoStitch.py
index 27a76f0..98c8b36 100644
--- a/stitcher/videoStitch.py
+++ b/stitcher/videoStitch.py
@@ -7,12 +7,13 @@ import imutils
 import cv2
 import numpy as np
 import time
- 
 
-vidcap1 = cv2.VideoCapture('../testVideo/output1.avi')
-vidcap2 = cv2.VideoCapture('../testVideo/output2.avi')
-vidcap3 = cv2.VideoCapture('../testVideo/output3.avi')
-vidcap4 = cv2.VideoCapture('../testVideo/output4.avi')
+stitch = stitcher.Stitcher()
+
+vidcap1 = cv2.VideoCapture('../data/testVideo/output1.avi')
+vidcap2 = cv2.VideoCapture('../data/testVideo/output2.avi')
+vidcap3 = cv2.VideoCapture('../data/testVideo/output3.avi')
+vidcap4 = cv2.VideoCapture('../data/testVideo/output4.avi')
 
 # load the two images and resize them to have a width of 400 pixels
 # (for faster processing)
@@ -38,21 +39,21 @@ while ((success1 & success2) & (success3 & success4)):
 
 
 	# stitch the images together to create a panorama
-	stitch = stitcher.Stitcher()
+
 
 	total = time.time()
 	im1 = np.concatenate((image1,image2),axis=1)
 	im2 =np.concatenate((image3,image4),axis=1)
 	im3 = np.concatenate((im1,im2),axis=0)
-	cv2.imshow("Frame",im3)
-	cv2.waitKey(0)
+	#cv2.imshow("Frame",im3)
+	#cv2.waitKey(0)
 
 	print "\n1:"
-	(result1, vis1,H) = stitch.stitch([image1, image2], showMatches=True)
+	(result1, vis1,H1) = stitch.stitch([image1, image2], showMatches=True)
 	print "\n2:"
-	(result2, vis2,H) = stitch.stitch([result1, image3], showMatches=True)
+	(result2, vis2,H2) = stitch.stitch([result1, image3], showMatches=True)
 	print "\n3:"
-	(result3, vis3,H) = stitch.stitch([result2, image4], showMatches=True)
+	(result3, vis3,H3) = stitch.stitch([result2, image4], showMatches=True)
 
 
 	elapsed = time.time() - total
@@ -60,6 +61,8 @@ while ((success1 & success2) & (success3 & success4)):
 
 	# show the images
 	cv2.imshow("Result", result3)
+	if cv2.waitKey(1) & 0xFF == ord('q'):
+        	break
 
 	success1,image1 = vidcap1.read()
 	success2,image2 = vidcap2.read()
@@ -67,5 +70,4 @@ while ((success1 & success2) & (success3 & success4)):
 	success4,image4 = vidcap4.read()
 
 cv2.waitKey(0)
-
 cv2.imwrite('vidStitched.jpg',result3);
diff --git a/stitcher/videoStitch2.py b/stitcher/videoStitch2.py
new file mode 100644
index 0000000..52d7a87
--- /dev/null
+++ b/stitcher/videoStitch2.py
@@ -0,0 +1,93 @@
+# Use this method to try and call the stitch function to stitch together all four images. (s1-s4)
+
+# import the necessary packages
+import stitcher
+import argparse
+import imutils
+import cv2
+import numpy as np
+import time
+
+stitch = stitcher.Stitcher()
+CALWINDOW = 5
+H1 = np.zeros([3,3])
+H2 = np.zeros([3,3])
+H3 = np.zeros([3,3])
+
+vidcap1 = cv2.VideoCapture('../data/testVideo/output1.avi')
+vidcap2 = cv2.VideoCapture('../data/testVideo/output2.avi')
+vidcap3 = cv2.VideoCapture('../data/testVideo/output3.avi')
+vidcap4 = cv2.VideoCapture('../data/testVideo/output4.avi')
+
+# load the two images and resize them to have a width of 400 pixels
+# (for faster processing)
+for k in range(0,5):
+	success1, image1 = vidcap1.read()
+	success2, image2 = vidcap2.read()
+	success3, image3 = vidcap3.read()
+	success4, image4 = vidcap4.read()
+for k in range(0,CALWINDOW):
+	success1, image1 = vidcap1.read()
+	success2, image2 = vidcap2.read()
+	success3, image3 = vidcap3.read()
+	success4, image4 = vidcap4.read()
+
+	print "\n1:"
+	(result1, vis1,H) = stitch.stitch([image1, image2], showMatches=True)
+	#H1 = H1 + H/CALWINDOW
+	H1 = H
+	print "\n2:"
+	(result2, vis2,H) = stitch.stitch([result1, image3], showMatches=True)
+	#H2 = H2 + H/CALWINDOW
+	H2 = H
+	print "\n3:"
+	(result3, vis3,H) = stitch.stitch([result2, image4], showMatches=True)
+	#H3 = H3+H/CALWINDOW
+	H3 = H
+
+while ((success1 & success2) & (success3 & success4)):
+ 	
+	#Attempt to denoise the images a bit.
+	#kernel = np.ones((5,5),np.float32)/25
+        #image1 = cv2.filter2D(image1,-1,kernel)
+	#image2 = cv2.filter2D(image2,-1,kernel)
+	#image3 = cv2.filter2D(image3,-1,kernel)
+	#image4 = cv2.filter2D(image4,-1,kernel)
+
+	#image1 = cv2.GaussianBlur(image1,(5,5),0)
+	#image2 = cv2.GaussianBlur(image2,(5,5),0)
+	#image3 = cv2.GaussianBlur(image3,(5,5),0)
+	#image4 = cv2.GaussianBlur(image4,(5,5),0)
+
+
+	# stitch the images together to create a panorama
+
+
+	total = time.time()
+	im1 = np.concatenate((image1,image2),axis=1)
+	im2 =np.concatenate((image3,image4),axis=1)
+	im3 = np.concatenate((im1,im2),axis=0)
+	#cv2.imshow("Frame",im3)
+	#cv2.waitKey(0)
+	
+
+	result1 = stitch.applyHomography(image1,image2,H1)
+	result2 = stitch.applyHomography(result1,image3,H2)
+	result3 = stitch.applyHomography(result2,image4,H3)
+
+
+	elapsed = time.time() - total
+	print "\n Total Time Elapsed: %f Seconds" % elapsed
+
+	# show the images
+	cv2.imshow("Result", result3)
+	if cv2.waitKey(1) & 0xFF == ord('q'):
+        	break
+
+	success1,image1 = vidcap1.read()
+	success2,image2 = vidcap2.read()
+	success3,image3 = vidcap3.read()
+	success4,image4 = vidcap4.read()
+
+cv2.waitKey(0)
+cv2.imwrite('vidStitched.jpg',result3);
