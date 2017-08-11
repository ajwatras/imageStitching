import numpy as np
import cv2
import glob

#Used for calcIntrinsics
cal_frames = 25
grid_x = 9
grid_y = 6

def calcF(image1,image2, ratio=.75):
	#A method for calculating the fundamental matrix between two feature rich images. 
	#This should be performed only during calibration. 

	(kpsA, featuresA) = detectAndDescribe(image1)
	(kpsB, featuresB) = detectAndDescribe(image2)

	#matcher = cv2.DescriptorMatcher_create("BruteForce-L1")     # replaced "BruteForce" because L1 seems to give slightly better results. 
	matcher = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)

	rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
	#rawMatches = matcher.match(featuresA,featuresB)
			
	matches = []

	# loop over the raw matches
	for m in rawMatches:
		# ensure the distance is within a certain ratio of each
		# other (i.e. Lowe's ratio test)
		if len(m) == 2 and m[0].distance < m[1].distance * ratio:
			matches.append((m[0].trainIdx, m[0].queryIdx))
			print len(featuresA)
			print len(featuresB)
                
			print len(matches), "Matches found"
	# computing a homography requires at least 4 matches, we use 10 to ensure a robust stitch.
	if len(matches) > 20:
                        
		# construct the two sets of points
		ptsA = np.float32([kpsA[i] for (_, i) in matches])
		ptsB = np.float32([kpsB[i] for (i, _) in matches])
 
	(F, mask) = cv2.findFundamentalMat(ptsA,ptsB,cv2.FM_RANSAC,1)

	pts1 = ptsA[mask.ravel()==1]
	pts2 = ptsB[mask.ravel()==1]

	return F,pts1,pts2

def calcIntrinsics(images):
	#Note: images is a glob of images. We need to figure out how to make this. 

	#termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((grid_x*grid_y,3),np.float32)
	objp[:,:2] = np.mgrid[0:grid_y,0:grid_x].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3D points in real world space
	imgpoints = [] # 2D points in image plane


	for img in images:
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
		#cv2.imshow("test",gray)
		#cv2.waitKey(500)

		# Find the chess board corners
		ret, corners = cv2.findChessboardCorners(gray,(grid_y,grid_x),None)

	
		# If found, add object points, image points (after refining them)
		if ret == True:
			objpoints.append(objp)

			cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgpoints.append(corners)

			# Draw and display the corners
			cv2.drawChessboardCorners(img, (grid_y,grid_x),corners,ret)
			#cv2.imshow('img',img)
			#cv2.waitKey(500)


	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

	return dist,mtx  # outputs the radial distortion vector followed by the camera internal parameters.

def singleFrameCal(image1,image2):
		#Note: images is a glob of images. We need to figure out how to make this. 

	#termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((grid_x*grid_y,3),np.float32)
	objp[:,:2] = np.mgrid[0:grid_y,0:grid_x].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3D points in real world space
	imgpoints1 = [] # 2D points in image plane
	imgpoints2 = [] # 2D points in second camera


	for k in range(0,len(images1)):

		img1 = images1[k]
		img2 = images2[k]

		height,width,depth = img1.shape

		gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	
		#cv2.imshow("test",gray)
		#cv2.waitKey(500)

		# Find the chess board corners
		ret1, corners1 = cv2.findChessboardCorners(gray1,(grid_y,grid_x),None)
		ret2, corners2 = cv2.findChessboardCorners(gray2,(grid_y,grid_x),None)

		aret1,circles1 = cv2.findCirclesGrid(gray1,(11,4),None,cv2.CALIB_CB_ASYMMETRIC_GRID)
		aret1,circles1 = cv2.findCirclesGrid(gray1,(11,4),None,cv2.CALIB_CB_ASYMMETRIC_GRID)

		# If found, add object points, image points (after refining them)
		if ret1 == True and ret2 == True and aret1 == True and aret2 == True:
			objpoints.append(objp)

			#cv2.imshow('img1',gray1)
			#cv2.imshow('img2',gray2)
			#cv2.waitKey()

						# Draw and display the corners
			#cv2.drawChessboardCorners(img1, (grid_y,grid_x),corners1,ret1)
			#cv2.imshow('img1',img1)
			#cv2.drawChessboardCorners(img2, (grid_y,grid_x),corners2,ret2)
			#cv2.imshow('img2',img2)
			#cv2.waitKey()

			cv2.cornerSubPix(gray1,corners1,(11,11),(-1,-1),criteria)
			cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
			imgpoints1.append(corners1)
			imgpoints2.append(corners2)




	cameraMatrix1 =None
	cameraMatrix2 = None
	distCoeffs1 = None
	distCoeffs2 = None
	R =None
	T = None
	E = None
	F = None
	retval, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints,imgpoints1, imgpoints2,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width,height), R, T, E, F)
	ret, K1, dist1, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1],None,None)
	ret, K2, dist2, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1],None,None)

	#retval, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, (width, height))
	
	return dist1,dist2,K1,K2,R,T,F  # outputs the radial distortion vector followed by the camera internal parameters.


def calibCam(images1,images2):
	#Note: images is a glob of images. We need to figure out how to make this. 

	#termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((grid_x*grid_y,3),np.float32)
	objp[:,:2] = np.mgrid[0:grid_y,0:grid_x].T.reshape(-1,2)

	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3D points in real world space
	imgpoints1 = [] # 2D points in image plane
	imgpoints2 = [] # 2D points in second camera


	for k in range(0,len(images1)):

		img1 = images1[k]
		img2 = images2[k]

		height,width,depth = img1.shape

		gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	
		#cv2.imshow("test",gray)
		#cv2.waitKey(500)

		# Find the chess board corners
		ret1, corners1 = cv2.findChessboardCorners(gray1,(grid_y,grid_x),None)
		ret2, corners2 = cv2.findChessboardCorners(gray2,(grid_y,grid_x),None)

	
		# If found, add object points, image points (after refining them)
		if ret1 == True and ret2 == True:
			objpoints.append(objp)

			#cv2.imshow('img1',gray1)
			#cv2.imshow('img2',gray2)
			#cv2.waitKey()

						# Draw and display the corners
			#cv2.drawChessboardCorners(img1, (grid_y,grid_x),corners1,ret1)
			#cv2.imshow('img1',img1)
			#cv2.drawChessboardCorners(img2, (grid_y,grid_x),corners2,ret2)
			#cv2.imshow('img2',img2)
			#cv2.waitKey()

			cv2.cornerSubPix(gray1,corners1,(11,11),(-1,-1),criteria)
			cv2.cornerSubPix(gray2,corners2,(11,11),(-1,-1),criteria)
			imgpoints1.append(corners1)
			imgpoints2.append(corners2)




	cameraMatrix1 =None
	cameraMatrix2 = None
	distCoeffs1 = None
	distCoeffs2 = None
	R =None
	T = None
	E = None
	F = None
	retval, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints,imgpoints1, imgpoints2,cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,(width,height), R, T, E, F)

	#Stereo calibrate doesn't really seem like it's doing a very good job. The following lines of code are to use as little as possible from stereo calibrate. 
	ret, K1, dist1, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1],None,None)
	ret, K2, dist2, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1],None,None)

	
	
	F = computeF(K1,K2,R,T)

	#retval, K1, dist1, K2, dist2, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, (width, height))
	
	return dist1,dist2,K1,K2,R,T,F  # outputs the radial distortion vector followed by the camera internal parameters.



def compE(R,t):
	S = np.mat([[0,-t[2],-t[1]],[t[2],0,t[1]],[-t[1],t[0],0]])

	E = np.dot(R,S)

	return E


def computeF(K1,K2,R,t):
	A = np.dot(np.dot(K1,np.transpose(R)),t)
	C = np.mat([[0,-A[2],A[1]],[A[2],0,-A[0]],[-A[1],-A[0],0]])
	K = np.transpose(np.linalg.inv(K2))

	return  np.dot(K,np.dot(R,np.dot(np.transpose(K1),C)))

def computeK(image1):
	K = np.eye(3)

	return K

def detectAndDescribe(image):
#This function computes the SURF features of the image. 

	# detect and extract features from the image
	#descriptor = cv2.xfeatures2d.SIFT_create()
	descriptor = cv2.xfeatures2d.SURF_create()
		
	(kps, features) = descriptor.detectAndCompute(image, None)
                
 
	# convert the keypoints from KeyPoint objects to NumPy
	# arrays
	kps = np.float32([kp.pt for kp in kps])
	# return a tuple of keypoints and features
	return (kps, features)


def displayEpipolar(img1,img2,F,pts1,pts2):
	# Find epilines corresponding to points in right image (second image) and
	# drawing its lines on left image
	#print pts2
	lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
	lines1 = lines1.reshape(-1,3)
	img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
	# Find epilines corresponding to points in left image (first image) and
	# drawing its lines on right image
	lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
	lines2 = lines2.reshape(-1,3)
	img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
	cv2.imshow("image1",img5)
	cv2.imshow("image2",img3)

	return

def drawlines(img1,img2,lines,pts1,pts2):
	''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''

	r,c = img1.shape
	img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
	img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
	for r,pt1,pt2 in zip(lines,pts1,pts2):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
		img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
		img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
	return img1,img2

def E2F(K1,E,K2):
	F = np.transpose(np.inverse(K2))

def gatherCalFrame(caps):
	cal_images = []
	n = len(caps)
	for k in range(0,n):
		cal_images.append([])

	while(1):
	#for k in range(0,cal_frames):
		while(1):
			for j in range(0,n):
				ret,frame = caps[j].read()
				fname = 'img'+str(j)
				cv2.imshow(fname,frame)
			if cv2.waitKey(100) & 0xFF == ord('p'):
				break

		for j in range(0,n):
			ret,frame = caps[j].read()
			cal_images[j].append(frame)
			fname = 'img'+str(j)
			cv2.imshow(fname,frame)

		if cv2.waitKey(0) & 0xFF == ord('q'):
				break

	return cal_images

def poseEstimate(points1,points2):
	R = np.eye(3)
	t = np.mat([0,0,1])

	return R,t

def saveCalibration(save_var,filename = 'calibration/output.npz'):
	if len(save_var) != 5:
		print "ERROR: Use appropriate compiled calibration variable for saving" 
		#return
	
	np.savez(filename,save_var)
	print "Calibration Saved"
	return

def loadCalibration(filename):
	file = np.load(filename)
	saved_var = file['arr_0']

	return saved_var
