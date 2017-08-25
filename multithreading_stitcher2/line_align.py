import numpy as np
import cv2

def calcF(image1,image2, ratio=.75):
	(kpsA, featuresA) = detectAndDescribe(image1)
	(kpsB, featuresB) = detectAndDescribe(image2)

	#matcher = cv2.DescriptorMatcher_create("BruteForce-L1")     # replaces "BruteForce" because L1 seems to give slightly better results.
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

	(F, mask) = cv2.findFundamentalMat(ptsA,ptsB)

	return F

def checkLine(line, n):
	rho = line[0,0]
	theta = line[0,1]
	#print "CHECKLINE: ", rho, n[0], n[1], theta
	#print (np.abs(rho - n[0]) < 5), (np.abs(theta - np.pi/2) < .001)
	if (np.abs(rho - n[1]) < 5) and (np.abs(theta) < .001):
		return False
	if (np.abs(rho - n[0]) < 5) and (np.abs(theta - np.pi/2) < .001):
		return False

	return True

def correctPoint(x, line):
#Adjust the point so that it lies on the desired line. This is used so that
#Neighborhoods only need to be approximate.
	rho = line[0]
	theta = line[1]
	y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)

	return y

def detectAndDescribe(image):

	# detect and extract features from the image
	#descriptor = cv2.xfeatures2d.SIFT_create()
	descriptor = cv2.xfeatures2d.SURF_create()

	(kps, features) = descriptor.detectAndCompute(image, None)


	# convert the keypoints from KeyPoint objects to NumPy
	# arrays
	kps = np.float32([kp.pt for kp in kps])
	# return a tuple of keypoints and features
	return (kps, features)


def drawLines(rho,theta,image,color=(0,0,255),width=2):
	#Copied code for drawing the lines on an image given rho and theta
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))

	cv2.line(image,(x1,y1),(x2,y2),color,width)

def epiMatch(point, line, F,D = 1):
	homPoint = np.mat([point[0],point[1],1]) #change point to homogeneous coordinates
	#print homPoint,F
	#Calculate epipolar lines and change formatting of both sets of lines.
	epiline = cv2.computeCorrespondEpilines(homPoint, D,F)
	A = epiline[0,0,0]
	B = epiline[0,0,1]
	C = epiline[0,0,2]
	rho = line[0]
	theta = line[1]

	#Intersect two lines
	x = (rho/np.sin(theta) +C/B)/(np.cos(theta)/np.sin(theta) - A/B)
	y1 = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)
	y2 = -A/B*x - C/B

	#Determine error in the intersection (used for debugging. )
#	err1 = A*x + B*y2 + C
#	err2 = -x*np.cos(theta) - y1*np.sin(theta) + rho
#	epiline2 = cv2.computeCorrespondEpilines(homPoint, 2,F)
#	print "POINT: ", point
#	print "EPILINE: ", epiline, epiline2
#	print "OBJECT LINE: ", line
#	print "MATCHED POINT: ", x,y1
#	print "ERROR FROM LINES: ", err1, err2

	return (int(x),int(y2))


def lineSelect(lines, image):
	n = image.shape
	#print "lineshape: ", lines.shape
	outlines = np.zeros((lines.shape[0], lines.shape[2]))
	cc = 0;
	for line in lines:
		#print "LINE: ",line, np.abs(line[0,1] - np.pi/2) > .001
		#Removing vertical and horizontal lines, this is probably not a great way of selecting lines.
		if checkLine(line,n):
			#print "Not 0: ", line
			outlines[cc,0] = line[0,0]
			outlines[cc,1] = line[0,1]
			cc = cc + 1

	outlines = outlines[0:cc]
	#print "FINAL LINE: ", outlines

	sortLines(outlines)

	return outlines

def matchLines(point1, matpoint1, point2, matpoint2):
	#Define variables use by the first point
	x1 = point1[0]
	y1 = point1[1]
	x2 = matpoint2[0]
	y2 = matpoint2[1]

	#Define variables used by the second point
	mx1 = matpoint1[0]
	my1 = matpoint1[1]
	mx2 = point2[0]
	my2 = point2[1]

	#Parameterize first line
	m1 = (y2 - y1)/(x2 - x1)
	b1 = y1 - m1*x1

	#Parameterize second line
	m2 = (my2 - my1)/(mx2 - mx1)
	b2 = my1 - m2*mx1

	#Calculate rotation required
	theta  = np.arctan2(y2 - y1, x2 - x1) - np.arctan2(my2 - my1, mx2 - mx1)
	R = [[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]

	#Calculate scaling required
	d1 = np.sqrt(pow(y2 - y1,2)) + np.sqrt(pow(x2 - x1,2))
	d2 = np.sqrt(pow(my2 - my1,2)) + np.sqrt(pow(mx2 - mx1,2))
	s = d1/d2
	S = [[s,0,0],[0,s,0],[0,0,1]]

	#Calculate translation to center rotation around origin requirement.
	t1 = [-mx1,-my1]
	tx2 = x2 - t1[0]
	ty2 = y2 - t1[1]

	T1 = [[1,0,t1[0]],[0,1,t1[1]],[0,0,1]]

	#Calculate secondary translation
	A =  np.dot(np.dot(S,R),T1)
	t2 = np.transpose([x1,y1,1] - np.dot(A,np.transpose([mx1,my1,1])))
	T2 = [[1,0,t2[0]],[0,1,t2[1]],[0,0,1]]

	#return the output transformation.
	output = np.dot(T2,A)
	return output


def sortLines(lines):
	#print "Pre-sort:", lines
	outlines = lines[lines[:,1].argsort()]
	#print "Post-sort: ", outlines


def lineAlign(points1, image1,points2, image2, F, N_size = 20, edgeThresh = 20,DRAW_LINES = False):

	if (N_size == []):
		N_size = 20
	if (edgeThresh == []):
		edgeThresh = 20
	#Function takes the following inputs:
	#points1: a 2x2 array of the form [x,y;x,y] which has the detected points in image1
	#image1: The main view destination image
	#points2: a 2x2 array of the points detected in image2
	#image2: the secondary view source image
	#F: the fundamental matrix relating image 1 and image 2.

	if (points1.shape[0] == 2):
		#Isolate point Neighborhoods
		sub11 = image1[points1[0,1] - N_size:points1[0,1]+N_size, points1[0,0] - N_size:points1[0,0]+N_size,:]
		#shift11 = np.mat([points1[0,0] - N_size, points1[0,1] - N_size])
		sub12 = image1[points1[1,1] - N_size:points1[1,1]+N_size, points1[1,0] - N_size:points1[1,0]+N_size,:]
		#shift12 = np.mat([points1[1,0] - N_size, points1[1,1] - N_size])

		sub21 = image2[points2[0,1] - N_size:points2[0,1]+N_size, points2[0,0] - N_size:points2[0,0]+N_size,:]
		#shift21 = np.mat([points2[0,0] - N_size, points2[0,1] - N_size])
		sub22 = image2[points2[1,1] - N_size:points2[1,1]+N_size, points2[1,0] - N_size:points2[1,0]+N_size,:]
		#shift22 = np.mat([points2[1,0] - N_size, points2[1,1] - N_size])

		temp = np.ones((80,80,3), np.uint8)
		temp[0:40,0:40,:] = sub11
		temp[0:40,40:80,:] = sub12
		temp[40:80,0:40,:] = sub21
		temp[40:80,40:80,:] = sub22
		temp = cv2.resize(temp,None,fx=10, fy=10, interpolation = cv2.INTER_CUBIC)
		cv2.imshow("temp", temp)
		cv2.waitKey(0)

		#Identify the coordinate shift for sub neighborhoods.
		shift11 = points1[0,:] - [N_size,N_size]
		shift12 = points1[1,:] - [N_size,N_size]
		shift21 = points2[0,:] - [N_size,N_size]
		shift22 = points2[1,:] - [N_size,N_size]


		#Identify dominant lines in sub images
		edges1 = cv2.Canny(sub11,edgeThresh,edgeThresh*3)
		lines1 = cv2.HoughLines(edges1,1,np.pi/180,N_size)


		edges2 = cv2.Canny(sub12,edgeThresh,edgeThresh*3)
		lines2 = cv2.HoughLines(edges2,1,np.pi/180,N_size)

		edges3 = cv2.Canny(sub21,edgeThresh,edgeThresh*3)
		lines3 = cv2.HoughLines(edges3,1,np.pi/180,N_size)

		edges4 = cv2.Canny(sub22,edgeThresh,edgeThresh*3)
		lines4 = cv2.HoughLines(edges4,1,np.pi/180,N_size)

	#	cv2.imshow("sub11",sub11)
	#	cv2.imshow("sub12",sub12)
	#	cv2.imshow("sub21",sub21)
	#	cv2.imshow("sub22",sub22)
	#	cv2.waitKey(0)

		#Select Lines (Each subimage should only give one line. This is just ensuring they are the right shape to avoid errors.)
		lines1 = lines1[0]
		lines2 = lines2[0]
		lines3 = lines3[0]
		lines4 = lines4[0]

		#Adjust lines for coordinate shift due to cropping.
		#np.cos(np.arctan2(y,x) - theta)*np.sqrt(x^2 + y^2) + rho
		lines1 = [np.cos(-np.arctan2(shift11[0,1],shift11[0,0]) + lines1[0,1])*np.sqrt(pow(shift11[0,0],2) + pow(shift11[0,1],2)) + lines1[0,0],lines1[0,1]]
		lines2 = [np.cos(-np.arctan2(shift12[0,1],shift12[0,0]) + lines2[0,1])*np.sqrt(pow(shift12[0,0],2) + pow(shift12[0,1],2)) + lines2[0,0],lines2[0,1]]
		lines3 = [np.cos(-np.arctan2(shift21[0,1],shift21[0,0]) + lines3[0,1])*np.sqrt(pow(shift21[0,0],2) + pow(shift21[0,1],2)) + lines3[0,0],lines3[0,1]]
		lines4 = [np.cos(-np.arctan2(shift22[0,1],shift22[0,0]) + lines4[0,1])*np.sqrt(pow(shift22[0,0],2) + pow(shift22[0,1],2)) + lines4[0,0],lines4[0,1]]


		#Ensure points lie on a line.
		points1[0,1] = correctPoint(points1[0,0],lines1)
		points1[1,1] = correctPoint(points1[1,0],lines2)
		points2[0,1] = correctPoint(points2[0,0],lines3)
		points2[1,1] = correctPoint(points2[1,0],lines4)


	#	print "LINES:",lines1,lines2,lines3,lines4
		#print points1[0][0],points1[1],points2[0],points2[1], (points1[0,0],points1[0,1])
		# Perform Epipolar Matching
		mat11 = epiMatch((points1[0,0],points1[0,1]),lines1, F)
		mat12 = epiMatch((points1[0,0],points1[0,1]),lines2, F)
		mat21 = epiMatch((points1[0,0],points1[0,1]),lines3, np.matrix.transpose(F))
		mat22 = epiMatch((points1[0,0],points1[0,1]),lines4, np.matrix.transpose(F))

		#Organize Points for findHomography
		ptsB = np.zeros((4,2))
		ptsB[0] = (points1[0,0],points1[0,1])
		ptsB[1] = (points1[1,0],points1[1,1])
		ptsB[2] = mat11
		ptsB[3] = mat12

		ptsA = np.zeros((4,2))
		ptsA[0] = mat21
		ptsA[1] = mat22
		ptsA[2] = (points2[0,0],points2[0,1])
		ptsA[3] = (points2[1,0],points2[1,1])

		if DRAW_LINES:
			drawLines(lines1[0],lines1[1],image1)
			drawLines(lines2[0],lines2[1],image1,(0,255,0))
			drawLines(lines3[0],lines3[1],image2)
			drawLines(lines4[0],lines4[1],image2,(0,255,0))

			cv2.circle(image1, mat11, 5, (255,0,0))
			cv2.circle(image2, mat21, 5, (0,255,0))
			cv2.circle(image1, mat12, 5, (0,0,255))
			cv2.circle(image2, mat22, 5, (0,255,255))

			cv2.circle(image1, (points1[0,0],points1[0,1]), 5, (0,255,0))
			cv2.circle(image2, (points2[0,0],points2[0,1]), 5, (255,0,0))
			cv2.circle(image1, (points1[1,0],points1[1,1]), 5, (0,255,255))
			cv2.circle(image2, (points2[1,0],points2[1,1]), 5, (0,0,255))

	#	print "PTS: ", ptsA,ptsB

		(LineT, status) = cv2.findHomography(ptsA, ptsB)

		return LineT
	if (points1.shape[0] == 1):
		#Isolate point Neighborhoods
		sub1 = image1[points1[0,1] - N_size:points1[0,1]+N_size, points1[0,0] - N_size:points1[0,0]+N_size,:]
		shift1 = np.mat([points1[0,0] - N_size, points1[0,1] - N_size])
		sub2 = image2[points2[0,1] - N_size:points2[0,1]+N_size, points2[0,0] - N_size:points2[0,0]+N_size,:]
		shift2 = np.mat([points2[0,0] - N_size, points2[0,1] - N_size])

		temp = np.ones((80,40,3), np.uint8)
		temp[0:40,:,:] = sub1
		temp[40:80,:,:] = sub2
		temp = cv2.resize(temp,None,fx=10, fy=10, interpolation = cv2.INTER_CUBIC)
		cv2.imshow("temp", temp)
		cv2.waitKey(0)

		#Identify the coordinate shift for sub neighborhoods.
		shift1 = points1[0,:] - [N_size,N_size]
		shift2 = points2[0,:] - [N_size,N_size]


		#Identify dominant lines in sub images
		edges1 = cv2.Canny(sub1,edgeThresh,edgeThresh*3)
		lines1 = cv2.HoughLines(edges1,1,np.pi/180,N_size)


		edges2 = cv2.Canny(sub2,edgeThresh,edgeThresh*3)
		lines2 = cv2.HoughLines(edges2,1,np.pi/180,N_size)


	#	cv2.imshow("sub11",sub11)
	#	cv2.imshow("sub12",sub12)
	#	cv2.imshow("sub21",sub21)
	#	cv2.imshow("sub22",sub22)
	#	cv2.waitKey(0)

		#Select Lines (Each subimage should only give one line. This is just ensuring they are the right shape to avoid errors.)
		lines1 = lines1[0]
		lines2 = lines2[0]


		#Adjust lines for coordinate shift due to cropping.
		#np.cos(np.arctan2(y,x) - theta)*np.sqrt(x^2 + y^2) + rho
		lines1 = [np.cos(-np.arctan2(shift1[0,1],shift1[0,0]) + lines1[0,1])*np.sqrt(pow(shift1[0,0],2) + pow(shift1[0,1],2)) + lines1[0,0],lines1[0,1]]
		lines2 = [np.cos(-np.arctan2(shift2[0,1],shift2[0,0]) + lines2[0,1])*np.sqrt(pow(shift2[0,0],2) + pow(shift2[0,1],2)) + lines2[0,0],lines2[0,1]]


		#Ensure points lie on a line.
		points1[0,1] = correctPoint(points1[0,0],lines1)
		points2[0,1] = correctPoint(points2[0,0],lines2)


	#	print "LINES:",lines1,lines2,lines3,lines4
		#print points1[0][0],points1[1],points2[0],points2[1], (points1[0,0],points1[0,1])
		# Perform Epipolar Matching
		points1 = (points1[0,0],points1[0,1])
		points2 = (points2[0,0],points2[0,1])
		mat1 = epiMatch(points1,lines2, F)
		mat2 = epiMatch(points2,lines1, np.matrix.transpose(F))

		if DRAW_LINES:
			drawLines(lines1[0],lines1[1],image1)
			drawLines(lines2[0],lines2[1],image2)

			cv2.circle(image1, mat2, 5, (255,0,0))
			cv2.circle(image2, mat1, 5, (0,255,0))
			cv2.circle(image1, points1, 5, (0,255,0))
			cv2.circle(image2, points2, 5, (255,0,0))


		LineT = matchLines(points1, mat1, points2, mat2)


		return LineT
	else:
		print "ERROR: Unexpected number of points."
