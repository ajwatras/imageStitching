import numpy as np
import cv2
from ulti import kmean
from ulti import find_pairs

def calcF(image1,image2,label=0, ratio=.75):
	# Calculates the fundamental matrix between two cameras using matched feature points. This will give bad results if all of the detected 
	# feature points are coplanar. This script is also currently 
	if (label is 0):
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

	else:
		#_,_,_,_,F = loadnpz('calibration_7_14_17/output.npz')
		#_,_,_,_,F = loadnpz('../tools/matlab_calibration.npz')
		_,_,_,_,F = loadnpz('../tools/camera_parameters.npz')
		F = F[label]

	#elif (label is 1):
		#F = np.loadtxt('fundamentalMatrices/F1.txt',delimiter=',')
	#elif (label is 2):
		#F = np.loadtxt('fundamentalMatrices/F2.txt',delimiter=',')
#	elif (label is 3):
		#F = np.loadtxt('fundamentalMatrices/F3.txt',delimiter=',')
	
	return F

def checkLine(line, n):
# This function checks to see if the lines are vertical or horizontal.
# it is used to ensure that the image edges are not detected. but it may result in 
# the dropping of desired lines. 

	rho = line[0,0]
	theta = line[0,1]
	#print "CHECKLINE: ", rho, n[0], n[1], theta
	#print (np.abs(rho - n[0]) < 5), (np.abs(theta - np.pi/2) < .001)
	if (np.abs(rho - n[1]) < 5) and (np.abs(theta) < .001):
		return False
	if (np.abs(rho - n[0]) < 5) and (np.abs(theta - np.pi/2) < .001):
		return False

	return True

def correctPoint(x, y, line):
#Adjust the point so that it lies on the desired line. This is used so that
#Neighborhoods only need to be approximate. 
	rho = line[0]
	theta = line[1]

	if (theta == 0):
		x = rho
	else: 
		y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)
	#print x,y,rho,theta
	return x,y

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


def drawEpilines(img1, img2, linesA,linesB, pts1, pts2):
#A function for drawing the epipolar geometry onto a set of images. 

    r,c,z = img1.shape
    #r,c,z = img1.shape
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r1,r2,pt1,pt2 in zip(linesA,linesB,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())

        x0,y0 = map(int, [0, -r1[2]/r1[1] ])
        x1,y1 = map(int, [c, -(r1[2]+r1[0]*c)/r1[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)

        x0,y0 = map(int, [0, -r2[2]/r2[1] ])
        x1,y1 = map(int, [c, -(r2[2]+r2[0]*c)/r2[1] ])
        img2 = cv2.line(img2, (x0,y0), (x1,y1), color,1)
        #print r1,r2,pt1,pt2,color
        #img1 = cv2.circle(img1,(int(pt1[0]),int(pt1[1])),5,color,-1)
        #img2 = cv2.circle(img2,(int(pt2[0]),int(pt2[1])),5,color,-1)
    return img1,img2

def drawLines(rho,theta,image,color=(0,0,255),width=2):
#Copied code for drawing the lines on an image given rho and theta
#It doesn't appear to be functioning correctly, hopefully that's on the code 
# and not on the line transform, but if you run into issues with this, it might be 
# worth debugging. 

	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
    
	#print (x1,y1)
	#print (x2,y2)
	#print color
	#print width
	#print image.shape

	cv2.line(image,(x1,y1),(x2,y2),color,width)

def epiMatch(point, line, F,D = 1):
#This function uses the epipolar geometry of a scene to find the corresponding feature
#matches for a point which we know lies along a given line. 
#point - The point for which we want to find a feature match.
#line - the line on which the point will be found in the other image. 
#F - The fundamental matrix for the images
#D - the direction in which the epipolar lines should be computed. (should F get inverted.)
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
	if (theta == 0):
		x = rho
		y2 = A/B*x + C/B
	else: 
		x = (rho/np.sin(theta) +C/B)/(np.cos(theta)/np.sin(theta) - A/B)
		y1 = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)
		y2 = -A/B*x - C/B

	#Determine error in the intersection (used for debugging. )
	#err1 = A*x + B*y2 + C
	#err2 = -x*np.cos(theta) - y1*np.sin(theta) + rho
	#epiline2 = cv2.computeCorrespondEpilines(homPoint, 2,F)
	#print "POINT: ", point
	#print "EPILINE: ", epiline, epiline2
	#print "OBJECT LINE: ", line
	#print "MATCHED POINT: ", x,y1
	#print "ERROR FROM LINES: ", err1, err2

	return (int(x),int(y2))

def findOuterEdge(image,seam,line):
	# TESTING: Code for finding the intersection of LINE with the outer edge of the IMAGE.
	# The code must detect which edge of the image is the outer edge, and then perform the intersection. 

	# Calculate image dimensions
	n = image.shape

	# Check where seam intersects with edges, our seam should be a mask of where the other views are located on top of the image. 
	# edge1, x = 0

	edge1 = np.zeros(n)
	edge1[:,0] = 1
	flag = [True, True, True, True]

	flag[0] = np.any(np.logical_and(edge1,seam))

	edge2 = np.zeros(n)
	edge2[0,:] = 1
	flag[1] = np.any(np.logical_and(edge2,seam))

	edge3 = np.zeros(n)
	edge3[n[0]-1,:] = 1
	flag[2] = np.any(np.logical_and(edge3,seam))

	edge4 = np.zeros(n)
	edge4[:,n[1]-1] = 1
	flag[3] = np.any(np.logical_and(edge4,seam))

	# Save the edge lines in Hough Line transform format. 
	edges = np.mat([[np.pi/2,0],[0,0],[0,n[0]],[np.pi/2,n[1]]])


	# intersect line with edge, 
	point = np.zeros([4,2])
	for i in range(0,4):
		if not flag[i]:
			#If a horizontal edge
			if edges[i,0] is 0:
				point[i,1] = np.int(edges[i,1])
				point[i,0] = np.int(-np.tan(line[0])*point[i,1] + line[1]/np.cos(line[0]))
			# If a vertical edge 
			else:
				point[i,0] = np.int(edges[i,1])
				point[i,1] = np.int(-point[i,0]/np.tan(line[0]) + line[1]/np.sin(line[0]))

			# Check if point is in image.
			if (point[i,0] <= n[0]) and (point[i,1] <= n[1]) and (point[i,0] >= 0) and (point[i,1] >= 0):
				return point[i,:]

	
	# return edge point
	return point[0,:]
def genBorderMasks(main_frame,side_frame, main_mask,side_mask,H,edgeThresh = 1):
	# generate initial masks
	side_border = np.zeros(side_frame.shape)
	side_border[0,:,:] = 1
	side_border[:,0,:] = 1
	side_border[side_frame.shape[0]-1,:,:] = 1
	side_border[:,side_frame.shape[1]-1,:] = 1


	main_seam = main_mask * side_mask
	side_seam = mapCoveredSide(H,main_frame,side_frame)
	transformed_side_border = cv2.warpPerspective(side_border,H,(side_mask.shape[1],side_mask.shape[0]))

	return main_seam, side_seam, side_border, transformed_side_border

def genObjMask(idx,main_view_frame, main_view_background, side_view_frame,side_view_background, crossing_edges_main_view_list, crossing_edges_side_views_list, homography_list ):
	# genObjMask detects a moving object which crosses the seam line between the main view and the side view. It returns a sub image containing the object,
	# and a point declaring the location of the 0.0 pixel of the mask in the original image.

    main_view_object_mask = identifyFG(main_view_frame, main_view_background).astype('uint8')
    main_view_edge = cv2.Canny(main_view_object_mask, 50, 150)
    main_view_object_mask = (main_view_object_mask/255).astype('uint8')
    kernel_gradient = np.mat([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    main_view_edge = cv2.filter2D(main_view_object_mask,-1,kernel_gradient)
    main_view_edge = (main_view_edge > 0).astype('uint8')

    side_view_object_mask = identifyFG(side_view_frame, side_view_background).astype('uint8')
    side_view_edge = cv2.Canny(side_view_object_mask,50,150)
    side_view_object_mask = (side_view_object_mask/255).astype('uint8')
    side_view_edge = cv2.filter2D(side_view_object_mask,-1,kernel_gradient)
    side_view_edge = (side_view_edge > 0).astype('uint8')

    # DISPLAY DETECTED OBJECTS
    #cv2.imshow("Side View Object"+str(idx),side_view_object_mask*255)
    #cv2.imshow("side View mask"+str(idx), np.logical_or(self.crossing_edges_side_views_list[idx], side_view_object_mask).astype('uint8')*255 ) 
    #cv2.imwrite("side_view_object"+str(idx)+".jpg",side_view_object_mask*255)
    #cv2.imwrite("side_view_mask"+str(idx)+".jpg",np.logical_or(self.crossing_edges_side_views_list[idx], side_view_object_mask).astype('uint8')*255 )
    #cv2.waitKey(10)

    pts = np.nonzero(main_view_edge * crossing_edges_main_view_list[idx])
    pts = np.asarray(pts)

    if pts.shape[1] >= 2:
            print "Object detected in main view"
            clusters_main_view, cluster_err = kmean(pts, 2)
            #if cluster_err <= 10:
            if True:    
                side_view_object_mask = identifyFG(side_view_frame, side_view_background).astype('uint8')
                side_view_edge = cv2.Canny(side_view_object_mask,50,150)
                side_view_object_mask = (side_view_object_mask/255).astype('uint8')
                side_view_edge = cv2.filter2D(side_view_object_mask,-1,kernel_gradient)
                side_view_edge = (side_view_edge > 0).astype('uint8')  
                
                #cv2.imshow("side View mask"+str(idx), side_view_object_mask*255 )              
                #cv2.imshow("Side View Frame " +str(idx), side_view_frame)
                #cv2.waitKey(10)
                #print np.nonzero(side_view_edge)

                pts = np.nonzero(side_view_edge * crossing_edges_side_views_list[idx])
                pts = np.asarray(pts)
                if pts.shape[1] >= 2:
                    print "Object Detected in side view"
                    clusters_side_view, cluster_err = kmean(pts, 2)
                    #if cluster_err <= 10:
                    if True:
                        dist_main = np.sum(np.square(clusters_main_view[:, 0] - clusters_main_view[:, 1]))
                        dist_side = np.sum(np.square(clusters_side_view[:, 0] - clusters_side_view[:, 1]))
                        print dist_main, dist_side
                        if True:
                        #if dist_main >= 100 and dist_main <= 5000 and dist_side >= 100 and dist_side <= 5000:

                            clusters_side_view = find_pairs(clusters_main_view, clusters_side_view, homography_list[idx])

                            ##############################
                            pts1 = np.mat([[clusters_main_view[1,0], clusters_main_view[0,0]], [clusters_main_view[1,1], clusters_main_view[0,1]]])
                            pts2 = np.mat([[clusters_side_view[1,0], clusters_side_view[0,0]], [clusters_side_view[1,1], clusters_side_view[0,1]]])
                            pts1 = np.mat([pts1[0,0], pts1[0,1]])
                            pts2 = np.mat([pts2[0,0], pts2[0,1]])

                            return True,pts1,pts2,main_view_object_mask,side_view_object_mask



    return False, [], [], [], []


def identifyFG(frame, model,thresh = 40):
	#Uses the model from modelBackground to separate foreground objects from background objects
	# 
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)

	diff_frame = np.abs(frame.astype('float') - model.astype('float'))
	binary = cv2.threshold(np.uint8(diff_frame),thresh,255,cv2.THRESH_BINARY)
	return binary[1]

def loadnpz(filename):
	#Loads a .npz file containing calibration information. 
	file_array = np.load(filename)
	(dst,K,R,t,F) = file_array['arr_0']

	return dst,K,R,t,F

def lineEdge(line, frame_mask, border_mask):
	# INDEVELOPMENT: lineEdge finds intersections between line and frame_edge. 
	# line=(rho,theta) stores the line information 
	# frame_mask shows a mask of the frame, with a mask denoting where the other image will be located

	if (len(frame_mask.shape) == 2):
		frame_mask = np.repeat(frame_mask[:,:,np.newaxis],3,axis=2)
	if (len(border_mask.shape) == 2):
		border_mask = np.repeat(border_mask[:,:,np.newaxis],3,axis=2)

	line_mask = np.zeros(frame_mask.shape)
	drawLines(line[0],line[1],line_mask,(1,1,1),1)

	possible_points = np.nonzero(line_mask[:,:,0]*border_mask[:,:,0]*(1-frame_mask[:,:,0]))

	if(len(possible_points[0]) > 1):
		print "Warning: line intersects with too many edges in lineEdge. Total number of edges ",len(possible_points[0])

	elif (len(possible_points[0]) == 0):
		print "Warning: no point found."
		return []

	print"Possible points: ", possible_points
	point = (possible_points[1][0],possible_points[0][0])
	temp_display = np.zeros(line_mask.shape)
	temp_display[:,:,0] = line_mask[:,:,0]
	temp_display[:,:,1] = border_mask[:,:,0]
	temp_display[:,:,2] = (1- frame_mask[:,:,0])
	
	cv2.imshow("temp display",temp_display)
	cv2.waitKey(0)
	cv2	.destroyWindow("temp display")
	return point


def lineSelect(image, edgeThresh, N_size):
# Computes the line corresponding to the most prominent edge in the image.

	n = image.shape
	outlines = (0,0)

	edges = cv2.Canny(image,edgeThresh,edgeThresh*3)

	lines = cv2.HoughLines(edges[1:edges.shape[0]-1,1:edges.shape[1]-1],1,np.pi/180,N_size/2)

	temp_image = np.repeat(edges[:, :, np.newaxis]/2, 3, axis=2)
	temp_image = 100*temp_image.astype('uint8')
	
	for line in lines:
		print line[0,0], line[0,1]
		drawLines(line[0,0],line[0,1],temp_image)

	cc = 0;
	outlines = []
	for line in lines:
		#print "LINE: ",line, np.abs(line[0,1] - np.pi/2) > .001
		#Removing vertical and horizontal lines, this is probably not a great way of selecting lines. 
		if checkLine(line,n):
			print "Not 0: ", cc,line[0,0], line[0,1]
			outlines.append([line[0,0],line[0,1]])
			cc = cc + 1

	outlines = sortLines(outlines, edges)

	# Grab line crossing points

	# Identify 2 dominant line crossings
	# Check for secondary line crossings
	#  
	print "FINAL LINE: ", outlines

	return outlines


def matchLines(point1, matpoint1, point2, matpoint2):
# 

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

def mapCoveredSide(H, main_view, side_view):
		main_view_mask = np.ones(main_view.shape)
		backwards_H = np.linalg.inv(H)

		transformed_mask = cv2.warpPerspective(main_view_mask,backwards_H,(side_view.shape[1],side_view.shape[0]))

		return transformed_mask

def modelBackground(caps, CAL_LENGTH = 10):
	# This function runs the image captures until the user inputs p. Then it uses the next
	# CAL_LENGTH frames to generate a model for the background. 

	frames = []
	output = []
	n = len(caps)


	for k in range(0,n):
		frames.append([])
		output.append([])	
		ret, frames[k] = caps[k].read()

	print "Press P to start esimating background, press q to quit."
	while ret:
		for k in range(0,n):
			ret,frames[k] = caps[k].read()
			cv2.imshow("frame " + str(k), frames[k])

		if cv2.waitKey(10) == ord('q'):
			return False, []

		if cv2.waitKey(10) == ord('p'):
			cv2.destroyWindow("frame 0")
			cv2.destroyWindow("frame 1")
			cv2.destroyWindow("frame 2")
			cv2.destroyWindow("frame 3")
			cv2.waitKey(1)

			break

	for k in range(0,n):
		output[k] = frames[k]/CAL_LENGTH
		for m in range(1,CAL_LENGTH):
			ret, frames[k] = caps[k].read()
			output[k] = output[k] + frames[k]/CAL_LENGTH


		output[k] = output[k].astype('uint8')
	return True, output

def setRight(line):
	# Shifts the point to the right to avoid overlapping features.
	x = 400
	y = 400
	rho = line[0]
	theta = line[1]

	if (theta == 0):
		x = rho
	else: 
		y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)

	return x,y
def setLeft(line):
	# Shifts the point to the right to avoid overlapping features.

	x = 5
	y = 5
	rho = line[0]
	theta = line[1]

	if (theta == 0):
		x = rho
	else: 
		y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)

	return x,y

def shiftRight(x,y,line,dist = 5):
	# Shifts the point to the right to avoid overlapping features.
	x = x+dist
	y = y + dist
	rho = line[0]
	theta = line[1]

	if (theta == 0):
		x = rho
	else: 
		y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)

	return x,y
def shiftLeft(x,y,line,dist = 5):
	# Shifts the point to the right to avoid overlapping features.

	x = x - dist
	y = y - dist
	rho = line[0]
	theta = line[1]

	if (theta == 0):
		x = rho
	else: 
		y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)

	return x,y


def sortLines(lines, edge):
# Sort the lines by how well they cover the edge. Still needs to bwe adjusted for multi-edge
	#print "Pre-sort:", lines
	H = np.eye(3)
	lines = np.mat(lines)

	# Transform the lines based on H

	# Evaluate for the seam 

	coverage = []
	for line in lines:
		print line
		temp= np.zeros((edge.shape[0], edge.shape[1],3)).astype('uint8')
		drawLines(line[0,0],line[0,1],temp)
		temp = np.nonzero(temp[:,:,0]*edge)
		coverage.append(len(temp[0]))
		print temp,coverage

	#idx = coverage.index(max(coverage))
	idx = np.argsort(-np.mat(coverage))
	np.array(lines)[idx]
	outlines = lines
	#outlines = lines[idx]	
	#outlines = lines[lines[:,1].argsort()]
	print "Post-sort: ", outlines



	return outlines

def surgSeg(image, model,N_size = 20, edgeThresh = 20): 
	# IN DEVELOPMENT: Code to detect the location of  a surgical tool and generate a mask to cover it. 

	# generate a blank mask of the correct shape
	mask = np.uint8(np.zeros(image.shape))

	# find the difference image between the current frame and the background model. 
	diff_img = np.uint8(abs(np.int16(image) - np.int16(model)))

	# Apply canny edge detector to the difference image
	edges = cv2.Canny(diff_img,edgeThresh,edgeThresh*3)	
	#cv2.imshow('edges', edges)
	#cv2.waitKey(0)
	#cv2.destroyWindow("edges")

	# Apply hough line transform to detect dominant edge lines
	#lines = cv2.HoughLines(edges,1,np.pi/180,N_size*2)
	lines = cv2.HoughLinesP(edges,rho = 1,theta = np.pi/180, threshold = 2, minLineLength = 5)



	# If DRAW_LINES, then we should draw our detected lines on the image (used for testing)
	lines2 = np.zeros((len(lines),1,2))
	if True and lines.any() != None:
		tmp_image = image
		for i in range(0, len(lines)):
			
			#drawLines(lines[i,0,0],lines[i,0,1], tmp_image,width = 1)

			lines2[i,0,1] = np.arctan2(lines[i,0,3] - lines[i,0,1], lines[i,0,2] - lines[i,0,0])
			lines2[i,0,0] = lines[i,0,1]*np.cos(lines2[i,0,1]) + lines[i,0,0]*np.sin(lines2[i,0,1])
			drawLines(lines2[i,0,0],lines2[i,0,1], tmp_image,width = 1)


			cv2.line(tmp_image, (lines[i,0,0],lines[i,0,1]),(lines[i,0,2],lines[i,0,3]),[0,255,0],2)


		
		#print lines
		#print lines2
		#cv2.imshow('surg_mask', tmp_image)
		#cv2.waitKey(0)
		lines = lines2
	# Bound the surgical tool with a set of bounding half planes
	lineBound(mask,lines)

	# Calculate mask using intersection of half planes.

	return mask

def warpObject(main_view_frame, side_view_frame, side_view_object_mask, tempH, main_homography, sti, result1, mask1, result2, shift, new_mask, trans_matrix):
	

    corners = np.mat([[0, side_view_frame.shape[1], 0, side_view_frame.shape[1]], [0, 0, side_view_frame.shape[0], side_view_frame.shape[0]], [1, 1, 1, 1]])
    transformed_corners = np.dot(tempH, corners)
    bound = np.zeros((2, 4), np.float)
    bound[0,0] = transformed_corners[0,0] / transformed_corners[2,0]
    bound[1,0] = transformed_corners[1,0] / transformed_corners[2,0]
    bound[0,1] = transformed_corners[0,1] / transformed_corners[2,1]
    bound[1,1] = transformed_corners[1,1] / transformed_corners[2,1]
    bound[0,2] = transformed_corners[0,2] / transformed_corners[2,2]
    bound[1,2] = transformed_corners[1,2] / transformed_corners[2,2]
    bound[0,3] = transformed_corners[0,3] / transformed_corners[2,3]
    bound[1,3] = transformed_corners[1,3] / transformed_corners[2,3]

    if (np.max(bound[0,:]) - np.min(bound[0,:])) <= 2500 and (np.max(bound[1,:]) - np.min(bound[1,:])) < 2500 and (tempH is not np.zeros((3,3))):
        #############################################################################
        result1_temp,result2_temp,_,mask2_temp, shift_temp, _ = sti.applyHomography(main_view_frame, side_view_frame, tempH)
        mask_temp_temp = (result1_temp == 0).astype('int')
        result2_temp_shape = result2_temp.shape

        OUTPUT_SIZE = [3000,3000,3]
        out_pos = np.array([OUTPUT_SIZE[0]/2-500,OUTPUT_SIZE[1]/2-500]).astype('int')
        pano = np.zeros(OUTPUT_SIZE, np.uint8)

        window = pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp.shape[1],:]

        if window.shape[0] != mask2_temp.shape[0] or window.shape[1] != mask2_temp.shape[1]:
            return result1,result2,mask1,new_mask, shift, trans_matrix

        print out_pos, shift_temp, result1.shape

        pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp.shape[1],:] = 0+result2_temp
        result2_temp = 0+pano[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]
        #cv2.imshow('pano_2', result2_temp)
        #cv2.waitKey(0)
        _,mask,_,_, _, _ = sti.applyHomography(main_view_frame, np.stack((side_view_object_mask,side_view_object_mask,side_view_object_mask), axis=2), tempH)

        pano_mask = np.zeros(OUTPUT_SIZE, np.uint8)
        pano_mask[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp_shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp_shape[1],:] = 0+mask
        mask = 0+pano_mask[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]

        pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+mask2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+mask2_temp.shape[1],:] = 0+mask2_temp
        mask2_temp = 0+pano[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]


        mask = mask.astype('uint8') * mask2_temp
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening_closing)
        mask = mask * (result2 > 0).astype('uint8')
        #cv2.imshow('mask',mask*255)

        _,mask_object_transformed,_,_, _, _ = sti.applyHomography(main_view_frame, np.stack((side_view_object_mask,side_view_object_mask,side_view_object_mask), axis=2), main_homography)
        #cv2.imshow('mask_object_transformed',mask_object_transformed*255)
        Background_mask = new_mask * (mask_object_transformed == 0).astype('uint8')
        #cv2.imshow('Background_mask',Background_mask.astype('uint8')*255)
        color1 = np.sum(result2[:,:,0] * Background_mask[:,:,0]) / np.sum(Background_mask[:,:,0])
        color2 = np.sum(result2[:,:,1] * Background_mask[:,:,1]) / np.sum(Background_mask[:,:,1])
        color3 = np.sum(result2[:,:,2] * Background_mask[:,:,2]) / np.sum(Background_mask[:,:,2])
        temp = np.ones(mask.shape, np.uint8)
        temp[:,:,0] = color1
        temp[:,:,1] = color2
        temp[:,:,2] = color3

        result2 = Background_mask.astype('uint8') * result2 + (result2 > 0).astype('uint8') * mask_object_transformed * temp
        #cv2.imshow('result2',result2)
        result2 = result2 * np.logical_not(mask) + result2_temp * mask
        #cv2.imshow('result2_2',result2)
        #cv2.imshow('result2 * np.logical_not(mask)',result2 * np.logical_not(mask))
        #cv2.waitKey(0)
        #mask_temp_temp = (result1 == 0).astype('int')
        #temp = (result2*mask_temp_temp + result1).astype('uint8')
        #############################################################################
        return result1,result2,mask1,new_mask, shift, trans_matrix


    return result1,[],[],[],[],[]


def lineAlign(points1, image1,points2, image2, F, main_seam, side_seam, side_border,transformed_side_border,shift, N_size = 40, edgeThresh = 50,DRAW_LINES = True):
	# lineAlign performs epipolar line alignment on the detected object. 
	# the Function takes the following inputs:
	# points1: a 1x2 or 2x2 matrix of the form [x,y] which denotes the approximate location of the intersection of
	# the edges of the object with the edges of image1
	# image1:  the main view image, this image should not be transformed.
	# points2: a 1x2 or 2x2 matrix of the form [x,y] which denotes the approximate location of the intersection of
	# the edges of the object with the edges of image2
	# image2:  the side view image which will be transformed.
	# F: the fundamental matrix relating image 1 to image 2.
	# main_seam: the mask for the edge of image 2 as seen in image 1
	# side_seam: the mas, for the edge of image 1 as seen in image 2
	# side_border: A mask for the outer edge of image 2 prior to transformation
	# transformed_side_border:  A mask for the outer edge of image 2 after transformtation
	# shift: The shift applied to the image 1 when the Homography is applied to image 2.
	# N_size: the size bounding box we should use to detect the dominant line associated with each object edge.
	# Too small of a bounding box will lead to a poor estimate of the line, but too large a bounding box may lead
	# to incorrect edges being detected. 
	# edgeThresh: the threshold value for the edge detector. 
	# DRAW_LINES: a flag used for Visualizing the line alignment process. 

	# This function first detects the two lines l_1, l_2 which make up the prominent edges of the object in image 1 and
	# and the two lines \hat{l}_1,\hat{l}_2 which make up the prominent edges of the object in image 2. 

	# The function then detects the points p_1,p_2,p_3, and p_4. where
	# p_1 and p_2 are the intersection of l_1, l_2 respectively with the edge of image1
	# and p_3, p_4 are the intersection of l_1,l_2 respectively with the far edge of image2

	# Then the desired destination of the object transformation is chosen using our epipolar line alignment.
	# We compute a set of features \hat{p}_1, \hat{p}_2,\hat{p}_3, \hat{p}_4 that will contain the positions 
	# of the side view points which must be aligned to p_1, \ldots, p_4. 

	# \hat{p}_1 = (x,y) s.t. l_1*[x,y,1]' = 0 and  [x,y,1]*F*p_1 = 0					(I'm not certain if p_1 should be on the other side.)
	# \hat{p}_2 = (x,y) s.t. l_2*[x,y,1]' = 0 and  [x,y,1]*F*p_2 = 0
	# \hat{p}_3 = p_3 
	# \hat{p}_4 = p_4

	# The function then computes a homography transformation H_{obj} that align the \hat{p}_i to the p_i.

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

		#Identify the coordinate shift for sub neighborhoods. 
		shift11 = points1[0,:] - [N_size,N_size]
		shift12 = points1[1,:] - [N_size,N_size]
		shift21 = points2[0,:] - [N_size,N_size]
		shift22 = points2[1,:] - [N_size,N_size]


		#Identify dominant lines in sub images
		edges1 = cv2.Canny(sub11,edgeThresh,edgeThresh*3)
		lines1 = cv2.HoughLines(edges1,1,np.pi/180,N_size/2)

		edges2 = cv2.Canny(sub12,edgeThresh,edgeThresh*3)
		lines2 = cv2.HoughLines(edges2,1,np.pi/180,N_size/2)

		edges3 = cv2.Canny(sub21,edgeThresh,edgeThresh*3)
		lines3 = cv2.HoughLines(edges3,1,np.pi/180,N_size/2)

		edges4 = cv2.Canny(sub22,edgeThresh,edgeThresh*3)
		lines4 = cv2.HoughLines(edges4,1,np.pi/180,N_size/2)

		#cv2.imshow("sub11",sub11)
		#cv2.imshow("sub12",sub12)
		#cv2.imshow("sub21",sub21)
		#cv2.imshow("sub22",sub22)
		#cv2.waitKey(0)
		# If no lines are found, print error and return identity
		if (lines1 is None) or (lines2 is None) or (lines3 is None) or (lines4 is None):
			print "Error: Lines not found"
			return np.eye(3)

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
		points1[0,0], points1[0,1] = correctPoint(points1[0,0],points1[0,1],lines1)
		points1[1,0], points1[1,1] = correctPoint(points1[1,0],points1[1,1],lines2)
		points2[0,0], points2[0,1] = correctPoint(points2[0,0],points2[0,1],lines3)
		points2[1,0], points2[1,1] = correctPoint(points2[1,0],points2[1,1],lines4)

		#points1[0,0], points1[0,1] = setRight(lines1)
		#points1[1,0], points1[1,1] = setLeft(lines2)
		#points2[0,0], points2[0,1] = setRight(lines3)
		#points2[1,0], points2[1,1] = setLeft(lines4)


		

	#	print "LINES:",lines1,lines2,lines3,lines4
		#print points1[0][0],points1[1],points2[0],points2[1], (points1[0,0],points1[0,1])
		# Perform Epipolar Matching
		mat21 = epiMatch((points1[0,0],points1[0,1]),lines3, F)
		mat22 = epiMatch((points1[1,0],points1[0,1]),lines4, F)
		mat11 = epiMatch((points2[0,0],points2[0,1]),lines1, np.matrix.transpose(F))
		mat12 = epiMatch((points2[1,0],points2[1,1]),lines2, np.matrix.transpose(F))
		#print "IS THIS RIGHT: ",(points2[1,0],points2[1,1]), mat12

		#Organize Points for findHomography
		ptsB = np.zeros((4,2))
		ptsB[0] = (points1[0,0],points1[0,1])
		ptsB[1] = (points1[1,0],points1[1,1])
		#ptsB[2] = mat11
		#ptsB[3] = mat12
		ptsB[2] = findOuterEdge(image2,seam,lines1[0,:])
		ptsB[3] = findOuterEdge(image2,seam,lines2[0,:])


		ptsA = np.zeros((4,2))
		ptsA[0] = mat21
		ptsA[1] = mat22
		#ptsA[2] = (points2[0,0],points2[0,1])
		#ptsA[3] = (points2[1,0],points2[1,1])
		ptsA[2] = findOuterEdge(image2,seam,lines1[0,:])
		ptsA[3] = findOuterEdge(image2,seam,lines2[0,:])

		# Display the epipolar feature matches if DRAW_LINES flag. 
		if DRAW_LINES:
			tmp_image1 = np.mat([np.copy(image1),np.copy(image1),np.copy(image1)])
			tmp_image2 = np.mat([np.copy(image2),np.copy(image2),np.copy(image2)])

			drawLines(lines1[0],lines1[1],tmp_image1)
			drawLines(lines2[0],lines2[1],tmp_image1,(0,255,0))
			drawLines(lines3[0],lines3[1],tmp_image2)
			drawLines(lines4[0],lines4[1],tmp_image2,(0,255,0))

			cv2.circle(tmp_image1, mat11, 5, (255,0,0))
			cv2.circle(tmp_image2, mat21, 5, (0,255,0))
			cv2.circle(tmp_image1, mat12, 5, (0,0,255))
			cv2.circle(tmp_image2, mat22, 5, (0,255,255))

			cv2.circle(tmp_image1, (points1[0,0],points1[0,1]), 5, (0,255,0))
			cv2.circle(tmp_image2, (points2[0,0],points2[0,1]), 5, (255,0,0))
			cv2.circle(tmp_image1, (points1[1,0],points1[1,1]), 5, (0,255,255))
			cv2.circle(tmp_image2, (points2[1,0],points2[1,1]), 5, (0,0,255))

			# Find epilines corresponding to points in right image (second image) and
			# drawing its lines on left image
			linesA = cv2.computeCorrespondEpilines(ptsA, 2,F)
			linesB = cv2.computeCorrespondEpilines(ptsB, 1,F)
			linesA = linesA.reshape(-1,3)
			linesB = linesB.reshape(-1,3)
			img5,img6 = drawEpilines(tmp_image1,tmp_image2,linesA,linesB,ptsB,ptsA)

			#cv2.imshow("image1 (2 edge)",img5)
			#cv2.imshow("IDK1",img6)
			#cv2.imshow("image2 (2 edge)",img6)
			#cv2.imshow("IDK2",img4)
			#cv2.waitKey(0)

		#print "PTS: ", ptsA,ptsB

		(LineT, status) = cv2.findHomography(ptsA, ptsB)

		return LineT
	if (points1.shape[0] == 1):
		#Error checking code. If the point is too close to the edge, this bugs out and causes the program to crash. We shift the window to prevent this from happening. 
		# This may be a source of innefficiency which may need to be changed later. 
		if points1[0,1] <= N_size:
			points1[0,1] = N_size
		if points1[0,0] <= N_size:
			points1[0,0] = N_size
		if points2[0,1] <= N_size:
			points2[0,1] = N_size
		if points2[0,0] <= N_size:
			points2[0,0] = N_size
		#End Error checking segment. 

		#Isolate point Neighborhoods
		#sub1 = image1[points1[0,1] - N_size:points1[0,1]+N_size, points1[0,0] - N_size:points1[0,0]+N_size,:]
		sub1 = image1[points1[0,1] - N_size:points1[0,1]+N_size, points1[0,0] - N_size:points1[0,0]+N_size]

		#shift1 = np.mat([points1[0,0] - N_size, points1[0,1] - N_size])
		#sub2 = image2[points2[0,1] - N_size:points2[0,1]+N_size, points2[0,0] - N_size:points2[0,0]+N_size,:]
		sub2 = image2[points2[0,1] - N_size:points2[0,1]+N_size, points2[0,0] - N_size:points2[0,0]+N_size]

		#shift2 = np.mat([points2[0,0] - N_size, points2[0,1] - N_size])

		#Identify the coordinate shift for sub neighborhoods. 
		shift1 = points1[0,:] - [N_size,N_size]
		shift2 = points2[0,:] - [N_size,N_size]


		#Identify dominant lines in sub images
		#edges1 = cv2.Canny(sub1,edgeThresh,edgeThresh*3)
		#lines1 = cv2.HoughLines(edges1,1,np.pi/180,N_size/2)


		#edges2 = cv2.Canny(sub2,edgeThresh,edgeThresh*3)
		#lines2 = cv2.HoughLines(edges2,1,np.pi/180,N_size/2)

		lines1 = lineSelect(sub1,edgeThresh,N_size)
		lines2 = lineSelect(sub2,edgeThresh,N_size)
		print "lines1  found ",len(lines1)," lines"
		print "lines2 found ",len(lines2)," lines"
		
		#cv2.imshow("sub1",sub1)
		#cv2.imshow("sub2",sub2)
		#cv2.waitKey(0)
		#cv2.destroyWindow('sub1')
		#cv2.destroyWindow('sub2')

		# If no lines are found, print error and return identity
		if (lines1 is None) or (lines2 is None):
			print "Error: Lines not found", lines1, lines2
			return np.zeros((3,3))
		#Select Lines (Each subimage should only give one line. This is just ensuring they are the right shape to avoid errors.)
		#lines1 = lines1[2]
		#lines2 = lines2[2]

		sub1 = np.repeat(sub1[:,:,np.newaxis],3,axis=2)
		sub2 = np.repeat(sub2[:,:,np.newaxis],3,axis=2)
		print lines1
		drawLines(lines1[0,0],lines1[0,1],sub1)
		drawLines(lines2[0,0],lines2[0,1],sub2)
		cv2.imshow("line1",sub1)
		cv2.imshow("line2",sub2)
		cv2.waitKey(0)
		cv2.destroyWindow('line1')
		cv2.destroyWindow('line2')

		#Adjust lines for coordinate shift due to cropping. 
		#np.cos(np.arctan2(y,x) - theta)*np.sqrt(x^2 + y^2) + rho

		#print "Lines 1: ",lines1
		#print "Lines 2: ",lines2
		lines1 = [np.cos(-np.arctan2(shift1[0,1],shift1[0,0]) + lines1[0,1])*np.sqrt(pow(shift1[0,0],2) + pow(shift1[0,1],2)) + lines1[0,0],lines1[0,1]]
		lines2 = [np.cos(-np.arctan2(shift2[0,1],shift2[0,0]) + lines2[0,1])*np.sqrt(pow(shift2[0,0],2) + pow(shift2[0,1],2)) + lines2[0,0],lines2[0,1]]
		#print "Lines 1: ",lines1
		#print "Lines 2: ",lines2

		#Ensure points lie on a line.
		#print points1, correctPoint(points1[0,0],points1[0,1],lines1)
		points1[0,0], points1[0,1] = correctPoint(points1[0,0],points1[0,1],lines1)
		#points2[0,0], points2[0,1] = correctPoint(points2[0,0],points2[0,1],lines2)


		#points1[0,0], points1[0,1] = setLeft(lines1)
		#points2[0,0], points2[0,1] = setRight(lines2)

		#points1[0,0], points1[0,1] = shiftLeft(points1[0,0],points1[0,1],lines1,10)
		#points2[0,0], points2[0,1] = shiftRight(points2[0,0],points2[0,1],lines2,10)


		#print "LINES:",lines1,lines2
		#print points1[0][0],points1[1],points2[0],points2[1], (points1[0,0],points1[0,1])
		seam2 = main_seam.copy()
		seam2 = 255*seam2.astype('uint8')
		drawLines(lines1[0],lines1[1],seam2)
		drawLines(lines2[0],lines2[1],seam2)
		cv2.imshow("seam", seam2)
		cv2.waitKey(0)
		cv2.destroyWindow('seam')

		# Perform Feature Matching
		points1 = (points1[0,0],points1[0,1])
		points2 = lineEdge(lines2,side_seam,side_border)
		#points2 = (points2[0,0],points2[0,1])
		mat1 = epiMatch(points1,lines2, F)
		mat2 = lineEdge(lines1,main_seam,transformed_side_border)
		points1 = (points1[0] + shift[0],points1[1] + shift[1])
		#mat2 = points2
		#print "Points1: ", points1
		#print "Points2: ",points2
		#print "mat1: ", mat1
		#print "mat2: ",mat2

		if (len(points1) == 2) and (len(points2) == 2) and (len(mat1) == 2) and (len(mat2) == 2):

			if DRAW_LINES:
				ptsB = np.mat([points1,mat2])
				ptsA = np.mat([mat1,points2])
				#tmp_image1 = np.repeat(image1[:, :, np.newaxis], 3, axis=2)
				tmp_image2 = np.repeat(image2[:, :, np.newaxis]/2, 3, axis=2)
				tmp_image1 = np.copy(100*main_seam).astype('uint8')

				print tmp_image1.shape

				drawLines(lines1[1],lines1[0],tmp_image1)
				drawLines(lines2[1],lines2[0],tmp_image2)


				
				cv2.circle(tmp_image1, mat2, 5, (255,0,0))
				cv2.circle(tmp_image2, mat1, 5, (0,255,0))
				cv2.circle(tmp_image1, points1, 5, (0,255,0))
				cv2.circle(tmp_image2, points2, 5, (255,0,0))

				#print points1,points2
				print "PtsA: ",ptsA
				print "PtsB: ",ptsB
				linesA = cv2.computeCorrespondEpilines(ptsA, 2,F)
				linesB = cv2.computeCorrespondEpilines(ptsB, 1,F)
				linesA = linesA.reshape(-1,3)
				linesB = linesB.reshape(-1,3)
				img5,img6 = drawEpilines(tmp_image1,tmp_image2,linesA,linesB,ptsB,ptsA)

				cv2.imshow("image1 (1 edge)",img5)
				cv2.imshow("image2 (1 edge)",img6)
				cv2.waitKey(0)
				cv2.destroyWindow("image1 (1 edge)")
				cv2.destroyWindow("image2 (1 edge)")



			LineT = matchLines(points1, mat1, points2, mat2)
			

			print "Proposed PtsB: ",np.dot(LineT,[[ptsA[0,0],ptsA[1,0]],[ptsA[0,1],ptsA[1,1]],[1,1]])
			print "Actual Pts B: ",ptsB
			print "LineT: ",LineT

		else:
			LineT = np.eye(3)

		return LineT
	else:
		print "ERROR: Unexpected number of points."
		return np.zeros((3,3))
