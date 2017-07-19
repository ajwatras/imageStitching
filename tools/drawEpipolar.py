import cv2
import numpy as np
from matplotlib import pyplot as plt
#import stitcher


def computeF(K1,K2,R,t):
	A = np.dot(np.dot(K1,np.transpose(R)),t)
	C = np.mat([[0,-A[2],A[1]],[A[2],0,-A[0]],[-A[1],-A[0],0]])
	K = np.transpose(np.inverse(K2))

	return  np.dot(K,np.dot(R,np.dot(np.transpose(K1),C)))
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


#cap1 = cv2.VideoCapture('../lazy_stitch_line_stitch_merge/drop3_fixed/output1.avi')
#cap2 = cv2.VideoCapture('../lazy_stitch_line_stitch_merge/drop3_fixed/output3.avi')

#cap1 = cv2.VideoCapture('http://10.42.0.101:8010/?action=stream')
#cap2 = cv2.VideoCapture('http://10.42.0.102:8020/?action=stream')

cap1 = cv2.VideoCapture('../data/7_18_17/bean1/output1.avi')
cap2 = cv2.VideoCapture('../data/7_18_17/bean1/output3.avi')

#cap1 = cv2.VideoCapture('../lazy_stitch_line_stitch_merge/sample/output1.avi')
#cap2 = cv2.VideoCapture('../lazy_stitch_line_stitch_merge/sample/output2.avi')

ret1,image1 = cap1.read()
ret2,image2 = cap2.read()

cv2.imshow("image1",image1)
cv2.imshow("image2",image2)
cv2.waitKey(0)

kps1,feat1 = detectAndDescribe(image1)
kps2,feat2 = detectAndDescribe(image2)

F,pts1,pts2 = calcF(image1,image2)
#file_array = np.load('calibration_7_14_17/output.npz')
file_array = np.load('./picked_calibration/calibration.npz')
(dst,K,R,t,F) = file_array['arr_0']
F = F[3]
#F = np.loadtxt('fundamentalMatrices/F1.txt',delimiter=',')
print pts1,pts2
while cap1.isOpened():
	ret1,image1 = cap1.read()
	ret2,image2 = cap2.read()


	image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
	image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)

	displayEpipolar(image1,image2,F,pts1,pts2)

	if cv2.waitKey(0) == ord('q'):
		exit(0)
