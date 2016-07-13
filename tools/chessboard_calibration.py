## This script can be used to generate the internal camera parameters of a camera
import numpy as np
import cv2
import glob

grid_x = 8
grid_y = 6
data_loc = '../data/calibration_images/radialDist/camera2/'

#termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((grid_x*grid_y,3),np.float32)
objp[:,:2] = np.mgrid[0:grid_y,0:grid_x].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

images = glob.glob(data_loc+'*.jpg')
#print images

for fname in images:
	print fname

	img = cv2.imread(fname)
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
		cv2.imshow('img',img)
		cv2.waitKey(500)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

print dist,mtx  # outputs the radial distortion vector followed by the camera internal parameters.

dst = cv2.undistort(img, mtx, dist, None, mtx)
for fname in images:
	img = cv2.imread(fname)
	cv2.imshow('img',dst)
	cv2.waitKey(500)

cv2.destroyAllWindows()

