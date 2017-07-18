import numpy as np
import cv2
import calibrateCameras as ccam
import glob

data_loc = '../data/calibration_images/radialDist/HandPickedC1/'


imglist = glob.glob(data_loc+'*.jpg')
images = []

for fname in imglist:
	print fname

	img = cv2.imread(fname)
	images.append(img)

dist, mtx = ccam.calcIntrinsics(images)

np.savetxt('calibration/distortion.txt',dist,delimiter=',')
np.savetxt('calibration/camera.txt',mtx,delimiter=',')


print dist, mtx
