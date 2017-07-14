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

np.save('calibration/distortion.txt',dist,delimiters=',')
np.save('calibration/camera.txt',mtx,delimiters=',')


print dist, mtx
