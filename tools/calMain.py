import numpy as np
import cv2
import calibrateCameras as ccam
import glob

cap1 = cv2.VideoCapture('http://10.42.0.104:8040/?action=stream')

images = ccam.gatherCalFrame([cap1])

dist, mtx = ccam.calcIntrinsics(images[0])

for fname in imglist:
	print fname

	img = cv2.imread(fname)
	images.append(img)

dist, mtx = ccam.calcIntrinsics(images)


np.savetxt('calibration/distortion.txt',dist,delimiter=',')
np.savetxt('calibration/camera.txt',mtx,delimiter=',')


print dist, mtx
