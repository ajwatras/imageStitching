import numpy as np
import cv2
import calibrateCameras as ccam
import glob

cap1 = cv2.VideoCapture('http://10.42.0.104:8040/?action=stream')

images = ccam.gatherCalFrame([cap1])

dist, mtx = ccam.calcIntrinsics(images[0])

<<<<<<< HEAD
=======
for fname in imglist:
	print fname

	img = cv2.imread(fname)
	images.append(img)

dist, mtx = ccam.calcIntrinsics(images)

>>>>>>> 10e0c484c46915b34b321d0d41b0650dd17acf08
np.savetxt('calibration/distortion.txt',dist,delimiter=',')
np.savetxt('calibration/camera.txt',mtx,delimiter=',')


print dist, mtx
