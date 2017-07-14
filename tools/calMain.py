import numpy as np
import cv2
import calibrateCameras as ccam
import glob

cap1 = cv2.VideoCapture('http://10.42.0.104:8040/?action=stream')

images = ccam.gatherCalFrame([cap1])

dist, mtx = ccam.calcIntrinsics(images[0])

np.savetxt('calibration/distortion.txt',dist,delimiter=',')
np.savetxt('calibration/camera.txt',mtx,delimiter=',')


print dist, mtx
