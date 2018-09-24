import numpy as np
import cv2
import calibrateCameras as ccam
import glob

data_loc1 = './picked_calibration/output1/'
data_loc2 = './picked_calibration/output2/'
data_loc3 = './picked_calibration/output3/'
data_loc4 = './picked_calibration/output4/'
data_loc4 = './picked_calibration/output5/'

imglist1 = glob.glob(data_loc1+'*.jpg')
imglist2 = glob.glob(data_loc2+'*.jpg')
imglist3 = glob.glob(data_loc3+'*.jpg')
imglist4 = glob.glob(data_loc4+'*.jpg')
imglist5 = glob.glob(data_loc5+'*.jpg')
images1 = []
images2 = []
images3 = []
images4 = []
images5 = []

for fname in imglist1:
	print fname

	img = cv2.imread(fname)
	images1.append(img)
for fname in imglist2:
	print fname

	img = cv2.imread(fname)
	images2.append(img)
for fname in imglist3:
	print fname

	img = cv2.imread(fname)
	images3.append(img)
for fname in imglist4:
	print fname

	img = cv2.imread(fname)
	images4.append(img)
for fname in imglist5:
	print fname

	img = cv2.imread(fname)
	images5.append(img)

dist = []
K = []
R = []
T = []
F = []



dist1,dist2,K1,K2,R_temp,T_temp,F_temp = ccam.calibCam(images1,images2)

dist.append(dist1)
K.append(K1)
R.append(np.eye(3))
T.append([[0,][0],[0]])
F.append(np.eye(3))

dist.append(dist2)
K.append(K2)
R.append(R_temp)
T.append(T_temp)
F.append(F_temp)

dist1,dist2,K1,K2,R_temp,T_temp,F_temp = ccam.calibCam(images1,images3)
dist.append(dist2)
K.append(K2)
R.append(R_temp)
T.append(T_temp)
F.append(F_temp)

dist1,dist2,K1,K2,R_temp,T_temp,F_temp = ccam.calibCam(images1,images4)
dist.append(dist2)
K.append(K2)
R.append(R_temp)
T.append(T_temp)
F.append(F_temp)

dist1,dist2,K1,K2,R_temp,T_temp,F_temp = ccam.calibCam(images1,images5)
dist.append(dist2)
K.append(K2)
R.append(R_temp)
T.append(T_temp)
F.append(F_temp)

print "distortion: ",dist1,dist2
print "K1: ",K1
print "K2: ",K2
print "R: ",R
print "T: ",T
print "F: ",F

ccam.saveCalibration((dist,K,R,T,F),'./picked_calibration/calibration')
