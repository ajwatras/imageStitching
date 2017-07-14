import numpy as np
import cv2
import calibrateCameras as ccam
import glob

#data_loc = '../data/calibration_images/radialDist/HandPickedC1/'


#imglist = glob.glob(data_loc+'*.jpg')
#images = []

#for fname in imglist:
#	print fname

#	img = cv2.imread(fname)
#	images.append(img)

#dist1,dist2,K1,K2,R,T,F = ccam.calibCam(images,images)

#print K1,K2,R,T

cap1 = cv2.VideoCapture('http://10.42.0.101:8010/?action=stream')
cap2 = cv2.VideoCapture('http://10.42.0.102:8020/?action=stream')
cap3 = cv2.VideoCapture('http://10.42.0.103:8030/?action=stream')
cap4 = cv2.VideoCapture('http://10.42.0.104:8040/?action=stream')
#cap2 = cv2.VideoCapture(0)

image_mat = ccam.gatherCalFrame((cap1,cap2,cap3,cap4))

print len(image_mat),len(image_mat[0]), image_mat[0][0].shape
dist = []
K = []
R = []
T = []
F = []

for k in range(1,len(image_mat)):
	print k
	dist1,dist2,K1,K2,R_temp,T_temp,F_temp = ccam.calibCam(image_mat[0],image_mat[k])

	dist.append(dist2)
	K.append(K2)
	R.append(R_temp)
	T.append(T_temp)
	F.append(F_temp)

dist.insert(0,dist1)
K.insert(0,K1)
R.insert(0,np.eye(3))
T.insert(0,np.mat([[0],[0],[0]]))
F.insert(0,np.eye(3))

save_var = (dist,K,R,T,F)

#np.savetxt('calibration.txt',save_var)
#np.savetxt('./calibration/distortion.txt',dist,delimiter=',')

ccam.saveCalibration(save_var)
save_var2 = ccam.loadCalibration('calibration/output.npz')

print 'Distortion Coefficients: ', dist
print 'K matrices: ',K
print 'Rotation matrices: ',R
print 'Translation Vectors ', T
print 'Fundamental Matrices: ',F

(dist,K,R,T,F) = save_var2

print 'Distortion Coefficients: ', dist
print 'K matrices: ',K
print 'Rotation matrices: ',R
print 'Translation Vectors ', T
print 'Fundamental Matrices: ',F
