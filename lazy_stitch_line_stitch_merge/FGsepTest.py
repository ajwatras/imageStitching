import numpy as np
import cv2

def drawLines(rho,theta,image,color=(0,0,255),width=2):
#Copied code for drawing the lines on an image given rho and theta
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
    
	cv2.line(image,(x1,y1),(x2,y2),color,width)


cap1 =  cv2.VideoCapture("output1.avi")
fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg2 = cv2.bgsegm.createBackgroundSubtractorGMG()
fgbg3 = cv2.bgsegm.createBackgroundSubtractorMOG()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

ret, frame = cap1.read()

while ret:
	ret,frame = cap1.read()
	fgmask1 = fgbg.apply(frame)

	fgmask2 = fgbg2.apply(frame)
	fgmask2 = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, kernel)

	fgmask3 = fgbg3.apply(frame)
	edges = cv2.Canny(fgmask3,.3,.9)
	edges2 = np.concatenate((np.zeros([420,640]),edges[420:480,:]),axis=0).astype('uint8')
	lines = cv2.HoughLines(edges2,1,np.pi/180,20)

	edge_slice = np.nonzero(edges[420,:])
	print edge_slice[0]
	if len(edge_slice[0]) is not 0:
		xmin = np.min(edge_slice)
		xmax = np.max(edge_slice)
		obj_found = True

	else:
		obj_found = False

	if obj_found:
		cv2.circle(frame,(xmin,420),5,(0,255,255),-1)
		cv2.circle(frame,(xmax,420),5,(0,255,255),-1)
	if lines is not None:
		for k in range(0,len(lines)):
			#print lines[k]
			drawLines(lines[k][0,0],lines[k][0,1],frame)

	#cv2.imshow("output",fgmask1)
	#cv2.imshow('GMG',fgmask2)
	cv2.imshow('MOG',fgmask3)
	cv2.imshow('edges',edges)
	cv2.imshow('lines',frame)
	if cv2.waitKey(100) & 0xFF == ord('q'):
		break




