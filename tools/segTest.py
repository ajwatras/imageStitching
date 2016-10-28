import cv2
import numpy as np


minLineLength = 100
maxLineGap = 10


vid = cv2.VideoCapture('../data/testVideo/WideView-BeanDrop1/output1.avi')


success,frame = vid.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
fgbg = cv2.createBackgroundSubtractorMOG2()

while success:
	fgmask = fgbg.apply(frame)
	edges = cv2.Canny(fgmask,100,200)
	cv2.imshow('edges',edges)


	lines = cv2.HoughLines(edges,1,np.pi/180,120)
	if lines is not None:
		for k in range(0,lines.shape[0]):
			for rho,theta in lines[k]:
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))
				print x1,y1,x2,y2,rho,theta
				cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)

	lines = cv2.HoughLinesP(edges,1,np.pi/180,10,minLineLength,maxLineGap)
	if lines is not None:
		for k in range(0,lines.shape[0]):
			for x1,y1,x2,y2 in lines[k]:
				cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)


	cv2.imshow('frame', frame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
        	break


	success,frame = vid.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	