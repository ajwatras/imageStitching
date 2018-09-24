import cv2
import numpy as np
import line_align as la

image = np.zeros((40,40,3))
mask = np.zeros((40,40,3))
for i in range(0,10):
	mask[:,i,:] = 1
line = [-np.pi/4,0]


print la.findOuterEdge(image,mask,line)

la.drawLines(line[1],line[0],mask)
cv2.imshow('mask',mask)
cv2.waitKey(0)