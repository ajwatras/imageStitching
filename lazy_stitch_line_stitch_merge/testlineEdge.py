import numpy as np
import line_align as la
import cv2

a = np.zeros((100,100,3))
a[:,0:50,:] = 255

edges = cv2.Canny(a[:,:,0],20,60)
line = cv2.HoughLines(edges,1,np.pi/180,N_size/2)
point = la.drawLines(line[0],line[1],a)
print point

cv2.imshow("image",a)
cv2.waitKey(0)
