import numpy as np 
import cv2

im = cv2.imread('houghTest.png')
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

imgSplit = cv2.split(im)
flag,b = cv2.threshold(imgSplit[2],0,255,cv2.THRESH_OTSU) 

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(1,1))
cv2.erode(b,element)

edges = cv2.Canny(b,150,200,3,5)

cv2.imshow('frame',edges)
cv2.waitKey(0)

print edges.shape
while(True):

    img = im.copy()

    lines = cv2.HoughLines(edges,1,np.pi/180,100)
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
				cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)


#    lines = cv2.HoughLinesP(edges,1,np.pi/2,2, minLineLength = 120, maxLineGap = 100)
#    print lines
#    for k in range(0,lines.shape[0]):
#	    for x1,y1,x2,y2 in lines[k]:
#	    	cv2.line(img,(x1,y1),(x2,y2),(0,255,0),1)

    cv2.imshow('houghlines',img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
    	break

cv2.destroyAllWindows()