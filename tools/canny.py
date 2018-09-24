import cv2
import numpy as np
from matplotlib import pyplot as plt


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
 
img = cv2.imread('../data/vid2frame/output1frame00351.jpg',0)
#plt.imshow(img)
cv2.imshow("a",img)
cv2.waitKey()
edges = cv2.Canny(img,20,60)
lines = cv2.HoughLines(edges,1,np.pi/180,10)
lines = lines[0]
print lines
drawLines(lines[0,0],lines[0,1],img)
 
#plt.subplot(121),cv2.imshow("original",img)
#plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#plt.subplot(122),plt.imshow(edges)
#plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#plt.show() 