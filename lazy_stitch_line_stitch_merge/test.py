import numpy as np
import cv2
import stitcher
import line_align

theta = np.pi/6
H = np.mat([[np.cos(theta),-np.sin(theta),-15],[np.sin(theta),np.cos(theta),-25],[0,0,1]])
image1 = np.zeros([480,640,3])
image2 = np.zeros([480,640,3])

stitch = stitcher.Stitcher()

map_image = stitch.mapMainView(H,image1,image2)

cv2.imshow("Edges",map_image)
cv2.waitKey(0)