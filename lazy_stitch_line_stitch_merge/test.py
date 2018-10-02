import stitcher as sti
import line_align as la
import cv2
import numpy as np






main_frame = 100*np.ones((480,640,3)).astype('uint8')
side_frame = 100*np.ones((480,640,3)).astype('uint8')
line1 = (-250,-np.pi/2.2)
line2 = (-250,-np.pi/1.8)
la.drawLines(line1[0],line1[1], main_frame, (255,255,255),10)
la.drawLines(line2[0],line2[1], side_frame, (255,255,255),10)

corners = np.mat([[0,0],[0,side_frame.shape[1]],[side_frame.shape[0],0],[side_frame.shape[0],side_frame.shape[1]]])
new_corners = np.mat([[100,100],[100,side_frame.shape[1]+100],[side_frame.shape[0]+100,100],[100+side_frame.shape[0],100+side_frame.shape[1]]])
H,h_mask = cv2.findHomography(corners,new_corners)

main_view_seam = np.zeros((480,640,3))
main_view_seam[:,639,:] = 1

print "H: ",H

line11,line12 = la.lineDetect(main_frame)
line21,line22 = la.lineDetect(side_frame)
line1,mat1,line2,mat2 = la.pairLines((line11,line12),(line21,line22),H,main_view_seam)

print "R of 5: ",la.checkHalfPlane(253.0, 1.5707964,np.transpose(np.mat([639,246])))
print "R of -5: ",la.checkHalfPlane(10,np.pi/2,np.transpose(np.mat([0,-5])))

la.drawLines(line1[0],line1[1],main_frame,(255,0,0),2)
la.drawLines(line2[0],line2[1],main_frame,(0,255,0),2)

la.drawLines(mat1[0],mat1[1],side_frame,(255,0,0),2)
la.drawLines(mat2[0],mat2[1],side_frame,(0,255,0),2)

cv2.imshow("detected lines",main_frame)
cv2.imshow("detected matches",side_frame)
cv2.waitKey(0)