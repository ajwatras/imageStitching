import numpy as np
import cv2
import line_align4 as la


filepath = '../data/pi_writer/'
caps = [cv2.VideoCapture(filepath+'output1.avi'),cv2.VideoCapture(filepath+'output2.avi'), cv2.VideoCapture(filepath+'output3.avi'), cv2.VideoCapture(filepath+'output4.avi')]

a = la.lazy_stitcher(caps)

main_view_background = np.ones([252,250,3]).astype('uint8')
side_view_background = np.ones([250,250,3]).astype('uint8')

main_view = np.copy(main_view_background)
side_view = np.copy(side_view_background)

la.drawLines(50,0,main_view,(255,255,255))
la.drawLines(19,0,side_view,(255,255,255))

cv2.imshow("a_back", main_view_background)
cv2.imshow("b_back",side_view_background)
cv2.imshow("a", main_view)
cv2.imshow("b",side_view)
cv2.waitKey(0)

a.background_models[0] = main_view_background
a.background_models[1] = side_view_background

a.quantizeAlignmentError(1,main_view,side_view)
