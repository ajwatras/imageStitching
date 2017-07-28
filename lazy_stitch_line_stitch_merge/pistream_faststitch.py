import cv2
import urllib
import numpy as np
import lazy_stitcher

cap3 = cv2.VideoCapture('http://10.42.0.101:8010/?action=stream')
cap1 = cv2.VideoCapture('http://10.42.0.102:8020/?action=stream')
cap2 = cv2.VideoCapture('http://10.42.0.103:8030/?action=stream')
cap4 = cv2.VideoCapture('http://10.42.0.104:8040/?action=stream')
cap_main = cap1
cap_side = [cap2,cap3,cap4]

#filepath = '../data/7_18_17/bean1/'


#cap_main = cv2.VideoCapture(filepath+'output1.avi')
#cap_side = [cv2.VideoCapture(filepath+'output2.avi'), cv2.VideoCapture(filepath+'output3.avi'), cv2.VideoCapture(filepath+'output4.avi')]
_, main_view_frame = cap_main.read()
_, side_view_frame_1 = cap_side[0].read()
_, side_view_frame_2 = cap_side[1].read()
_, side_view_frame_3 = cap_side[2].read()

print main_view_frame.shape, side_view_frame_1.shape, side_view_frame_2.shape, side_view_frame_3.shape

a = lazy_stitcher.lazy_stitcher(main_view_frame, [side_view_frame_1, side_view_frame_2, side_view_frame_3])
while True:
    _, main_view_frame = cap_main.read()
    _, side_view_frame_1 = cap_side[0].read()
    _, side_view_frame_2 = cap_side[1].read()
    _, side_view_frame_3 = cap_side[2].read()


    pano, main_view_frame, side_view_frames = a.stitch(main_view_frame, [side_view_frame_1, side_view_frame_2, side_view_frame_3])
    #pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('pano',pano)

    if cv2.waitKey(1) == ord('q'):
        exit(0)
