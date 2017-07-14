import cv2
import urllib
import numpy as np
import lazy_stitcher

cap_main = cv2.VideoCapture('sample/output1.avi')
cap_side = [cv2.VideoCapture('sample/output2.avi'), cv2.VideoCapture('sample/output3.avi'), cv2.VideoCapture('sample/output4.avi')]
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
    pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('pano',pano)

    if cv2.waitKey(1) == ord('q'):
        exit(0)
