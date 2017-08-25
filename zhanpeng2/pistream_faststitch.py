import cv2
import urllib
import numpy as np
import lazy_stitcher

def recal(cap_main, cap_side):
    _, main_view_frame = cap_main.read()
    _, side_view_frame_1 = cap_side[0].read()
    _, side_view_frame_2 = cap_side[1].read()
    _, side_view_frame_3 = cap_side[2].read()

    a = lazy_stitcher.lazy_stitcher(main_view_frame, [side_view_frame_1, side_view_frame_2, side_view_frame_3])
    return a


cap4 = cv2.VideoCapture('http://10.42.0.102:8020/?action=stream')
cap3 = cv2.VideoCapture('http://10.42.0.103:8030/?action=stream')
cap2 = cv2.VideoCapture('http://10.42.0.104:8040/?action=stream')
cap1 = cv2.VideoCapture('http://10.42.0.101:8010/?action=stream')
cap5 = cv2.VideoCapture('http://10.42.0.105:8050/?action=stream')


cap_main = cap1
cap_side = [cap2, cap3, cap4]
_, main_view_frame = cap_main.read()
_, side_view_frame_1 = cap_side[0].read()
_, side_view_frame_2 = cap_side[1].read()
_, side_view_frame_3 = cap_side[2].read()
#cv2.imshow('1', main_view_frame)
#cv2.imshow('2', side_view_frame_1)
#cv2.imshow('3', side_view_frame_2)
#cv2.imshow('4', side_view_frame_3)
#cv2.waitKey(0)
print main_view_frame.shape, side_view_frame_1.shape, side_view_frame_2.shape, side_view_frame_3.shape

a = lazy_stitcher.lazy_stitcher(main_view_frame, [side_view_frame_1, side_view_frame_2, side_view_frame_3])
time_wait = 1
while True:
    _, main_view_frame = cap_main.read()
    _, side_view_frame_1 = cap_side[0].read();#_, frame_ref = cap_ref.read()
    _, side_view_frame_2 = cap_side[1].read()
    _, side_view_frame_3 = cap_side[2].read()
    ret5, image5 = cap5.read()
    #_, frame_ref = cap_ref.read()
    #print frame_ref.shape


    pano, main_view_frame, side_view_frames = a.stitch(main_view_frame, [side_view_frame_1, side_view_frame_2, side_view_frame_3])
    #pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
    cv2.imshow('pano',pano)
    #cv2.imshow('ref',frame_ref)
    cv2.imshow("DepthCam", cv2.warpPerspective(image5,np.mat([[-1,0,image5.shape[1]],[0,-1,image5.shape[0]],[0,0,1]], np.float),(image5.shape[1],image5.shape[0])))

    rep = cv2.waitKey(time_wait)
    if rep == ord('p'):
        if time_wait == 1:
            time_wait = 0
        else:
            time_wait = 1
    if rep == ord('q'):
        exit(0)
    if rep == ord('1'):
        cap_main = cap1
        cap_side = [cap2, cap3, cap4]
        a = recal(cap_main, cap_side)
    if rep == ord('2'):
        cap_main = cap2
        cap_side = [cap3, cap4, cap1]
        a = recal(cap_main, cap_side)
    if rep == ord('3'):
        cap_main = cap3
        cap_side = [cap4, cap1, cap2]
        a = recal(cap_main, cap_side)
    if rep == ord('4'):
        cap_main = cap4
        cap_side = [cap1, cap2, cap3]
        a = recal(cap_main, cap_side)
