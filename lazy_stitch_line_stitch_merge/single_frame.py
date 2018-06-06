import cv2
import urllib
import numpy as np
import lazy_stitcher3 as lazy_stitcher
import line_align as la



filepath = '../data/Bookshelf/'

main_view_frame = cv2.imread('../data/Bookshelf/S1.jpg')
side_view_frame_1 = cv2.imread('../data/Bookshelf/S2.jpg')
side_view_frame_2 = cv2.imread('../data/Bookshelf/S3.jpg')
side_view_frame_3 = cv2.imread('../data/Bookshelf/S4.jpg')

a = lazy_stitcher.lazy_stitcher(main_view_frame, [side_view_frame_1, side_view_frame_2, side_view_frame_3])

ret, models = la.modelBackground(caps,1)


pano, main_view_frame, side_view_frames = a.stitch(main_view_frame, [side_view_frame_1, side_view_frame_2, side_view_frame_3],models)
pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
cv2.imshow('pano',pano)
cv2.waitKey(0)