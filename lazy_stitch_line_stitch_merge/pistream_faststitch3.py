import cv2
import urllib
import numpy as np
import lazy_stitcher3 as lazy_stitcher
import line_align as la
import time

################################## Set Input Video Locations ##################################################################

#cap1 = cv2.VideoCapture('http://10.42.0.101:8010/?action=stream')
#cap2 = cv2.VideoCapture('http://10.42.0.102:8020/?action=stream')
#cap3 = cv2.VideoCapture('http://10.42.0.103:8030/?action=stream')
#cap4 = cv2.VideoCapture('http://10.42.0.104:8040/?action=stream')
#cap_main = cap1
#cap_side = [cap2,cap3,cap4]

filepath = '../data/pi_writer/'
cap_main = cv2.VideoCapture(filepath+'output1.avi')
cap_side = [cv2.VideoCapture(filepath+'output2.avi'), cv2.VideoCapture(filepath+'output3.avi'), cv2.VideoCapture(filepath+'output4.avi')]

#filepath = '../data/Line_Align/test2/'
#cap_main = cv2.VideoCapture(filepath+'m.avi')
#cap_side = [cv2.VideoCapture(filepath+'s1.avi'), cv2.VideoCapture(filepath+'s2.avi'), cv2.VideoCapture(filepath+'s3.avi'),cv2.VideoCapture(filepath+'s4.avi')]

#print filepath+'m.avi' 
#cap_main = cv2.VideoCapture(1)
#cap_side = [cv2.VideoCapture(2), cv2.VideoCapture(3), cv2.VideoCapture(4)]

################################ Reset Timing Info #############################################################################

#Clear timing info
open('obj_align_timing.txt','w').close()
open('obj_det_timing.txt','w').close()
open('obj_warp_timing.txt','w').close()

############################### Perform Calibration ############################################################################

# Calibrate lazy stitcher (See lazy_stitcher3.py)
a = lazy_stitcher.lazy_stitcher(cap_main,cap_side)

################################# Compute Intensity Correction ##################################################################
# NOTE: Method is incomplete, and currently disabled
#ave_intensity = [[]] * (len(side_view_frames) + 1)
#a.intensity_weights = [[]] * (len(side_view_frames) + 1)

#for i in range(len(ave_intensity)):
#    ave_intensity[i] = np.mean(a.background_models[i])

#max_weight = np.max(ave_intensity)
#for i in range(len(ave_intensity)):
#    a.intensity_weights[i] = ave_intensity[i]/max_weight

#print a.intensity_weights

#for i in range(len(ave_intensity)):
#    a.intensity_weights = ave_intensity - max_weight


#print a.intensity_weights
##########################################################

# Initialize Output Video Writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./pano.avi',fourcc, 20.0, (467,454))


################################ Begin Streaming Phase #############################################################################
while True:

    t = time.time()
    # Read new frames
    _, main_view_frame = cap_main.read()
    _, side_view_frame_1 = cap_side[0].read()
    _, side_view_frame_2 = cap_side[1].read()
    _, side_view_frame_3 = cap_side[2].read()


    # Correct for intensity discrepancy
    #main_view_frame = (a.intensity_weights[0]*main_view_frame).astype('uint8')
    #side_view_frame_1 = (a.intensity_weights[1]*side_view_frame_1).astype('uint8')
    #side_view_frame_2 = (a.intensity_weights[2]*side_view_frame_2).astype('uint8')
    #side_view_frame_3 = (a.intensity_weights[3]*side_view_frame_3).astype('uint8')

    # Apply stitching (See lazy_stitcher3.py)
    pano, main_view_frame, side_view_frames = a.stitch(main_view_frame, [side_view_frame_1, side_view_frame_2, side_view_frame_3],a.background_models)

    # Adjust image size
    pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    #Display Output
    cv2.imshow('pano',pano)
    print "Frame Time: ",time.time() - t
    # Write frame to file
    print "write size"
    print pano.shape
    out.write(pano)

    #Check Termination Key
    if cv2.waitKey(1) == ord('q'):
        exit(0)
