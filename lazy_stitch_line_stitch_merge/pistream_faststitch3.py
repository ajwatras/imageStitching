import cv2
import urllib
import numpy as np
import line_align4 as la
import time

################################## Set Input Video Locations ##################################################################
# Video can be read from a variety of different locations. OpenCV by default handles USB webcams, but reading from video files 
# can be useful for testing purposes, and the EasyVis prototype uses ethernet for video streaming (See *Paper Location TBD*).
# Set the approppriate video source below. 


# If reading from streaming video
#cap1 = cv2.VideoCapture('http://10.42.0.101:8010/?action=stream')
#cap2 = cv2.VideoCapture('http://10.42.0.102:8020/?action=stream')
#cap3 = cv2.VideoCapture('http://10.42.0.103:8030/?action=stream')
#cap4 = cv2.VideoCapture('http://10.42.0.104:8040/?action=stream')
#cap_main = cap1
#cap_side = [cap2,cap3,cap4]

# If reading from file
filepath = '../data/pi_writer/'         #If reading from file, put video file location here. 
cap_main = cv2.VideoCapture(filepath+'output1.avi')
cap_side = [cv2.VideoCapture(filepath+'output2.avi'), cv2.VideoCapture(filepath+'output3.avi'), cv2.VideoCapture(filepath+'output4.avi')]

# If reading from webcam
#print filepath+'m.avi' 
#cap_main = cv2.VideoCapture(1)
#cap_side = [cv2.VideoCapture(2), cv2.VideoCapture(3), cv2.VideoCapture(4)]

################################ Reset Timing Info #############################################################################
# In order to benchmark the line alignment code, we divide the streaming phase of the algorithm into three major steps: Detection
# Alignment, and Warping. The "timing_compile.sh" script in the parent directory can be used to get average runtimes for these
# steps, but we want to ensure that we only include the latest code run in our averaging, so we reset that here. 

#Clear timing info
open('obj_align_timing.txt','w').close()
open('obj_det_timing.txt','w').close()
open('obj_warp_timing.txt','w').close()
open('frame_timing.txt','w').close()

############################### Perform Calibration ############################################################################
# In order to speed up runtime during video streaming, we use a calibration phase to pre-compute many different values. This is 
# different from traditional camera calibration as it does not seek to identify the same parameters (camera pose, etc. ). However
# it serves a similar purpose in allowing us to shortcut computation later on. To read more about the calibration phase, see
# *Paper Location TBD*


# Calibrate lazy stitcher (See lazy_stitcher3.py)
a = la.lazy_stitcher(cap_main,cap_side)

# Initialize Output Video Writer 
# For Real Time Recording, use Kazam software instead to capture video with variable frame rates.
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./pano.avi',fourcc, 20.0, (467,454))


################################ Begin Streaming Phase #############################################################################
# The large bulk of our runtime is spent converting video streams into a single video panorama. This contains the implementation of
# *Paper Location TBD*.

side_view_frames = [[]] * len(cap_side)
## Read Initial Frames (Reading new frames done at end of loop to ensure proper termination if video feed ends)
ret,main_view_frame = cap_main.read()
for i in range(len(cap_side)):
    _,side_view_frames[i] = cap_side[i].read()

## Loop through each frame
while ret:
    t = time.time()

    ## Correct for intensity discrepancy (See *Paper Location TBD*)
    frame_list = [[]] * (len(cap_side) + 1)
    frame_list[0] = main_view_frame
    for i in range(len(cap_side)):
        frame_list[i+1] = side_view_frames[i]
    #Apply Correction
    main_view_frame,side_view_frames = a.correctIntensity(main_view_frame,side_view_frames)


    ## Apply stitching (See lazy_stitcher3.py)
    pano = a.stitch(main_view_frame, side_view_frames,a.background_models)

    ## Fill in Background with average color (See *Paper Location TBD*)
    #hole_mask = (pano == 0).astype('uint8')
    #pano[:,:,0] = hole_mask[:,:,0]*a.max_weight[0] + pano[:,:,0]
    #pano[:,:,1] = hole_mask[:,:,1]*a.max_weight[1] + pano[:,:,1]
    #pano[:,:,2] = hole_mask[:,:,2]*a.max_weight[2] + pano[:,:,2]

    ## Adjust image size to more manageable window size
    pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    ##Display Output
    cv2.imshow('pano',pano)
    
    print "Frame Time: ",time.time() - t

    
    ## Write frame to file
    print "Write size:" ,pano.shape
    out.write(pano)

    #Check Termination Key
    if cv2.waitKey(1) == ord('q'):
        exit(0)

    ## Read next frames
    ret,main_view_frame = cap_main.read()
    for i in range(len(cap_side)):
        _,side_view_frames[i] = cap_side[i].read()
################################################### End Streaming ##############################################################################################3