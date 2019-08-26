import cv2
import urllib
import numpy as np
import line_align4 as la
import time
import threading
import os
import copy

class Main(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
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
        #cap_main = cv2.VideoCapture(filepath+'output1.avi')
        #cap_side = [cv2.VideoCapture(filepath+'output2.avi'), cv2.VideoCapture(filepath+'output3.avi'), cv2.VideoCapture(filepath+'output4.avi')]
        caps = [cv2.VideoCapture(filepath+'output1.avi'),cv2.VideoCapture(filepath+'output2.avi'), cv2.VideoCapture(filepath+'output3.avi'), cv2.VideoCapture(filepath+'output4.avi')]
        #filepath = './AlignmentTraining/'         #If reading from file, put video file location here. 
        #caps = [cv2.VideoCapture(filepath+'m.avi'),cv2.VideoCapture(filepath+'s1.avi'), cv2.VideoCapture(filepath+'s2.avi'), cv2.VideoCapture(filepath+'s3.avi'),cv2.VideoCapture(filepath+'s4.avi')]

        # If reading from webcam
        #print filepath+'m.avi' 
        #cap_main = cv2.VideoCapture(1)
        #cap_side = [cv2.VideoCapture(2), cv2.VideoCapture(3), cv2.VideoCapture(4)]

        ############################### Set Running Parameters #########################################################################
        # Set runtime flags in order to  turn on or off certain functionality. DO_LINE_ALINE turns on parallax correction. 
        # DO_QUANTIZE_ERROR turns on 
        self.DO_LINE_ALINE = True
        self.DO_QUANTIZE_ERROR = True
        self.NUM_SIMUL_FRAMES = 5
        self.num_cam = len(caps)
        global frames_q
        frames_q = [[]] * (self.num_cam* self.NUM_SIMUL_FRAMES)
        im_readers = [[]] * self.num_cam


        ############################### Perform Calibration ############################################################################
        # In order to speed up runtime during video streaming, we use a calibration phase to pre-compute many different values. This is 
        # different from traditional camera calibration as it does not seek to identify the same parameters (camera pose, etc. ). However
        # it serves a similar purpose in allowing us to shortcut computation later on. To read more about the calibration phase, see
        # *Paper Location TBD*


        # Calibrate lazy stitcher (See lazy_stitcher3.py)
        a = la.lazy_stitcher(caps)

        # For Real Time Recording, use Kazam software instead to capture video with variable frame rates.
        #fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter('./pano_singleH.avi',fourcc, 20.0, (467,454))

        ############################### Begin Streaming ##################################################################################
        stitchThreads = [[]] * self.NUM_SIMUL_FRAMES
        for i in range(self.num_cam):
            im_readers[i] = imageReader(i,caps[i])
            im_readers[i].start()
        for i in range(self.NUM_SIMUL_FRAMES):
            stitchThreads[i] = stitchingThread(i,copy.deepcopy(a))
            stitchThreads[i].start()


        while True:


            # Read new frames
            t = time.time()
            for i in range(self.num_cam):
                im_readers[i].runThread = True
            for i in range(self.num_cam):
                print i
                while im_readers[i].runThread:
                    time.sleep(0.001)
            read_time = time.time() - t

            # Stitch latest frames
            t = time.time()
            for i in range(self.NUM_SIMUL_FRAMES):
                stitchThreads[i].frames = frames_q[self.num_cam * i:self.num_cam * (i+1)]
                stitchThreads[i].runThread = True
            for i in range(self.NUM_SIMUL_FRAMES):
                while stitchThreads[i].runThread:
                    time.sleep(0.001)

            stitch_time = time.time() - t

            # Play latest frames
            t = time.time()
            for i in range(self.NUM_SIMUL_FRAMES):
                cv2.imshow("frame",stitchThreads[i].pano)
                rep = cv2.waitKey(1)

                if rep == ord('q'):
                    os._exit(1)

            display_time = time.time() - t

            print "Read time: " + str(read_time* 1000)+ " ms"
            print "Stitch time: " + str(stitch_time* 1000)+ " ms"
            print "Display time: " + str(display_time * 1000) + " ms"

            

        ################################################### End Streaming ##############################################################################################
        os._exit(1)

class stitchingThread(threading.Thread):
    def __init__(self,idx, lazy_stitcher):
        threading.Thread.__init__(self)
        self.num_frames = len(frames_q)
        self.frames = [] * self.num_frames
        self.pano = []
        self.sti = lazy_stitcher
        self.idx = idx
        self.runThread = False


    def run(self):
        while True:
            if self.runThread:
                t = time.time()
                self.pano,background_models = self.sti.stitch(self.frames,self.sti.background_models)
                self.runThread = False
                print "Stitching Thread Completion Time: " + str((time.time() - t)*1000) + " ms"



class imageReader(threading.Thread):
    def __init__(self,idx, cap):
        threading.Thread.__init__(self)
        self.idx = idx 
        self.NUM_SIMUL_FRAMES = 5
        self.num_cam = 4
        self.cap = cap
        self.runThread = False

    def run(self):
        global frames_q
        while True:
            if self.runThread:
                for i in range(self.NUM_SIMUL_FRAMES):
                    #ret,frames_q[i + (self.idx) * self.NUM_SIMUL_FRAMES] = self.cap.read()
                    ret,frames_q[i * self.num_cam + self.idx]= self.cap.read()
                    #ret, self.frames[i] = self.cap.read()
                self.runThread = False

class displayThread(threading.Thread):
    def __init__(self):
        self.pano_queue = []
        self.runThread = False
    def run():
        return


################################ Set Global Variables #########################################################################
frames_q = []
NUM_SIMUL_FRAMES = []
num_cam = []
################################ Reset Timing Info #############################################################################
# In order to benchmark the line alignment code, we divide the streaming phase of the algorithm into three major steps: Detection
# Alignment, and Warping. The "timing_compile.sh" script in the parent directory can be used to get average runtimes for these
# steps, but we want to ensure that we only include the latest code run in our averaging, so we reset that here. 

#Clear timing info
open('obj_align_timing.txt','w').close()
open('obj_det_timing.txt','w').close()
open('obj_warp_timing.txt','w').close()
open('frame_timing.txt','w').close

# For finer grain timing info add these files
open('read_frame_timing.txt','w').close()
open('bg_subtract_timing.txt','w').close()
open('check_seam_timing.txt','w').close()
open('Det_edge_line_timing.txt','w').close()
open('Match_lines_timing.txt','w').close()
open('det_feat_timing.txt','w').close()
open('comp_H_timing.txt','w').close()
open('apply_H_timing.txt','w').close()
open('blend_timing.txt','w').close()

# For test purposes we use timing_test.txt
open('test_time.txt','w').close()

############################### Initialize Threads #################################################################################
# For multithreading purposes, we initialize 5 threads to simultaneously process 5 frames. 
frames_q = []
main = Main()

main.start()
main.join()



################################ Begin Streaming Phase #############################################################################
# The large bulk of our runtime is spent converting video streams into a single video panorama. This contains the implementation of
# *Paper Location TBD*.

#side_view_frames = [[]] * len(cap_side)
## Read Initial Frames (Reading new frames done at end of loop to ensure proper termination if video feed ends)
#ret,main_view_frame = cap_main.read()
#for i in range(len(cap_side)):
#    _,side_view_frames[i] = cap_side[i].read()


frame_q = [[]] * NUM_SIMUL_FRAMES
for i in range(NUM_SIMUL_FRAMES):
    frames = [[]] * len(caps)
    for j in range(len(caps)):
        ret,frames[j] = caps[j].read()
    frame_q[i] = frames

print "Frame Length 1: ",len(frames)
## Loop through each frame
while ret:
    t = time.time()

    ## Correct for intensity discrepancy (See *Paper Location TBD*)
    #frame_list = [[]] * (len(cap_side) + 1)
    #frame_list[0] = main_view_frame
    #for i in range(len(cap_side)):
    #    frame_list[i+1] = side_view_frames[i]
    #Apply Correction
    #main_view_frame,side_view_frames = a.correctIntensity(main_view_frame,side_view_frames)


    ## Apply stitching (See lazy_stitcher3.py)
    if DO_LINE_ALINE:
        pano,background_models = a.stitch(frames,a.background_models)
    else: 
        pano = a.stitch2(frames, a.background_models)


    ## Fill in Background with average color (See *Paper Location TBD*)
    #hole_mask = (pano == 0).astype('uint8')
    #pano[:,:,0] = hole_mask[:,:,0]*a.max_weight[0] + pano[:,:,0]
    #pano[:,:,1] = hole_mask[:,:,1]*a.max_weight[1] + pano[:,:,1]
    #pano[:,:,2] = hole_mask[:,:,2]*a.max_weight[2] + pano[:,:,2]

    ## Adjust image size to more manageable window size
    print "Pano: ", pano.shape,pano.dtype

    pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

    ##Display Output
    cv2.imshow('pano',pano)
    
    print "Frame Time: ",time.time() - t
    file.write("Frame Time: ")
    file.write(str(time.time() - t))
    file.write("\n")
    
    ## Write frame to file
    print "Write size:" ,pano.shape
    out.write(pano)

    #Check Termination Key
    rep = cv2.waitKey(10)

    ## Read next frames
    #ret,main_view_frame = cap_main.read()
    #for i in range(len(cap_side)):
    #    _,side_view_frames[i] = cap_side[i].read()
    print "Frame Length: ",len(frames)
    t = time.time()
    for i in range(len(caps)):
        ret,frames[i] = caps[i].read()
    read_time = time.time() - t
    read_file = open('read_frame_timing.txt','a')
    read_file.write("Frame Reading time : ")
    read_file.write(str(read_time))
    read_file.write('\n')



    if rep == ord('q'):
       ret = False
    if rep == ord('r'):
        stitcher = a.calibrate(frames_q)
    if rep == ord('1'):
        print "Changing Main View to 1"
        a.changeMainView(0,frames)
    if rep == ord('2'):
        print "Changing Main View to 2"
        a.changeMainView(1,frames)
    if rep == ord('3'):
        print "Changing Main View to 3"
        a.changeMainView(2,frames)
    if rep == ord('4'):
        print "Changing Main View to 4"
        a.changeMainView(3,frames)
    if rep == ord('5'):
        print "Changing Main View to 5"
        a.changeMainView(4,frames)


file.close()