import cv2
import urllib
import numpy as np
import lazy_stitcher
from math import cos, sin, pi
import threading
import Queue
import time
import os


frames_q = [Queue.LifoQueue(0), Queue.LifoQueue(0), Queue.LifoQueue(0), Queue.LifoQueue(0), Queue.LifoQueue(0)]

record_frame_q = [Queue.Queue(0), Queue.Queue(0), Queue.Queue(0), Queue.Queue(0), Queue.Queue(0), Queue.Queue(0)]

class video_record(threading.Thread):
    def __init__(self, frame1, frame2, frame3, frame4, frame5, frame_pano):
        threading.Thread.__init__(self)
        self.shape1 = (frame1.shape[1], frame1.shape[0])
        self.shape2 = (frame2.shape[1], frame2.shape[0])
        self.shape3 = (frame3.shape[1], frame3.shape[0])
        self.shape4 = (frame4.shape[1], frame4.shape[0])
        self.shape5 = (frame5.shape[1], frame5.shape[0])
        self.shape_pano = (frame_pano.shape[1], frame_pano.shape[0])
        self.is_end = False

#    def run(self):
#        global record_frame_q

#        fourcc = cv2.VideoWriter_fourcc(*'XVID')
#        out1 = cv2.VideoWriter('m.avi',fourcc, 20.0, self.shape1)
#        out2 = cv2.VideoWriter('s1.avi',fourcc, 20.0, self.shape2)
#        out3 = cv2.VideoWriter('s2.avi',fourcc, 20.0, self.shape3)
#        out4 = cv2.VideoWriter('s3.avi',fourcc, 20.0, self.shape4)
#        out5 = cv2.VideoWriter('s4.avi',fourcc, 20.0, self.shape5)
#        out_pano = cv2.VideoWriter('pano.avi',fourcc, 20.0, self.shape_pano)
##       while not self.is_end:
#            if (not record_frame_q[0].empty()):
#                out1.write(record_frame_q[0].get())
#            if (not record_frame_q[1].empty()):
#                out2.write(record_frame_q[1].get())
#            if (not record_frame_q[2].empty()):
#                out3.write(record_frame_q[2].get())
#            if (not record_frame_q[3].empty()):
#                out4.write(record_frame_q[3].get())
#            if (not record_frame_q[4].empty()):
#                out5.write(record_frame_q[4].get())
#            if (not record_frame_q[5].empty()):
#                out_pano.write(record_frame_q[5].get())

#        out1.release()
#        out2.release()
#        out3.release()
#        out4.release()
#        out5.release()
#        out_pano.release()


class image_grabber(threading.Thread):
    def __init__(self, cap_address, idx):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(cap_address)
        self.idx = idx;

    def run(self):
        global frames_q

        while True:
            #t = time.time()
            try:
                _, frame = self.cap.read()
            except:
                os._exit(1)

            # change resize ratio here to resize pano size
            frame = cv2.resize(frame,None,fx=0.9, fy=0.9, interpolation = cv2.INTER_CUBIC)
            frame = (frame < 255).astype('uint8') * (frame + 1) + (frame == 255).astype('uint8') * 255
            frames_q[self.idx].put(frame)
            #print('read frames ' + str(time.time()-t))

            time.sleep(0.01)

class Main(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def calibrate(self, frames_queue):
        flag = [0,0,0,0,0]
        while (flag[0] + flag[1] + flag[2] + flag[3] + flag[4] < 5):
            print "waiting"
            if (not frames_queue[0].empty()):
                self.main_view_frame = frames_queue[0].get()
                #cv2.imshow('main', self.main_view_frame)
                #cv2.waitKey(0)
                flag[0] = 1
            if (not frames_queue[1].empty()):
                self.side_view_frame_1 = frames_queue[1].get()
                #cv2.imshow('1', self.side_view_frame_1)
                #cv2.waitKey(0)
                flag[1] = 1
            if (not frames_queue[2].empty()):
                self.side_view_frame_2 = frames_queue[2].get()
                #cv2.imshow('2', self.side_view_frame_2)
                #cv2.waitKey(0)
                flag[2] = 1
            if (not frames_queue[3].empty()):
                self.side_view_frame_3 = frames_queue[3].get()
                #cv2.imshow('3', self.side_view_frame_3)
                #cv2.waitKey(0)
                flag[3] = 1
            if (not frames_queue[4].empty()):
                self.side_view_frame_4 = frames_queue[4].get()
                #cv2.imshow('4', self.side_view_frame_4)
                #cv2.waitKey(0)
                flag[4] = 1

            if (flag[0] + flag[1] + flag[2] + flag[3] + flag[4] < 5):
                time.sleep(0.02)

        time_begin = time.time()
        stitcher = lazy_stitcher.lazy_stitcher(self.main_view_frame, [self.side_view_frame_1, self.side_view_frame_2, self.side_view_frame_3, self.side_view_frame_4])
        time_end = time.time()
        print('Calibration time: ' + str(time_end - time_begin) + ' s')
        return stitcher

    def run(self):
        global frames_q

        stitcher = self.calibrate(frames_q)
        record_thread = video_record(self.main_view_frame, self.side_view_frame_1, self.side_view_frame_2, self.side_view_frame_3, self.side_view_frame_4, stitcher.final_pano)
        record_thread.start()

        rotate_angle = 0.
        pre_translation = np.mat([[1,0,-stitcher.final_pano.shape[1]/2], [0,1,-stitcher.final_pano.shape[0]/2], [0,0,1]], np.float)
        post_translation = np.mat([[1,0,stitcher.final_pano.shape[1]/2], [0,1,stitcher.final_pano.shape[0]/2], [0,0,1]], np.float)

        while True:

            time_begin = time.time()

            dequeue_flag = False

            if (not frames_q[0].empty()):
                self.main_view_frame = frames_q[0].get()
                dequeue_flag = True
                while (not frames_q[0].empty()):
                    frames_q[0].get()
            if (not frames_q[1].empty()):
                self.side_view_frame_1 = frames_q[1].get()
                dequeue_flag = True
                while (not frames_q[1].empty()):
                    frames_q[1].get()
            if (not frames_q[2].empty()):
                self.side_view_frame_2 = frames_q[2].get()
                dequeue_flag = True
                while (not frames_q[2].empty()):
                    frames_q[2].get()
            if (not frames_q[3].empty()):
                self.side_view_frame_3 = frames_q[3].get()
                dequeue_flag = True
                while (not frames_q[3].empty()):
                    frames_q[3].get()
            if (not frames_q[4].empty()):
                self.side_view_frame_4 = frames_q[4].get()
                dequeue_flag = True
                while (not frames_q[4].empty()):
                    frames_q[4].get()

            if not dequeue_flag:
                continue

            rotation_matrix = np.mat([[cos(rotate_angle/180.*pi), -sin(rotate_angle/180.*pi), 0], [sin(rotate_angle/180.*pi), cos(rotate_angle/180.*pi), 0], [0,0,1]], np.float)

            #try:
            pano = stitcher.stitch(self.main_view_frame, [self.side_view_frame_1, self.side_view_frame_2, self.side_view_frame_3, self.side_view_frame_4])
            #except:
                #print 'error1'
                #os._exit(1)
            #pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
            if (not (rotate_angle == 0)):
                pano = cv2.warpPerspective(pano, np.dot(post_translation, np.dot(rotation_matrix, pre_translation)), (pano.shape[1], pano.shape[0]))

            cv2.imshow('pano', pano)
            record_frame_q[0].put(self.main_view_frame)
            record_frame_q[1].put(self.side_view_frame_1)
            record_frame_q[2].put(self.side_view_frame_2)
            record_frame_q[3].put(self.side_view_frame_3)
            record_frame_q[4].put(self.side_view_frame_4)
            record_frame_q[5].put(pano)

            time_end = time.time()
            print(1/(time_end - time_begin))

            rep = cv2.waitKey(1)


            if rep == ord('q'):
                record_thread.is_end = True
                record_thread.join()
                os._exit(1)
            if rep == ord('r'):
                stitcher = self.calibrate(frames_q)

            if rep == ord('0') or rep == ord('1') or rep == ord('2') or rep == ord('3') or rep == ord('4'):
                print int(rep) - 48
                stitcher.top_view = int(rep) - 48

            if rep == ord(','):
                if rotate_angle == 0:
                    rotate_angle = 350.
                else:
                    rotate_angle = rotate_angle - 10.
            if rep == ord('.'):
                if rotate_angle == 350:
                    rotate_angle = 0.
                else:
                    rotate_angle = rotate_angle + 10.


addr1 = 'http://10.42.0.105:8050/?action=stream'
addr2 = 'http://10.42.0.101:8010/?action=stream'
addr3 = 'http://10.42.0.104:8040/?action=stream'
addr4 = 'http://10.42.0.103:8030/?action=stream'
addr5 = 'http://10.42.0.102:8020/?action=stream'

#addr1 = 'Sample/m.avi'
#addr2 = 'Sample/s1.avi'
#addr3 = 'Sample/s2.avi'
#addr4 = 'Sample/s3.avi'
#addr5 = 'Sample/s4.avi'

grabber0 = image_grabber(addr1, 0)
grabber1 = image_grabber(addr2, 1)
grabber2 = image_grabber(addr3, 2)
grabber3 = image_grabber(addr4, 3)
grabber4 = image_grabber(addr5, 4)

main = Main()

grabber0.start()
grabber1.start()
grabber2.start()
grabber3.start()
grabber4.start()

main.start()
main.join()

grabber0.join()
grabber1.join()
grabber2.join()
grabber3.join()
grabber4.join()

####
