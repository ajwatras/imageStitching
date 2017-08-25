import cv2
import urllib
import numpy as np
import lazy_stitcher
from math import cos, sin, pi
import threading
import Queue
import time
import os

q = [Queue.LifoQueue(0), Queue.LifoQueue(0), Queue.LifoQueue(0), Queue.LifoQueue(0)]
frames_q = [q[0], q[1], q[3], q[2]]


class image_grabber(threading.Thread):
    def __init__(self, cap_address, idx):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(cap_address)
        self.idx = idx;

    def run(self):
        global q

        while True:
            #t = time.time()
            try:
                _, frame = self.cap.read()
            except:
                os._exit(1)

            q[self.idx].put(frame)
            #print('read frames ' + str(time.time()-t))

            time.sleep(0.02)

class Main(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def calibrate(self, frames_queue):
        flag = [0,0,0,0]
        while (flag[0] + flag[1] + flag[2] + flag[3] < 4):
            print "waiting"
            if (not frames_queue[0].empty()):
                self.main_view_frame = frames_queue[0].get()
                flag[0] = 1
            if (not frames_queue[1].empty()):
                self.side_view_frame_1 = frames_queue[1].get()
                flag[1] = 1
            if (not frames_queue[2].empty()):
                self.side_view_frame_2 = frames_queue[2].get()
                flag[2] = 1
            if (not frames_queue[3].empty()):
                self.side_view_frame_3 = frames_queue[3].get()
                flag[3] = 1

            if (flag[0] + flag[1] + flag[2] + flag[3] < 4):
                time.sleep(0.1)

        time_begin = time.time()
        stitcher = lazy_stitcher.lazy_stitcher(self.main_view_frame, [self.side_view_frame_1, self.side_view_frame_2, self.side_view_frame_3])
        time_end = time.time()
        print('Calibration time: ' + str(time_end - time_begin) + ' s')
        return stitcher

    def run(self):
        global frames_q
        print('reached1')
        stitcher = self.calibrate(frames_q)
        print('reached2')
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


            rotation_matrix = np.mat([[cos(rotate_angle/180.*pi), -sin(rotate_angle/180.*pi), 0], [sin(rotate_angle/180.*pi), cos(rotate_angle/180.*pi), 0], [0,0,1]], np.float)

            try:
                pano = stitcher.stitch(self.main_view_frame, [self.side_view_frame_1, self.side_view_frame_2, self.side_view_frame_3])
            except:
                os._exit(1)
            #pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

            if (rotate_angle != 0):
                pano = cv2.warpPerspective(pano, np.dot(post_translation, np.dot(rotation_matrix, pre_translation)), (pano.shape[1], pano.shape[0]))

            cv2.imshow('pano',pano)

            rep = cv2.waitKey(1)

            time_end = time.time()
            print(1/(time_end - time_begin))

            if rep == ord('q'):
                os._exit(1)
            if rep == ord('1'):
                frames_q = [q[0],q[1],q[3],q[2]]
                stitcher = self.calibrate(frames_q)
            if rep == ord('2'):
                frames_q = [q[1],q[2],q[0],q[3]]
                stitcher = self.calibrate(frames_q)
            if rep == ord('3'):
                frames_q = [q[2],q[3],q[1],q[0]]
                stitcher = self.calibrate(frames_q)
            if rep == ord('4'):
                frames_q = [q[3],q[0],q[2],q[1]]
                stitcher = self.calibrate(frames_q)
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


addr4 = 'http://10.42.0.104:8040/?action=stream'
addr3 = 'http://10.42.0.103:8030/?action=stream'
addr2 = 'http://10.42.0.102:8020/?action=stream'
addr1 = 'http://10.42.0.101:8010/?action=stream'


grabber0 = image_grabber(addr1, 0)
grabber1 = image_grabber(addr2, 1)
grabber2 = image_grabber(addr3, 2)
grabber3 = image_grabber(addr4, 3)
main = Main()

grabber0.start()
grabber1.start()
grabber2.start()
grabber3.start()
main.start()
main.join()
grabber0.join()
grabber1.join()
grabber2.join()
grabber3.join()


####
