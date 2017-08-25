import cv2
import urllib
import numpy as np
import lazy_stitcher
from math import cos, sin, pi
import threading
import Queue
import time
import os

q1 = Queue.LifoQueue(0)
q2 = Queue.LifoQueue(0)
q3 = Queue.LifoQueue(0)
q4 = Queue.LifoQueue(0)
frames_q = [q1,q2,q4,q3]


class image_grabber(threading.Thread):
    def __init__(self, cap_address1, cap_address2, cap_address3, cap_address4):
        threading.Thread.__init__(self)
        self.cap1 = cv2.VideoCapture(cap_address1)
        self.cap2 = cv2.VideoCapture(cap_address2)
        self.cap3 = cv2.VideoCapture(cap_address3)
        self.cap4 = cv2.VideoCapture(cap_address4)

    def run(self):
        global q1
        global q2
        global q3
        global q4

        while True:

            try:
                _, frame1 = self.cap1.read()
                _, frame2 = self.cap2.read()
                _, frame3 = self.cap3.read()
                _, frame4 = self.cap4.read()
            except:
                os._exit(1)

            q1.put(frame1)
            q2.put(frame2)
            q3.put(frame3)
            q4.put(frame4)
            print "enqueue >>>", q1.qsize(), "  ", q2.qsize(), "  ", q3.qsize(), "  ", q4.qsize(), "  "

            time.sleep(0.01)

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

        stitcher = lazy_stitcher.lazy_stitcher(self.main_view_frame, [self.side_view_frame_1, self.side_view_frame_2, self.side_view_frame_3])

        return stitcher

    def run(self):
        global frames_q

        stitcher = self.calibrate(frames_q)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out1 = cv2.VideoWriter('output1.avi',fourcc, 20.0, (640,480))
        out2 = cv2.VideoWriter('output2.avi',fourcc, 20.0, (640,480))
        out3 = cv2.VideoWriter('output3.avi',fourcc, 20.0, (640,480))
        out4 = cv2.VideoWriter('output4.avi',fourcc, 20.0, (640,480))
        out_stitched = cv2.VideoWriter('stitched.avi',fourcc, 20.0, (stitcher.final_pano.shape[1],stitcher.final_pano.shape[0]))

        rotate_angle = 0.
        pre_translation = np.mat([[1,0,-stitcher.final_pano.shape[1]/2], [0,1,-stitcher.final_pano.shape[0]/2], [0,0,1]], np.float)
        post_translation = np.mat([[1,0,stitcher.final_pano.shape[1]/2], [0,1,stitcher.final_pano.shape[0]/2], [0,0,1]], np.float)

        while True:

            t_begin = time.time()
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

            if dequeue_flag:
                print "<<< dequeue"
            else:
                print "empty queue"

            rotation_matrix = np.mat([[cos(rotate_angle/180.*pi), -sin(rotate_angle/180.*pi), 0], [sin(rotate_angle/180.*pi), cos(rotate_angle/180.*pi), 0], [0,0,1]], np.float)

            try:
                pano = stitcher.stitch(self.main_view_frame, [self.side_view_frame_1, self.side_view_frame_2, self.side_view_frame_3])
            except:
                os._exit(1)
            #pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

            rotated_pano = cv2.warpPerspective(pano, np.dot(post_translation, np.dot(rotation_matrix, pre_translation)), (pano.shape[1], pano.shape[0]));
            cv2.imshow('pano',rotated_pano)

            out1.write(self.main_view_frame)
            out2.write(self.side_view_frame_1)
            out3.write(self.side_view_frame_2)
            out4.write(self.side_view_frame_3)
            out_stitched.write(rotated_pano)

            rep = cv2.waitKey(1)
            if rep == ord('q'):
                out1.release()
                out2.release()
                out3.release()
                out4.release()
                out_stitched.release()
                os._exit(1)
            if rep == ord('1'):
                frames_q = [q1,q2,q4,q3]
                stitcher = self.calibrate(frames_q)
            if rep == ord('2'):
                frames_q = [q2,q3,q1,q4]
                stitcher = self.calibrate(frames_q)
            if rep == ord('3'):
                frames_q = [q3,q4,q2,q1]
                stitcher = self.calibrate(frames_q)
            if rep == ord('4'):
                frames_q = [q4,q1,q3,q2]
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
            t_end = time.time()
            print 1/(t_end - t_begin)


addr4 = 'http://10.42.0.102:8020/?action=stream'
addr3 = 'http://10.42.0.103:8030/?action=stream'
addr2 = 'http://10.42.0.104:8040/?action=stream'
addr1 = 'http://10.42.0.101:8010/?action=stream'

grabber = image_grabber(addr1, addr2, addr3, addr4)
main = Main()

grabber.start()
main.start()
main.join()
grabber.join()



####
