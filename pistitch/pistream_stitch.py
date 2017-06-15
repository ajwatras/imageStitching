import cv2
import urllib
import numpy as np
import stitcher

OUTPUT_SIZE = [1080,1920,3]
stitch = stitcher.Stitcher()
H1 = np.zeros([3,3])
H2 = np.zeros([3,3])
H3 = np.zeros([3,3])
output_template = np.zeros(OUTPUT_SIZE)
output_center = np.array([OUTPUT_SIZE[0]/2,OUTPUT_SIZE[1]/2]).astype('int')

#cap1 = cv2.VideoCapture('http://10.42.0.105:8060/?action=stream')
#cap2 = cv2.VideoCapture('http://10.42.0.124:8070/?action=stream')
#cap3 = cv2.VideoCapture('http://10.42.0.104:8080/?action=stream')
#cap4 = cv2.VideoCapture('http://10.42.0.102:8090/?action=stream')

cap1 = cv2.VideoCapture('../data/testVideo/chopsticker2/output1.avi')
cap2 = cv2.VideoCapture('../data/testVideo/chopsticker2/output2.avi')
cap3 = cv2.VideoCapture('../data/testVideo/chopsticker2/output3.avi')
cap4 = cv2.VideoCapture('../data/testVideo/chopsticker2/output4.avi')

while cap1.isOpened():
        ret1, image1 = cap1.read()
        ret2, image2 = cap2.read()
        ret3, image3 = cap3.read()
        ret4, image4 = cap4.read()

        (result1, vis1,H,mask11,mask12,coord_shift1) = stitch.stitch([image1, image2], showMatches=True)
        (result2, vis2,H,mask21,mask22,coord_shift2) = stitch.stitch([image1, image3], showMatches=True)
        (result3, vis3,H,mask31,mask32,coord_shift3) = stitch.stitch([image1, image4], showMatches=True)

        out_pos = output_center - np.array(image1.shape[0:2])/2
    
        if result1 is not 0:
            print "result1", coord_shift1
            result_window = result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:]
            result1 = result1[0:result_window.shape[0],0:result_window.shape[1]]
            mask11 = mask11[0:result_window.shape[0],0:result_window.shape[1]]
            mask12 = mask12[0:result_window.shape[0],0:result_window.shape[1]]
    
            result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:] = result1*mask12+ result_window*np.logical_not(mask12)

        if result2 is not 0:
            print "result 2", coord_shift2
            result_window = result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result2.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result2.shape[1],:]
            result2 = result2[0:result_window.shape[0],0:result_window.shape[1]]
            mask21 = mask21[0:result_window.shape[0],0:result_window.shape[1]]
            mask22 = mask22[0:result_window.shape[0],0:result_window.shape[1]]
    
            result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result2.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result2.shape[1],:] = result2*mask22+ result_window*np.logical_not(mask22)
        
        if result3 is not 0:
            print "result 3", coord_shift3
            result_window = result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result3.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result3.shape[1],:]
            result3 = result3[0:result_window.shape[0],0:result_window.shape[1]]
            mask31 = mask31[0:result_window.shape[0],0:result_window.shape[1]]
            mask32 = mask32[0:result_window.shape[0],0:result_window.shape[1]]
    
            result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result3.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result3.shape[1],:] = result3*mask32+ result_window*np.logical_not(mask32)

        result = result.astype('uint8')

        cv2.imshow("Result",result)
        if cv2.waitKey(1) == ord('q'):
            break

cap1.release()
cap2.release()
cap3.release()
cap4.release()
