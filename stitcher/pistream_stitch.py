import cv2
import urllib
import numpy as np
import stitcher2

OUTPUT_SIZE = [1080,1920,3]
stitch = stitcher2.Stitcher()
H1 = np.zeros([3,3])
H2 = np.zeros([3,3])
H3 = np.zeros([3,3])
output_template = np.zeros(OUTPUT_SIZE)
output_center = np.array([OUTPUT_SIZE[0]/2,OUTPUT_SIZE[1]/2]).astype('int')


stream1=urllib.urlopen('http://10.42.0.105:8060/?action=stream')
stream2=urllib.urlopen('http://10.42.0.124:8070/?action=stream')
stream3=urllib.urlopen('http://10.42.0.104:8080/?action=stream')
stream4=urllib.urlopen('http://10.42.0.102:8090/?action=stream')

flag1 = False
flag2 = False
flag3 = False
flag4 = False

bytes1=''
bytes2=''
bytes3=''
bytes4=''


image1 = np.zeros((960,1280,3)).astype(np.uint8)
image2 = np.zeros((960,1280,3)).astype(np.uint8)
image3 = np.zeros((960,1280,3)).astype(np.uint8)
image4 = np.zeros((960,1280,3)).astype(np.uint8)
while True:
    result = np.zeros((1900,1900,3)).astype(np.uint8)

    bytes1+=stream1.read(10000)
    bytes2+=stream2.read(10000)
    bytes3+=stream3.read(10000)
    bytes4+=stream4.read(10000)

    a1 = bytes1.find('\xff\xd8')
    b1 = bytes1.find('\xff\xd9')
    a2 = bytes2.find('\xff\xd8')
    b2 = bytes2.find('\xff\xd9')
    a3 = bytes3.find('\xff\xd8')
    b3 = bytes3.find('\xff\xd9')
    a4 = bytes4.find('\xff\xd8')
    b4 = bytes4.find('\xff\xd9')


    if a1!=-1 and b1!=-1:
        flag1 = True
        jpg1 = bytes1[a1:b1+2]
        bytes1= bytes1[b1+2:]

        image1 = cv2.imdecode(np.fromstring(jpg1, dtype=np.uint8),cv2.IMREAD_COLOR)
        #cv2.imshow('stream1',image1)
        
	if cv2.waitKey(1) == ord('q'):
            exit(0)

    if a2!=-1 and b2!=-1:
        flag2 = True        
        jpg2 = bytes2[a2:b2+2]
        bytes2= bytes2[b2+2:]
        image2 = cv2.imdecode(np.fromstring(jpg2, dtype=np.uint8),cv2.IMREAD_COLOR)
        #cv2.imshow('stream2',image2)
	if cv2.waitKey(1) == ord('q'):
            exit(0)

    if a3!=-1 and b3!=-1:
        flag3 = True
        jpg3 = bytes3[a3:b3+2]
        bytes3= bytes3[b3+2:]
        image3 = cv2.imdecode(np.fromstring(jpg3, dtype=np.uint8),cv2.IMREAD_COLOR)
        #cv2.imshow('stream3',image3)
	if cv2.waitKey(1) == ord('q'):
            exit(0)

    if a4!=-1 and b4!=-1:
        flag4 = True
        jpg4 = bytes4[a4:b4+2]
        bytes4= bytes4[b4+2:]
        image4 = cv2.imdecode(np.fromstring(jpg4, dtype=np.uint8),cv2.IMREAD_COLOR)
        #cv2.imshow('stream4',image4)
	if cv2.waitKey(1) == ord('q'):
            exit(0)

    if flag1 and flag2 and flag3 and flag4:
        flag1 = False
        flag2 = False
        flag3 = False
        flag4 = False

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
            exit(0)