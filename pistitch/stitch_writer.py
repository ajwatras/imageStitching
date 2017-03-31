import cv2
import urllib
import numpy as np
import stitcher2

FULLSTITCH = 0
OUTPUT_SIZE = [1920,1920,3]
VIDEOWRITER_OUTPUT_PATH = '../data/stitch_writer/'
stitch = stitcher2.Stitcher()
H1 = np.zeros([3,3])
H2 = np.zeros([3,3])
H3 = np.zeros([3,3])
output_template = np.zeros(OUTPUT_SIZE)
output_center = np.array([OUTPUT_SIZE[0]/2,OUTPUT_SIZE[1]/2]).astype('int')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out1 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output1.avi',fourcc, 20.0, (640,480))
out2 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output2.avi',fourcc, 20.0, (640,480))
out3 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output3.avi',fourcc, 20.0, (640,480))
out4 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'output4.avi',fourcc, 20.0, (640,480))
out5 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'stitched.avi',fourcc, 20.0, (1920,1920))


stream1=urllib.urlopen('http://10.42.0.105:8060/?action=stream')
stream2=urllib.urlopen('http://10.42.0.124:8070/?action=stream')
stream3=urllib.urlopen('http://10.42.0.104:8080/?action=stream')
stream4=urllib.urlopen('http://10.42.0.102:8090/?action=stream')

CALIBRATION = True
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
        if CALIBRATION:
            (result1, vis1,H1,mask11,mask12,coord_shift1) = stitch.stitch([image1, image2], showMatches=True)
            (result2, vis2,H2,mask21,mask22,coord_shift2) = stitch.stitch([image1, image3], showMatches=True)
            (result3, vis3,H3,mask31,mask32,coord_shift3) = stitch.stitch([image1, image4], showMatches=True)


            out_pos = output_center - np.array(image1.shape[0:2])/2
            #if (H1 is not 0) and (H2 is not 0) and (H3 is not 0):
            CALIBRATION = False

        if not CALIBRATION:
            result = np.zeros(OUTPUT_SIZE)
            print "\nA:"
            result1,result2,mask1,mask2 = stitch.applyHomography(image1,image2,H1)
            resultA = (result2*np.logical_not(mask1) + result1).astype('uint8')
    
            result_window = result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:] = resultA*mask2+ result_window*np.logical_not(mask2)
        
            # result,fgbg1B = stitch.reStitch(result1,result2,result,fgbg1B,seam1,[out_pos[0] - coord_shift1[0],out_pos[1]-coord_shift1[1]])

        
            print "\nB:"
            result1,result2,mask1,mask2 = stitch.applyHomography(image1,image3,H2)
            resultB = (result2*np.logical_not(mask1) + result1).astype('uint8')

            print out_pos[0]-coord_shift2[0], out_pos[0]-coord_shift2[0]+result1.shape[0]
            result_window = result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:] =resultB*mask2 + result_window*np.logical_not(mask2)

            #result,fgbg2B = stitch.reStitch(result1,result2,result,fgbg2B,seam2,[out_pos[0] - coord_shift2[0],out_pos[1]-coord_shift2[1]])

            print "\nC:"
            result1,result2,mask1,mask2 = stitch.applyHomography(image1,image4,H3)
            result3 = (result2*np.logical_not(mask1) + result1).astype('uint8')
            #seam3 = stitch.locateSeam(mask1[:,:,0],mask2[:,:,0])   # Locate the seam between the two images.
    
            result_window = result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:] =result3*mask2 + result_window*np.logical_not(mask2)

            #result,fgbg3B = stitch.reStitch(result1,result2,result,fgbg3B,seam3,[out_pos[0] - coord_shift3[0],out_pos[1]-coord_shift3[1]])
        

            out_write_coord1 = [out_pos[0] - coord_shift1[0], out_pos[1] - coord_shift1[1]]
        
            #result[out_write_coord1[0]:out_write_coord1[0]+resultA.shape[0],out_write_coord1[0]:out_write_coord1[0]+resultA.shape[0],:] = resultA
            print coord_shift1

            result[out_pos[0]:out_pos[0]+image1.shape[0],out_pos[1]:out_pos[1]+image1.shape[1],:] =image1 
 


            result = result.astype('uint8')

            out1.write(image1)
            out2.write(image2)
            out3.write(image3)
            out4.write(image4)
            out5.write(result)

            cv2.imshow("Result",result)
            if cv2.waitKey(1) == ord('q'):
                exit(0)
