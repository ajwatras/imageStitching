import cv2
import urllib
import numpy as np
import stitcher as stitcher2
import time
OUTPUT_SIZE = [1500,1920,3]
stitch = stitcher2.Stitcher()
H1 = np.zeros([3,3])
H2 = np.zeros([3,3])
H3 = np.zeros([3,3])
output_template = np.zeros(OUTPUT_SIZE)
output_center = np.array([OUTPUT_SIZE[0]/2-400,OUTPUT_SIZE[1]/2-100]).astype('int')

CALIBRATION = True

cap3 = cv2.VideoCapture('http://10.42.0.124:8070/?action=stream')
cap1 = cv2.VideoCapture('http://10.42.0.102:8090/?action=stream')
cap2 = cv2.VideoCapture('http://10.42.0.104:8050/?action=stream')
cap4 = cv2.VideoCapture('http://10.42.0.105:8060/?action=stream')
cap5 = cv2.VideoCapture('http://10.42.0.106:8040/?action=stream')

while cap1.isOpened():
        t = time.time()
        ret1, image1 = cap1.read()
        ret2, image2 = cap2.read()
        ret3, image3 = cap3.read()
        ret4, image4 = cap4.read()
	ret5, image5 = cap5.read()

	t2 = time.time() - t


        if CALIBRATION:
            (result1, vis1,H1,mask11,mask12,coord_shift1) = stitch.stitch([image1, image2], showMatches=True)
            (result2, vis2,H2,mask21,mask22,coord_shift2) = stitch.stitch([image1, image3], showMatches=True)
            (result3, vis3,H3,mask31,mask32,coord_shift3) = stitch.stitch([image1, image4], showMatches=True)

	    for i in range(0,10):
        	cap1.grab()
        	cap2.grab()
        	cap3.grab()
        	cap4.grab()
		cap5.grab()
            out_pos = output_center - np.array(image1.shape[0:2])/2
            #if (H1 is not 0) and (H2 is not 0) and (H3 is not 0):
            CALIBRATION = False

        #result_image1 = cv2.putText(image1, 'frame %i' %(i), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        #result_image2 = cv2.putText(image2, 'frame %i' %(i), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        #result_image3 = cv2.putText(image3, 'frame %i' %(i), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        #result_image4 = cv2.putText(image4, 'frame %i' %(i), (10,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        #cv2.imshow("1", result_image1)
        #cv2.imshow("2", result_image2)
        #cv2.imshow("3", result_image3)
        #cv2.imshow("4", result_image4)

        if not CALIBRATION:
            result = np.zeros(OUTPUT_SIZE)
            #print "\nA:"
            result1,result2,mask1,mask2 = stitch.applyHomography(image1,image2,H1)
            resultA = (result2*np.logical_not(mask1) + result1).astype('uint8')
    
            result_window = result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:] = resultA*mask2+ result_window*np.logical_not(mask2)
        
            # result,fgbg1B = stitch.reStitch(result1,result2,result,fgbg1B,seam1,[out_pos[0] - coord_shift1[0],out_pos[1]-coord_shift1[1]])

        
            #print "\nB:"
            result1,result2,mask1,mask2 = stitch.applyHomography(image1,image3,H2)
            resultB = (result2*np.logical_not(mask1) + result1).astype('uint8')

            #print out_pos[0]-coord_shift2[0], out_pos[0]-coord_shift2[0]+result1.shape[0]
            result_window = result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:] =resultB*mask2 + result_window*np.logical_not(mask2)

            #result,fgbg2B = stitch.reStitch(result1,result2,result,fgbg2B,seam2,[out_pos[0] - coord_shift2[0],out_pos[1]-coord_shift2[1]])

            #print "\nC:"
            result1,result2,mask1,mask2 = stitch.applyHomography(image1,image4,H3)
            result3 = (result2*np.logical_not(mask1) + result1).astype('uint8')
            #seam3 = stitch.locateSeam(mask1[:,:,0],mask2[:,:,0])   # Locate the seam between the two images.
    
            result_window = result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:] =result3*mask2 + result_window*np.logical_not(mask2)

            #result,fgbg3B = stitch.reStitch(result1,result2,result,fgbg3B,seam3,[out_pos[0] - coord_shift3[0],out_pos[1]-coord_shift3[1]])
        

            out_write_coord1 = [out_pos[0] - coord_shift1[0], out_pos[1] - coord_shift1[1]]
        
            #result[out_write_coord1[0]:out_write_coord1[0]+resultA.shape[0],out_write_coord1[0]:out_write_coord1[0]+resultA.shape[0],:] = resultA
            #print coord_shift1

            result[out_pos[0]:out_pos[0]+image1.shape[0],out_pos[1]:out_pos[1]+image1.shape[1],:] =image1 
 
            result = result.astype('uint8')
	    t3 = time.time()
            #result = cv2.resize(result, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Result",result)
            
	    print (image5.shape[1], image5.shape[0])
            print np.mat([[-1,0,0],[0,-1,0],[0,0,1]])
            #cv2.imshow("DepthCam",image5)
            cv2.imshow("DepthCam", cv2.warpPerspective(image5,np.mat([[-1,0,image5.shape[1]],[0,-1,image5.shape[0]],[0,0,1]], np.float),(image5.shape[1],image5.shape[0])))
            #cv2.imshow("DepthCam", cv2.warpPerspective(image5,np.mat([[-1,0,0],[0,-1,0],[0,0,1]]),(image5.shape[1],image5.shape[0])))
            t3 = time.time() - t3
            if cv2.waitKey(1) == ord('q'):
                exit(0)
        t = time.time() - t

        print 1/t, t, t2, t3

cap1.release()
cap2.release()
cap3.release()
cap4.release()

