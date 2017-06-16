import cv2
import urllib
import numpy as np
import stitcher2
import time
from numpy.linalg import inv

OUTPUT_SIZE = [1500,1920,3]
stitch = stitcher2.Stitcher()
H1 = np.zeros([3,3])
H2 = np.zeros([3,3])
H3 = np.zeros([3,3])
output_template = np.zeros(OUTPUT_SIZE)
output_center = np.array([OUTPUT_SIZE[0]/2-300,OUTPUT_SIZE[1]/2-200]).astype('int')

CALIBRATION = True

cap3 = cv2.VideoCapture('http://10.42.0.104:8060/?action=stream')
cap1 = cv2.VideoCapture('http://10.42.0.124:8070/?action=stream')
cap2 = cv2.VideoCapture('http://10.42.0.105:8050/?action=stream')
cap4 = cv2.VideoCapture('http://10.42.0.102:8090/?action=stream')

shape = np.asarray([480, 640])
pos_flag = np.asarray([False, False, False, False])
diff_buffer = np.zeros((4, 480, 640, 2))
stitch_flag = np.asarray([True, True, True, True])
cross_flag = np.asarray([False, False, False])

result = np.zeros(OUTPUT_SIZE)

while cap1.isOpened():
        t = time.time()
        ret1, image1 = cap1.read()
        ret2, image2 = cap2.read()
        ret3, image3 = cap3.read()
        ret4, image4 = cap4.read()

        image1 = image1 + 1
        image2 = image2 + 1
        image3 = image3 + 1
        image4 = image4 + 1

        if CALIBRATION:


            (result1, vis1,H1,mask11,mask12,coord_shift1) = stitch.stitch([image1, image2], showMatches=True)
            (result2, vis2,H2,mask21,mask22,coord_shift2) = stitch.stitch([image1, image3], showMatches=True)
            (result3, vis3,H3,mask31,mask32,coord_shift3) = stitch.stitch([image1, image4], showMatches=True)

            out_pos = output_center - np.array(image1.shape[0:2])/2
            #if (H1 is not 0) and (H2 is not 0) and (H3 is not 0):
            CALIBRATION = False

            result = result.astype('uint8')
            #------------------------------------------------------------------------
            result1,result2,mask1,mask2, shift, trans_matrix = stitch.applyHomography(np.ones((shape[0], shape[1], 3)).astype('uint8'),2*np.ones((shape[0], shape[1], 3)).astype('uint8'),H1)
            seam1 = stitch.locateSeam(mask1[:,:,0],mask2[:,:,0])   # Locate the seam between the two images.
            boundry1 = seam1[shift[1]:shift[1]+shape[0], shift[0]:shift[0]+shape[1]]

            resultA = (result2*np.logical_not(mask1) + result1).astype('uint8')
            result_window = result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:] = resultA*mask2+ result_window*np.logical_not(mask2)

            shape1 = result1.shape
            mask_1_1 = mask1
            trans_matrix_1 = trans_matrix
            #------------------------------------------------------------------------


            #------------------------------------------------------------------------
            result1,result2,mask1,mask2, shift, trans_matrix = stitch.applyHomography(np.ones((shape[0], shape[1], 3)).astype('uint8'),3*np.ones((shape[0], shape[1], 3)).astype('uint8'),H2)
            seam2 = stitch.locateSeam(mask1[:,:,0],mask2[:,:,0])   # Locate the seam between the two images.
            boundry2 = seam2[shift[1]:shift[1]+shape[0], shift[0]:shift[0]+shape[1]]

            resultB = (result2*np.logical_not(mask1) + result1).astype('uint8')
            result_window = result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:] =resultB*mask2 + result_window*np.logical_not(mask2)

            shape2 = result1.shape
            mask_2_1 = mask1
            trans_matrix_2 = trans_matrix
            #------------------------------------------------------------------------


            #------------------------------------------------------------------------
            result1,result2,mask1,mask2, shift, trans_matrix = stitch.applyHomography(np.ones((shape[0], shape[1], 3)).astype('uint8'),4*np.ones((shape[0], shape[1], 3)).astype('uint8'),H3)
            seam3 = stitch.locateSeam(mask1[:,:,0],mask2[:,:,0])   # Locate the seam between the two images.
            boundry3 = seam3[shift[1]:shift[1]+shape[0], shift[0]:shift[0]+shape[1]]

            result3 = (result2*np.logical_not(mask1) + result1).astype('uint8')
            result_window = result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:] =result3*mask2 + result_window*np.logical_not(mask2)

            shape3 = result1.shape
            mask_3_1 = mask1
            trans_matrix_3 = trans_matrix
            #------------------------------------------------------------------------

            #------------------------------------------------------------------------
            result[out_pos[0]:out_pos[0]+image1.shape[0],out_pos[1]:out_pos[1]+image1.shape[1],:] =np.ones((shape[0], shape[1], 3)).astype('uint8')
            #------------------------------------------------------------------------

            boundry1 = boundry1.astype('bool')
            boundry2 = boundry2.astype('bool')
            boundry3 = boundry3.astype('bool')
            boundry1 = np.logical_and(boundry1, np.logical_and(np.logical_not(boundry2), np.logical_not(boundry3)))
            boundry2 = np.logical_and(boundry2, np.logical_not(boundry3))
            boundry = 1 * boundry1.astype('uint8') + 2 * boundry2.astype('uint8') + 3 * boundry3.astype('uint8')

            #------------------------------------------------------------------------
            mask2 = (result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+shape1[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+shape1[1],:] == 2).astype('uint8')
            mask = (mask_1_1 == 0).astype('uint8') * mask2
            mask = mask[:,:,1]
            mask_motion_detect_1 = cv2.warpPerspective(mask.astype('uint8'), inv(trans_matrix_1), (shape[1],shape[0]))

            mask2 = (result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+shape2[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+shape2[1],:] == 3).astype('uint8')
            mask = (mask_2_1 == 0).astype('uint8') * mask2
            mask = mask[:,:,1]
            mask_motion_detect_2 = cv2.warpPerspective(mask.astype('uint8'), inv(trans_matrix_2), (shape[1],shape[0]))

            mask2 = (result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+shape3[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+shape3[1],:] == 4).astype('uint8')
            mask = (mask_3_1 == 0).astype('uint8') * mask2
            mask = mask[:,:,1]
            mask_motion_detect_3 = cv2.warpPerspective(mask.astype('uint8'), inv(trans_matrix_3), (shape[1],shape[0]))


            #cv2.imshow("mask_total", result * 50)
            #cv2.imshow("mask1", mask_motion_detect_1*255)
            #cv2.imshow("mask2", mask_motion_detect_2*255)
            #cv2.imshow("mask3", mask_motion_detect_3*255)
            #cv2.waitKey(0)
            
            mask_motion_detect = np.stack((mask_motion_detect_1, mask_motion_detect_2, mask_motion_detect_3), axis = 2)




        additional_t = time.time()
        images = np.stack((image1[:,:,1], image2[:,:,1], image3[:,:,1], image4[:,:,1]), axis = 2)
        for i in range(0, 4):

            if pos_flag[i]:
                diff_buffer[i,:,:,0] = images[:,:,i]
                if stitch_flag[i]:
                    pos_flag[i] = False
            else:
                diff_buffer[i,:,:,1] = images[:,:,i]
                if stitch_flag[i]:
                    pos_flag[i] = True

            motion_detect = np.absolute(diff_buffer[i,:,:,0] - diff_buffer[i,:,:,1]) >= 20
            motion_detect = motion_detect.astype('uint8')
            if i == 0:
                boundry_motion = motion_detect * boundry
                cross1 = boundry_motion == 1
                cross2 = boundry_motion == 2
                cross3 = boundry_motion == 3
                cross_flag[0] = np.sum(cross1.astype('uint8')) > 1
                cross_flag[1] = np.sum(cross2.astype('uint8')) > 1
                cross_flag[2] = np.sum(cross3.astype('uint8')) > 1
            else:
                stitch_flag[i] = np.sum(motion_detect * mask_motion_detect[:,:,i-1]) > 30

        additional_t = time.time() - additional_t

        if not CALIBRATION:


            if stitch_flag[1]:
                result1,result2,mask1,mask2,_,_ = stitch.applyHomography(image1,image2,H1)
                resultA = (result2*np.logical_not(mask1) + result1).astype('uint8')

                result_window = result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:]
                result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:] = resultA*mask2+ result_window*np.logical_not(mask2)


            if stitch_flag[2]:
                result1,result2,mask1,mask2,_,_ = stitch.applyHomography(image1,image3,H2)
                resultB = (result2*np.logical_not(mask1) + result1).astype('uint8')

                result_window = result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:]
                result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:] =resultB*mask2 + result_window*np.logical_not(mask2)


            if stitch_flag[3]:
                result1,result2,mask1,mask2,_,_ = stitch.applyHomography(image1,image4,H3)
                result3 = (result2*np.logical_not(mask1) + result1).astype('uint8')

                result_window = result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:]
                result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:] =result3*mask2 + result_window*np.logical_not(mask2)

            out_write_coord1 = [out_pos[0] - coord_shift1[0], out_pos[1] - coord_shift1[1]]

            result[out_pos[0]:out_pos[0]+image1.shape[0],out_pos[1]:out_pos[1]+image1.shape[1],:] =image1

            result = result.astype('uint8')

            t = time.time() - t
            print stitch_flag.astype('uint8'), cross_flag.astype('uint8'), int(round(1/t*10))/10., int(round(additional_t/t*1000))/1000.

            text = np.zeros((190, 500))
            cv2.putText(text,"FPS    : %.1f" %(int(round(1/t*10))/10.), (30,50), cv2.FONT_HERSHEY_DUPLEX, 2, 255)
            cv2.putText(text,"Update : %d,%d,%d" %(stitch_flag[1].astype('uint8'), stitch_flag[2].astype('uint8'), stitch_flag[3].astype('uint8')), (30,110), cv2.FONT_HERSHEY_DUPLEX, 2, 255)
            cv2.putText(text,"Cross  : %d,%d,%d" %(cross_flag[0].astype('uint8'), cross_flag[1].astype('uint8'), cross_flag[2].astype('uint8')), (30,170), cv2.FONT_HERSHEY_DUPLEX, 2, 255)

            text = np.stack((text, text, text), axis = 2)
            result[0:190,0:500] = text
            cv2.imshow("Result",result)


            if cv2.waitKey(1) == ord('q'):
                exit(0)
