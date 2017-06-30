import cv2
import urllib
import numpy as np
import stitcher
import time
from numpy.linalg import inv
import line_align as la

kernel = np.ones((50,50),np.uint8)
kernel2 = np.ones((42,42),np.uint8)
kernel3 = np.ones((10,10),np.uint8)
kernel_find_tip = np.ones((7,7),np.int8)
kernel_find_tip[3,3] = -20
kernel_opening_closing = np.ones((5,5),np.uint8)
kernel_gradient = np.ones((3,3),np.int8)
kernel_gradient[0,0] = 0;
kernel_gradient[0,2] = 0;
kernel_gradient[2,0] = 0;
kernel_gradient[2,2] = 0;
kernel_gradient[1,1] = -4;

def rotateFrame(frame, angle):
    if (angle == 0):
    # Output 2
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame,1)
    elif (angle == 1):
    # Output 3
        frame = cv2.flip(frame,0)
        frame = cv2. flip(frame,1)
    elif (angle == 2):
    # Output 4
        frame = cv2.transpose(frame)
        frame = cv2.flip(frame,0)

    return frame


def find_ref(seam1_1_cross, shift1, trans_matrix_1):
    temp = cv2.filter2D(seam1_1_cross, -1, kernel_find_tip)
    pts_tip_1_1 = np.asarray(np.nonzero((temp == np.min(temp)).astype('uint8')))
    pts_tip_1_1 = np.mat([pts_tip_1_1[1,0],pts_tip_1_1[0,0]])
    temp = np.ones((3,1), np.int); temp[0,0] = pts_tip_1_1[0,0] + shift1[0]; temp[1,0] = pts_tip_1_1[0,1] + shift1[1]
    pts_tip_1_2 = np.matmul(inv(trans_matrix_1), temp)
    pts_tip_1_2 = np.mat([pts_tip_1_2[0,0],pts_tip_1_2[1,0]])/pts_tip_1_2[2,0]
    pts_tip_1_2 = pts_tip_1_2.astype('int')
    return pts_tip_1_1, pts_tip_1_2

def line_stitch(image1, mean1, boundry1_1, seam1_1_cross, pts_tip_1_1, image2, mean2, boundry1_2, seam1_2_cross, pts_tip_1_2, F1, H1):

    result1,result2,mask1,mask2_original, shift, trans_matrix = stitch.applyHomography(image1,image2,H1)
    new_mask = (result2 > 0).astype('uint8')

    ret,thresh1 = cv2.threshold(image1[:,:,1],mean1,255,cv2.THRESH_BINARY_INV)
    thresh1 = cv2.morphologyEx(thresh1 * boundry1_1, cv2.MORPH_OPEN, kernel_opening_closing)
    ret,thresh2 = cv2.threshold(image2[:,:,1],mean2,255,cv2.THRESH_BINARY_INV)
    thresh2 = cv2.morphologyEx(thresh2 * boundry1_2, cv2.MORPH_OPEN, kernel_opening_closing)

    gradient1 = cv2.filter2D(thresh1,-1,kernel_gradient)
    pts1_mask = (gradient1 * seam1_1_cross > 0).astype('uint8')
    pts1 = np.nonzero(pts1_mask)

    gradient2 = cv2.filter2D(thresh2,-1,kernel_gradient)
    pts2_mask = (gradient2 * seam1_2_cross > 0).astype('uint8')
    pts2 = np.nonzero(pts2_mask)

    if len(pts1[0]) > 0 and len(pts1[0]) < 8 and len(pts2[0]) > 0 and len(pts2[0]) < 8:

        pts1 = np.asarray(pts1)
        for i in range(pts1.shape[1]):
            if np.sum(pts1_mask[pts1[0,i]-3:pts1[0,i]+3,pts1[1,i]-3:pts1[1,i]+3]) > 1:
                pts1_mask[pts1[0,i],pts1[1,i]] = 0
        pts1 = np.nonzero(pts1_mask)

        pts2 = np.asarray(pts2)
        for i in range(pts2.shape[1]):
            if np.sum(pts2_mask[pts2[0,i]-3:pts2[0,i]+3,pts2[1,i]-3:pts2[1,i]+3]) > 1:
                pts2_mask[pts2[0,i],pts2[1,i]] = 0
        pts2 = np.nonzero(pts2_mask)

        if len(pts1[0]) == 2 and len(pts2[0]) == 2:
            pts1 = np.asarray(pts1)
            dist1 = (pts1[1,0] - pts_tip_1_1[0,0])**2 + (pts1[0,0] - pts_tip_1_1[0,1])**2
            dist2 = (pts1[1,1] - pts_tip_1_1[0,0])**2 + (pts1[0,1] - pts_tip_1_1[0,1])**2
            if dist1 > dist2:
                pts1 = np.mat([[pts1[1,0],pts1[0,0]], [pts1[1,1],pts1[0,1]]])
            else:
                pts1 = np.mat([[pts1[1,1],pts1[0,1]], [pts1[1,0],pts1[0,0]]])

            pts2 = np.asarray(pts2)
            dist1 = (pts2[1,0] - pts_tip_1_2[0,0])**2 + (pts2[0,0] - pts_tip_1_2[0,1])**2
            dist2 = (pts2[1,1] - pts_tip_1_2[0,0])**2 + (pts2[0,1] - pts_tip_1_2[0,1])**2
            if dist1 > dist2:
                pts2 = np.mat([[pts2[1,0],pts2[0,0]], [pts2[1,1],pts2[0,1]]])
            else:
                pts2 = np.mat([[pts2[1,1],pts2[0,1]], [pts2[1,0],pts2[0,0]]])

            pts1 = np.mat([pts1[0,0], pts1[0,1]])
            pts2 = np.mat([pts2[0,0], pts2[0,1]])
            tempH = la.lineAlign(pts1,image1,pts2,image2,F1)
            result1_temp,result2_temp,_,mask2_temp, shift_temp, _ = stitch.applyHomography(image1,image2,tempH)

            mask_temp_temp = (result1_temp == 0).astype('int')
            temp = (result2_temp*mask_temp_temp + result1_temp).astype('uint8')
            #cv2.imshow('pano', temp)

            OUTPUT_SIZE = [3000,3000,3]
            out_pos = np.array([OUTPUT_SIZE[0]/2-500,OUTPUT_SIZE[1]/2-500]).astype('int')
            pano = np.zeros(OUTPUT_SIZE, np.uint8)

            window = pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp.shape[1],:]
            if window.shape[0] != mask2_temp.shape[0] or window.shape[1] != mask2_temp.shape[1]:
                return result1,result2,mask1,mask2, shift, trans_matrix

            pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp.shape[1],:] = 0+result2_temp
            result2_temp = 0+pano[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]
            #cv2.imshow('pano_2', result2_temp)
            mask = result2_temp[:,:,1] < mean2
            mask = np.stack((mask,mask,mask),axis=2)

            pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+mask2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+mask2_temp.shape[1],:] = 0+mask2_temp
            mask2_temp = 0+pano[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]

            mask = mask.astype('uint8') * mask2_temp
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening_closing)
            mask = mask * (result2 > 0).astype('uint8')

            Background_mask = (result2 > mean2).astype('int')
            color = np.sum(result2 * Background_mask) / np.sum(Background_mask)
            temp = np.ones(mask.shape, np.uint8) * color
            result2 = Background_mask.astype('uint8') * result2 + (result2 > 0).astype('uint8') * (result2 < mean2).astype('uint8') * temp

            result2 = result2 * np.logical_not(mask) + result2_temp * mask

            mask_temp_temp = (result1 == 0).astype('int')
            temp = (result2*mask_temp_temp + result1).astype('uint8')
            cv2.imshow('pano_3', temp)
            cv2.waitKey(0)
    return result1,result2,mask1, new_mask, shift, trans_matrix

OUTPUT_SIZE = [2500,2920,3]
stitch = stitcher.Stitcher()
H1 = np.zeros([3,3])
H2 = np.zeros([3,3])
H3 = np.zeros([3,3])
output_template = np.zeros(OUTPUT_SIZE)
output_center = np.array([OUTPUT_SIZE[0]/2-300,OUTPUT_SIZE[1]/2-200]).astype('int')

CALIBRATION = True

#Recorded Video
cap2 = cv2.VideoCapture('./Drop2/output2.avi')
cap1 = cv2.VideoCapture('./Drop2/output1.avi')
cap3 = cv2.VideoCapture('./Drop2/output3.avi')
cap4 = cv2.VideoCapture('./Drop2/output4.avi')

# Live Video
#cap3 = cv2.VideoCapture('http://10.42.0.124:8070/?action=stream')
#cap1 = cv2.VideoCapture('http://10.42.0.105:8080/?action=stream')
#cap2 = cv2.VideoCapture('http://10.42.0.106:8060/?action=stream')
#cap4 = cv2.VideoCapture('http://10.42.0.102:8090/?action=stream')


shape = np.asarray([480, 640])
pos_flag = np.asarray([False, False, False, False])
diff_buffer = np.zeros((4, 480, 640, 2))
stitch_flag = np.asarray([True, True, True, True])
cross_flag = np.asarray([False, False, False])

result = np.zeros(OUTPUT_SIZE)

while cap1.isOpened():
        t = time.time()
        t_read = time.time()
        ret1, image1 = cap1.read()
        ret2, image2 = cap2.read()
        ret3, image3 = cap3.read()
        ret4, image4 = cap4.read()

        image1 = image1 + 1
        image2 = image2 + 1
        image3 = image3 + 1
        image4 = image4 + 1

        #image2 = rotateFrame(image2,0)
        #image3 = rotateFrame(image3,1)
        #image4 = rotateFrame(image4,2)
        t_read = time.time() - t_read

        t_cal = time.time()
        if CALIBRATION:

            mean1 = np.mean(image1[:,:,1]) - 50
            mean2 = np.mean(image2[:,:,1]) - 50
            mean3 = np.mean(image3[:,:,1]) - 50
            mean4 = np.mean(image4[:,:,1]) - 50

            (result1, vis1,H1,mask11,mask12,coord_shift1) = stitch.stitch([image1, image2], showMatches=True)
            (result2, vis2,H2,mask21,mask22,coord_shift2) = stitch.stitch([image1, image3], showMatches=True)
            (result3, vis3,H3,mask31,mask32,coord_shift3) = stitch.stitch([image1, image4], showMatches=True)
            F1 = la.calcF(image1,image2)
            F2 = la.calcF(image1,image3)
            F3 = la.calcF(image1,image4)

            out_pos = output_center - np.array(image1.shape[0:2])/2
            #if (H1 is not 0) and (H2 is not 0) and (H3 is not 0):
            CALIBRATION = False


            result = result.astype('uint8')
            #------------------------------------------------------------------------
            result1,result2,mask1,mask2, shift, trans_matrix = stitch.applyHomography(np.ones((shape[0], shape[1], 3)).astype('uint8'),2*np.ones((shape[0], shape[1], 3)).astype('uint8'),H1)
            seam1 = stitch.locateSeam(mask1[:,:,0],mask2[:,:,0])   # Locate the seam between the two images.
            seam1_cross = seam1
            seam1 = cv2.dilate(seam1,kernel,iterations = 1) * mask1[:,:,1] * mask2[:,:,1]

            boundry1 = seam1[shift[1]:shift[1]+shape[0], shift[0]:shift[0]+shape[1]]
            #boundry1_B = cv2.warpPerspective(seam1.astype('uint8'), inv(trans_matrix), (shape[1], shape[0]))

            resultA = (result2*np.logical_not(mask1) + result1).astype('uint8')
            result_window = result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:] = resultA*mask2+ result_window*np.logical_not(mask2)

            shape1 = result1.shape
            shift1 = shift
            mask_1_1 = mask1
            trans_matrix_1 = trans_matrix
            #------------------------------------------------------------------------


            #------------------------------------------------------------------------
            result1,result2,mask1,mask2, shift, trans_matrix = stitch.applyHomography(np.ones((shape[0], shape[1], 3)).astype('uint8'),3*np.ones((shape[0], shape[1], 3)).astype('uint8'),H2)
            seam2 = stitch.locateSeam(mask1[:,:,0],mask2[:,:,0])   # Locate the seam between the two images.
            seam2_cross = seam2
            seam2 = cv2.dilate(seam2,kernel,iterations = 1) * mask1[:,:,1] * mask2[:,:,1]
            boundry2 = seam2[shift[1]:shift[1]+shape[0], shift[0]:shift[0]+shape[1]]
            #boundry2_B = cv2.warpPerspective(seam2.astype('uint8'), inv(trans_matrix), (shape[1], shape[0]))

            resultB = (result2*np.logical_not(mask1) + result1).astype('uint8')
            result_window = result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:] =resultB*mask2 + result_window*np.logical_not(mask2)

            shape2 = result1.shape
            shift2 = shift
            mask_2_1 = mask1
            trans_matrix_2 = trans_matrix
            #------------------------------------------------------------------------


            #------------------------------------------------------------------------
            result1,result2,mask1,mask2, shift, trans_matrix = stitch.applyHomography(np.ones((shape[0], shape[1], 3)).astype('uint8'),4*np.ones((shape[0], shape[1], 3)).astype('uint8'),H3)
            seam3 = stitch.locateSeam(mask1[:,:,0],mask2[:,:,0])   # Locate the seam between the two images.
            seam3_cross = seam3
            seam3 = cv2.dilate(seam3,kernel,iterations = 1) * mask1[:,:,1] * mask2[:,:,1]
            boundry3 = seam3[shift[1]:shift[1]+shape[0], shift[0]:shift[0]+shape[1]]
            #boundry3_B = cv2.warpPerspective(seam3.astype('uint8'), inv(trans_matrix), (shape[1], shape[0]))

            result3 = (result2*np.logical_not(mask1) + result1).astype('uint8')
            result_window = result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:]
            result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:] =result3*mask2 + result_window*np.logical_not(mask2)

            shape3 = result1.shape
            shift3 = shift
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

            mask_motion_detect = np.stack((mask_motion_detect_1, mask_motion_detect_2, mask_motion_detect_3), axis = 2)
            #------------------------------------------------------------------------

            #------------------------------------------------------------------------
            boundry1_1 = boundry1.astype('uint8')
            boundry1_2 = np.zeros(seam1.shape, 'uint8')
            boundry1_2[shift1[1]:shift1[1]+shape[0], shift1[0]:shift1[0]+shape[1]] = boundry1_1
            boundry1_2 = cv2.dilate(boundry1_2,kernel3,iterations = 1)
            boundry1_2 = cv2.warpPerspective(boundry1_2, inv(trans_matrix_1), (shape[1],shape[0]))

            seam1_1_cross = seam1_cross[shift1[1]:shift1[1]+shape[0], shift1[0]:shift1[0]+shape[1]]

            pts_tip_1_1, pts_tip_1_2 = find_ref(seam1_1_cross, shift1, trans_matrix_1)

            seam1_1_cross = cv2.dilate(seam1_1_cross,kernel2,iterations = 1)
            seam1_1_cross = cv2.filter2D(seam1_1_cross,-1,kernel_gradient)
            seam1_1_cross = (seam1_1_cross > 0).astype('uint8')
            pts = np.nonzero(seam1_1_cross)
            pts = np.asarray(pts)
            for i in range(pts.shape[1]):
                if pts[0,i] + 22 > shape[0] or pts[0,i] - 22 < 0 or pts[1,i] + 22 > shape[1] or pts[1,i] - 22 < 0:
                    seam1_1_cross[pts[0,i],pts[1,i]] = 0

            seam1_2_cross = cv2.warpPerspective(seam1_cross, inv(trans_matrix_1), (shape[1],shape[0]), flags=cv2.INTER_NEAREST)
            pts = np.nonzero(seam1_2_cross)
            pts = np.asarray(pts)
            for i in range(pts.shape[1]):
                seam1_2_cross[pts[0,i],pts[1,i]] = 1
                if pts[0,i] + 22 > shape[0] or pts[0,i] - 22 < 0 or pts[1,i] + 22 > shape[1] or pts[1,i] - 22 < 0:
                    seam1_2_cross[pts[0,i],pts[1,i]] = 0
            #------------------------------------------------------------------------

            #------------------------------------------------------------------------
            boundry2_1 = boundry2.astype('uint8')
            boundry2_2 = np.zeros(seam2.shape, 'uint8')
            boundry2_2[shift2[1]:shift2[1]+shape[0], shift2[0]:shift2[0]+shape[1]] = boundry2_1
            boundry2_2 = cv2.dilate(boundry2_2,kernel3,iterations = 1)
            boundry2_2 = cv2.warpPerspective(boundry2_2, inv(trans_matrix_2), (shape[1],shape[0]))

            seam2_1_cross = seam2_cross[shift2[1]:shift2[1]+shape[0], shift2[0]:shift2[0]+shape[1]]

            pts_tip_2_1, pts_tip_2_2 = find_ref(seam2_1_cross, shift2, trans_matrix_2)

            seam2_1_cross = cv2.dilate(seam2_1_cross,kernel2,iterations = 1)
            seam2_1_cross = cv2.filter2D(seam2_1_cross,-1,kernel_gradient)
            seam2_1_cross = (seam2_1_cross > 0).astype('uint8')
            pts = np.nonzero(seam2_1_cross)
            pts = np.asarray(pts)
            for i in range(pts.shape[1]):
                if pts[0,i] + 22 > shape[0] or pts[0,i] - 22 < 0 or pts[1,i] + 22 > shape[1] or pts[1,i] - 22 < 0:
                    seam2_1_cross[pts[0,i],pts[1,i]] = 0

            seam2_2_cross = cv2.warpPerspective(seam2_cross, inv(trans_matrix_2), (shape[1],shape[0]), flags=cv2.INTER_NEAREST)
            pts = np.nonzero(seam2_2_cross)
            pts = np.asarray(pts)
            for i in range(pts.shape[1]):
                seam2_2_cross[pts[0,i],pts[1,i]] = 1
                if pts[0,i] + 22 > shape[0] or pts[0,i] - 22 < 0 or pts[1,i] + 22 > shape[1] or pts[1,i] - 22 < 0:
                    seam2_2_cross[pts[0,i],pts[1,i]] = 0
            #------------------------------------------------------------------------

            #------------------------------------------------------------------------
            boundry3_1 = boundry3.astype('uint8')
            boundry3_2 = np.zeros(seam3.shape, 'uint8')
            boundry3_2[shift3[1]:shift3[1]+shape[0], shift3[0]:shift3[0]+shape[1]] = boundry3_1
            boundry3_2 = cv2.dilate(boundry3_2,kernel3,iterations = 1)
            boundry3_2 = cv2.warpPerspective(boundry3_2, inv(trans_matrix_3), (shape[1],shape[0]))

            seam3_1_cross = seam3_cross[shift3[1]:shift3[1]+shape[0], shift3[0]:shift3[0]+shape[1]]

            pts_tip_3_1, pts_tip_3_2 = find_ref(seam3_1_cross, shift3, trans_matrix_3)

            seam3_1_cross = cv2.dilate(seam3_1_cross,kernel2,iterations = 1)
            seam3_1_cross = cv2.filter2D(seam3_1_cross,-1,kernel_gradient)
            seam3_1_cross = (seam3_1_cross > 0).astype('uint8')
            pts = np.nonzero(seam3_1_cross)
            pts = np.asarray(pts)
            for i in range(pts.shape[1]):
                if pts[0,i] + 22 > shape[0] or pts[0,i] - 22 < 0 or pts[1,i] + 22 > shape[1] or pts[1,i] - 22 < 0:
                    seam3_1_cross[pts[0,i],pts[1,i]] = 0

            seam3_2_cross = cv2.warpPerspective(seam3_cross, inv(trans_matrix_3), (shape[1],shape[0]), flags=cv2.INTER_NEAREST)
            pts = np.nonzero(seam3_2_cross)
            pts = np.asarray(pts)
            for i in range(pts.shape[1]):
                seam3_2_cross[pts[0,i],pts[1,i]] = 1
                if pts[0,i] + 22 > shape[0] or pts[0,i] - 22 < 0 or pts[1,i] + 22 > shape[1] or pts[1,i] - 22 < 0:
                    seam3_2_cross[pts[0,i],pts[1,i]] = 0
            #------------------------------------------------------------------------

            seam1_1_cross = seam1_1_cross.astype('uint8')
            seam2_1_cross = seam2_1_cross.astype('uint8')
            seam3_1_cross = seam3_1_cross.astype('uint8')
            seam1_2_cross = seam1_2_cross.astype('uint8')
            seam2_2_cross = seam2_2_cross.astype('uint8')
            seam3_2_cross = seam3_2_cross.astype('uint8')

        t_cal = time.time() - t_cal


        additional_t = time.time()
        images = np.stack((image1[:,:,1], image2[:,:,1], image3[:,:,1], image4[:,:,1]), axis = 2)
        #images = np.mat([image1,image2,image3,image4])
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
            t_1 = 0; t_2 = 0; t_3 = 0; t_1_cross = 0; t_2_cross = 0; t_3_cross = 0;

            t_1 = time.time()
            if stitch_flag[1]:

                if cross_flag[0]:
                    t_1_cross = time.time()
                    result1,result2,mask1,mask2,_,_ = line_stitch(image1, mean1, boundry1_1, seam1_1_cross, pts_tip_1_1, image2, mean2, boundry1_2, seam1_2_cross, pts_tip_1_2, F1, H1);
                    t_1_cross = time.time() - t_1_cross
                else:
                    result1,result2,mask1,mask2,_,_ = stitch.applyHomography(image1,image2,H1)

                resultA = (result2*np.logical_not(mask1) + result1).astype('uint8')

                result_window = result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:]
                result[out_pos[0]-coord_shift1[0]:out_pos[0]-coord_shift1[0]+result1.shape[0],out_pos[1]-coord_shift1[1]:out_pos[1]-coord_shift1[1]+result1.shape[1],:] = resultA*mask2+ result_window*np.logical_not(mask2)

            t_1 = time.time() - t_1
            t_2 = time.time()

            if stitch_flag[2]:

                if cross_flag[1]:
                    t_2_cross = time.time()
                    result1,result2,mask1,mask2,_,_ = line_stitch(image1, mean1, boundry2_1, seam2_1_cross, pts_tip_2_1, image3, mean3, boundry2_2, seam2_2_cross, pts_tip_2_2, F2, H2);
                    t_2_cross = time.time() - t_2_cross
                else:
                    result1,result2,mask1,mask2,_,_ = stitch.applyHomography(image1,image3,H2)

                resultB = (result2*np.logical_not(mask1) + result1).astype('uint8')
                mask2 = (resultB > 0 )


                result_window = result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:]
                print result_window.shape, resultB.shape, mask2.shape
                resultB = resultB[0:result_window.shape[0],0:result_window.shape[1],:]
                mask2 = mask2[0:result_window.shape[0],0:result_window.shape[1],:]
                print result_window.shape, resultB.shape, mask2.shape
                result[out_pos[0]-coord_shift2[0]:out_pos[0]-coord_shift2[0]+result1.shape[0],out_pos[1]-coord_shift2[1]:out_pos[1]-coord_shift2[1]+result1.shape[1],:] =resultB*mask2 + result_window*np.logical_not(mask2)

            t_2 = time.time() - t_2
            t_3 = time.time()

            if stitch_flag[3]:


                if cross_flag[2]:
                    t_3_cross = time.time()
                    result1,result2,mask1,mask2,_,_ = line_stitch(image1, mean1, boundry3_1, seam3_1_cross, pts_tip_3_1, image4, mean4, boundry3_2, seam3_2_cross, pts_tip_3_2, F3, H3);
                    t_3_cross = time.time() - t_3_cross
                else:
                    result1,result2,mask1,mask2,_,_ = stitch.applyHomography(image1,image4,H3)

                result3 = (result2*np.logical_not(mask1) + result1).astype('uint8')

                result_window = result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:]
                result[out_pos[0]-coord_shift3[0]:out_pos[0]-coord_shift3[0]+result1.shape[0],out_pos[1]-coord_shift3[1]:out_pos[1]-coord_shift3[1]+result1.shape[1],:] =result3*mask2 + result_window*np.logical_not(mask2)

            t_3 = time.time() - t_3

            out_write_coord1 = [out_pos[0] - coord_shift1[0], out_pos[1] - coord_shift1[1]]

            result[out_pos[0]:out_pos[0]+image1.shape[0],out_pos[1]:out_pos[1]+image1.shape[1],:] =image1

            result = result.astype('uint8')

            t = time.time() - t
            print stitch_flag.astype('uint8'), cross_flag.astype('uint8'), int(t_cal*10000.)/10000., int(t*10000.)/10000., int(t_cal*10000.)/10000., int(t_read*10000.)/10000., int(additional_t*10000.)/10000., int(t_1*10000.)/10000., int(t_1_cross*10000.)/10000., int(t_2*10000.)/10000., int(t_2_cross*10000.)/10000., int(t_3*10000.)/10000., int(t_3_cross*10000.)/10000.


            temp = cv2.resize(result,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
            cv2.imshow("Result",temp)

            if cv2.waitKey(1) == ord('q'):
                exit(0)
