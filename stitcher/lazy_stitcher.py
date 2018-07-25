import cv2
import urllib
import numpy as np
import stitcher2
import Queue
import time
import os
import threading
from numpy.linalg import inv
import line_align_2 as la
from ulti import kmean
from ulti import find_pairs


# The warping queue serves to hold the images used for image transformations. The first column holds the main view, the second column holds the side frame, and the third column holds the transformed side view. 
warpping_queue = [[Queue.Queue(1), Queue.Queue(1), Queue.Queue(1)], [Queue.LifoQueue(1), Queue.LifoQueue(1), Queue.Queue(1)], [Queue.LifoQueue(1), Queue.LifoQueue(1), Queue.Queue(1)], [Queue.LifoQueue(1), Queue.LifoQueue(1), Queue.Queue(1)]]

class lazy_stitcher:
    # The lazy stitcher contains multithreading and selective updating methods which can significantly speed up image stitching. 
    def __init__(self, main_view_frame, side_view_frames):
        # Object constructor - Builds a lazy stitcher using a main_view_frame and a side_view_frame

        sti = stitcher2.Stitcher();                                     # Used for image stitching

        self.top_view = 0                                               # Sets camera 0 as the top image

        main_view_cal_image = main_view_frame + 1
        self.main_view_image_shape = main_view_cal_image.shape

        side_view_cal_images = side_view_frames
        for i in range(len(side_view_frames)):
            side_view_cal_images[i] = side_view_cal_images[i] + 1

        self.side_view_image_shape = [];
        for i in range(len(side_view_frames)):
            self.side_view_image_shape.append(side_view_cal_images[i].shape)

        ############################################## CALIBRATION #############################################################################################
        self.homography_list = [];                          # Store H matrices
        self.coord_shift_list = [];                         # Store coordinate shifts
        self.fundamental_matrices_list = [];

        t = time.time()
        # Calibrate cameras for stitching. 
        calibration_thread_list = [];
        for i in range(len(side_view_frames)):
             thread = calibration_thread(main_view_cal_image, side_view_cal_images[i])
             thread.start()
             calibration_thread_list.append(thread)
        # Once Calibration has finished. Store Homographies, coordinate shifts, and Fundamental Matrices (used for line alignment)
        for i in range(len(side_view_frames)):
            calibration_thread_list[i].join();
            self.homography_list.append(calibration_thread_list[i].H)
            self.coord_shift_list.append(calibration_thread_list[i].coord_shift)
            self.fundamental_matrices_list.append(calibration_thread_list[i].fundamental_matrix)

        print('compute H and F: ' + str(time.time() - t))

        ############################################## SEAM DETECTION ###########################################################################################
        seams_main_view_list = [];
        transformed_image_shapes_list = [];
        trans_matrices_list = [];
        shift_list = [];
        pano = np.zeros((10000, 10000, 3), np.uint8)
        out_pos = np.array([5000-self.main_view_image_shape[0]/2,5000-self.main_view_image_shape[1]/2]).astype('int')

        for i in range(len(side_view_frames)):
            (transformed_main_view, transformed_side_view, mask_main_view, mask_side_view, shift, trans_matrix) = sti.applyHomography(np.ones(self.main_view_image_shape, np.uint8), (i + 2) * np.ones(self.side_view_image_shape[i], np.uint8), self.homography_list[i])
            seam = sti.locateSeam(mask_main_view[:,:,0], mask_side_view[:,:,0])

            seams_main_view_list.append(seam[shift[1]:shift[1]+self.main_view_image_shape[0], shift[0]:shift[0]+self.main_view_image_shape[1]])
            transformed_image_shapes_list.append(transformed_main_view.shape)
            trans_matrices_list.append(trans_matrix)
            shift_list.append(shift)

            temp_result = (transformed_side_view * np.logical_not(mask_main_view) + transformed_main_view).astype('uint8')
            temp_result_window = pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_main_view.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_main_view.shape[1], :]
            pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_main_view.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_main_view.shape[1], :] = temp_result * mask_side_view + temp_result_window * np.logical_not(mask_side_view)

        pano[out_pos[0]:out_pos[0]+self.main_view_image_shape[0], out_pos[1]:out_pos[1]+self.main_view_image_shape[1],:] = np.ones(self.main_view_image_shape, np.uint8)

        ############################################## BOUNDARY DETECTION ########################################################################################
        pts = np.nonzero(pano)
        pts = np.asarray(pts)
        left_most = np.min(pts[1,:])-5
        right_most = np.max(pts[1,:])+5
        up_most = np.min(pts[0,:])-5
        down_most = np.max(pts[0,:])+5
        self.main_view_upleft_coord = [out_pos[0] - up_most, out_pos[1] - left_most]
        self.final_pano = np.zeros((pano[up_most:down_most, left_most:right_most, :]).shape, np.uint8)

        ############################################## MASK CALCULATION ##########################################################################################
        self.transformed_mask_side_view = [];
        self.masks_side_view = [];
        kernel_opening_closing = np.ones((5,5),np.uint8)
        for i in range(len(side_view_frames)):
            transformed_mask = pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_image_shapes_list[i][0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_image_shapes_list[i][1], :]
            transformed_mask = cv2.morphologyEx((transformed_mask == (i + 2)).astype('uint8'), cv2.MORPH_OPEN, kernel_opening_closing)
            self.transformed_mask_side_view.append(transformed_mask)
            self.masks_side_view.append(cv2.warpPerspective(self.transformed_mask_side_view[i], inv(trans_matrices_list[i]), (self.side_view_image_shape[i][1], self.side_view_image_shape[i][0])))

        # ############################################## LINE ALIGNMENT PREP ######################################################################################
        # self.seam = np.zeros((self.main_view_image_shape[0], self.main_view_image_shape[1]))
        # #temp_seam = np.zeros((self.main_view_image_shape[0], self.main_view_image_shape[1]))
        # kernel = np.ones((50,50),np.uint8)
        # for i in range(len(side_view_frames), 0, -1):
        #     self.seam = self.seam + (self.seam == 0) * (cv2.dilate(seams_main_view_list[i-1], kernel, iterations = 1) * i)
        #     #temp_seam = temp_seam + (temp_seam == 0) * (seams_main_view_list[i-1] * i)
        #
        # self.crossing_edges_main_view_list = [];
        # self.crossing_edges_side_views_list = [];
        # kernel_gradient = np.mat([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
        # for i in range(len(side_view_frames)):
        #     crossing_edge = (self.seam == (i+1)).astype('uint8')
        #     crossing_edge = cv2.filter2D(crossing_edge, -1, kernel_gradient)
        #     crossing_edge = (crossing_edge > 0).astype('uint8')
        #     pts = np.nonzero(crossing_edge)
        #     pts = np.asarray(pts)
        #     for j in range(pts.shape[1]):
        #         if pts[0,j] + 24 > self.main_view_image_shape[0] or pts[0,j] - 24 < 0 or pts[1,j] + 24 > self.main_view_image_shape[1] or pts[1,j] - 24 < 0:
        #             crossing_edge[pts[0,j],pts[1,j]] = 0
        #
        #     self.crossing_edges_main_view_list.append(crossing_edge)
        #
        #     temp_seam_side_view = np.zeros(transformed_image_shapes_list[i][0:2])
        #     temp_seam_side_view[shift_list[i][1]:shift_list[i][1]+self.main_view_image_shape[0], shift_list[i][0]:shift_list[i][0]+self.main_view_image_shape[1]] = crossing_edge #(temp_seam == (i+1)).astype('uint8')
        #     temp_seam_side_view = cv2.warpPerspective(temp_seam_side_view, inv(trans_matrices_list[i]), (self.side_view_image_shape[i][1], self.side_view_image_shape[i][0]), flags=cv2.INTER_CUBIC)
        #     pts = np.nonzero(temp_seam_side_view)
        #     pts = np.asarray(pts)
        #     for j in range(pts.shape[1]):
        #         if pts[0,j] + 24 > self.side_view_image_shape[i][0] or pts[0,j] - 24 < 0 or pts[1,j] + 24 > self.side_view_image_shape[i][1] or pts[1,j] - 24 < 0:
        #             temp_seam_side_view[pts[0,j],pts[1,j]] = 0
        #
        #     self.crossing_edges_side_views_list.append(temp_seam_side_view)
        #
        # ##############################################
        # self.diff_buffer = [];
        # self.buffer_current_idx = [];
        # for i in range(len(side_view_frames)):
        #     self.diff_buffer.append(np.zeros((self.side_view_image_shape[i][0], self.side_view_image_shape[i][1], 2), np.int))
        #     self.buffer_current_idx.append(False)
        #     self.diff_buffer[i][:,:,int(self.buffer_current_idx[i])] = side_view_cal_images[i][:,:,1];
        #
        # self.diff_buffer.append(np.zeros((self.main_view_image_shape[0], self.main_view_image_shape[1], 2), np.int))
        # self.buffer_current_idx.append(False)
        # self.diff_buffer[len(side_view_frames)][:,:,int(self.buffer_current_idx[len(side_view_frames)])] = main_view_cal_image[:,:,1];

        # Start streaming phase
        self.warpping_thread_list = [];
        for i in range(len(side_view_frames)):
            thread = warpping_thread(i, self.homography_list[i])
            thread.start();
            self.warpping_thread_list.append(thread)


    def __del__(self):
        # Object destructor - ends warpping threads once lazy_stitcher is destroyed.
        for i in range(len(self.homography_list)):
            self.warpping_thread_list[i].is_end = True;
        for i in range(len(self.homography_list)):
            self.warpping_thread_list[i].join();


    # def read_next_frame(self, main_view_frame, side_view_frames):
    #     intensity_diff_threshold = 20;
    #     pixel_diff_threshold = 30;
    #     pixel_seam_diff_threshold = 2;
    #
    #     for i in range(len(side_view_frames)):
    #         side_view_frames[i] = side_view_frames[i] + 1
    #
    #     side_view_has_motion = [];
    #     for i in range(len(side_view_frames)):
    #         motion_detect = ((np.absolute(self.diff_buffer[i][:,:,int(self.buffer_current_idx[i])].astype('int') - side_view_frames[i][:,:,1]).astype('int')) >= intensity_diff_threshold).astype('int') * self.masks_side_view[i][:,:,0]
    #         side_view_has_motion.append(np.sum(motion_detect) > pixel_diff_threshold)
    #         if side_view_has_motion[i]:
    #             self.buffer_current_idx[i] = not self.buffer_current_idx[i]
    #             self.diff_buffer[i][:,:,int(self.buffer_current_idx[i])] = side_view_frames[i][:,:,1];
    #
    #     seam_has_motion = [];
    #     motion_detect = (np.absolute(self.diff_buffer[len(side_view_frames)][:,:,int(self.buffer_current_idx[len(side_view_frames)])] - main_view_frame[:,:,1]) >= intensity_diff_threshold).astype('uint8') * self.seam
    #     self.buffer_current_idx[len(side_view_frames)] = not self.buffer_current_idx[len(side_view_frames)]
    #     self.diff_buffer[len(side_view_frames)][:,:,int(self.buffer_current_idx[len(side_view_frames)])] = main_view_frame[:,:,1];
    #     for i in range(len(side_view_frames)):
    #         seam_has_motion.append(np.sum(motion_detect == (i+1)).astype('uint8') >= pixel_seam_diff_threshold)
    #
    #     return side_view_has_motion, seam_has_motion

    def stitch(self, main_view_frame, side_view_frames):
        # Generate final panorama. 
        out_pos = self.main_view_upleft_coord

        #side_view_has_motion, seam_has_motion = self.read_next_frame(main_view_frame, side_view_frames)
        #print side_view_has_motion, seam_has_motion

        #t = time.time()
        # Gather most recent frames.
        for i in range(len(side_view_frames)):
            warpping_queue[i][0].put(main_view_frame)
            warpping_queue[i][1].put(side_view_frames[i])

        # 
        num_stitch_frames = len(side_view_frames)
        flag_finished = np.zeros(num_stitch_frames)
        while (np.sum(flag_finished) < num_stitch_frames):
            # For each side view
            for i in range(len(side_view_frames)):
                # If warping has been completed for a previous frame.
                if (not warpping_queue[i][2].empty()):
                    flag_finished[i] = 1
                    transformed_side_view = warpping_queue[i][2].get()

                    if self.top_view == i + 1:
                        top_view_frame = transformed_side_view
                    #print transformed_side_view.shape, self.transformed_mask_side_view[i].shape, self.transformed_mask_side_view[i].shape
               
                    temp_result_window = self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_view.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_view.shape[1], :]
                    
                    ## DISPLAY PARAMETERS
                    #print temp_result_window.shape
                    #print out_pos[0]-self.coord_shift_list[i][0], out_pos[0]-self.coord_shift_list[i][0]+transformed_side_view.shape[0]
                    #print out_pos[1]-self.coord_shift_list[i][1], out_pos[1]-self.coord_shift_list[i][1]+transformed_side_view.shape[1]
                    #print self.final_pano.shape
                    
                    # Add side view to panorama
                    self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_view.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_view.shape[1], :] = transformed_side_view * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])
                    #print 'complete'

        # Add main view to panorama.
        self.final_pano[out_pos[0]:out_pos[0]+self.main_view_image_shape[0], out_pos[1]:out_pos[1]+self.main_view_image_shape[1],:] = main_view_frame


        #print('stitch time: ' + str(time.time() - t))
        # Adjust for top view
        if not self.top_view == 0:
            i = self.top_view - 1
            temp_result_window = self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+top_view_frame.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+top_view_frame.shape[1], :]
            self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+top_view_frame.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+top_view_frame.shape[1], :] = top_view_frame * (top_view_frame > 0).astype('uint8') + temp_result_window * (top_view_frame == 0).astype('uint8')

        return self.final_pano



    # Legacy Code - To revisit later
    # def line_stitch(self, main_view_frame, side_view_frame, idx):
    #     # line_stitch runs stitching code with parallax correction methods. 
    #     sti = stitcher2.Stitcher()
    #     result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frame,self.homography_list[idx])
    #     new_mask = (result2 > 0).astype('uint8')
    #
    #     main_view_edge = cv2.Canny(main_view_frame, 50, 150)
    #     kernel_opening_closing = np.ones((30,30),np.uint8)
    #     main_view_object_mask = (cv2.morphologyEx(main_view_edge, cv2.MORPH_CLOSE, kernel_opening_closing)/255).astype('uint8')
    #     kernel_gradient = np.mat([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    #     main_view_edge = cv2.filter2D(main_view_object_mask,-1,kernel_gradient)
    #     main_view_edge = (main_view_edge > 0).astype('uint8')
    #
    #     pts = np.nonzero(main_view_edge * self.crossing_edges_main_view_list[idx])
    #     pts = np.asarray(pts)
    #     if pts.shape[1] >= 2:
    #         clusters_main_view, cluster_err = kmean(pts, 2)
    #         if cluster_err <= 10:
    #             side_view_edge = cv2.Canny(side_view_frame, 50, 150)
    #             side_view_object_mask = (cv2.morphologyEx(side_view_edge, cv2.MORPH_CLOSE, kernel_opening_closing)/255).astype('uint8')
    #             side_view_edge = cv2.filter2D(side_view_object_mask,-1,kernel_gradient)
    #             side_view_edge = (side_view_edge > 0).astype('uint8')
    #
    #             pts = np.nonzero(side_view_edge * self.crossing_edges_side_views_list[idx])
    #             pts = np.asarray(pts)
    #             if pts.shape[1] >= 2:
    #                 clusters_side_view, cluster_err = kmean(pts, 2)
    #                 if cluster_err <= 10:
    #                     dist_main = np.sum(np.square(clusters_main_view[:, 0] - clusters_main_view[:, 1]))
    #                     dist_side = np.sum(np.square(clusters_side_view[:, 0] - clusters_side_view[:, 1]))
    #                     if dist_main >= 100 and dist_main <= 1000 and dist_side >= 100 and dist_side <= 1000:
    #                         clusters_side_view = find_pairs(clusters_main_view, clusters_side_view, self.homography_list[idx])
    #                         ##############################
    #                         pts1 = np.mat([[clusters_main_view[1,0], clusters_main_view[0,0]], [clusters_main_view[1,1], clusters_main_view[0,1]]])
    #                         pts2 = np.mat([[clusters_side_view[1,0], clusters_side_view[0,0]], [clusters_side_view[1,1], clusters_side_view[0,1]]])
    #                         pts1 = np.mat([pts1[0,0], pts1[0,1]])
    #                         pts2 = np.mat([pts2[0,0], pts2[0,1]])
    #
    #                         tempH = la.lineAlign(pts1,main_view_frame,pts2,side_view_frame,self.fundamental_matrices_list[idx])
    #                         corners = np.mat([[0, side_view_frame.shape[1], 0, side_view_frame.shape[1]], [0, 0, side_view_frame.shape[0], side_view_frame.shape[0]], [1, 1, 1, 1]])
    #                         transformed_corners = np.dot(tempH, corners)
    #                         bound = np.zeros((2, 4), np.float)
    #                         bound[0,0] = transformed_corners[0,0] / transformed_corners[2,0]
    #                         bound[1,0] = transformed_corners[1,0] / transformed_corners[2,0]
    #                         bound[0,1] = transformed_corners[0,1] / transformed_corners[2,1]
    #                         bound[1,1] = transformed_corners[1,1] / transformed_corners[2,1]
    #                         bound[0,2] = transformed_corners[0,2] / transformed_corners[2,2]
    #                         bound[1,2] = transformed_corners[1,2] / transformed_corners[2,2]
    #                         bound[0,3] = transformed_corners[0,3] / transformed_corners[2,3]
    #                         bound[1,3] = transformed_corners[1,3] / transformed_corners[2,3]
    #
    #                         if (np.max(bound[0,:]) - np.min(bound[0,:])) <= 2500 and (np.max(bound[1,:]) - np.min(bound[1,:])) < 2500:
    #                             #############################################################################
    #                             result1_temp,result2_temp,_,mask2_temp, shift_temp, _ = sti.applyHomography(main_view_frame, side_view_frame, tempH)
    #                             mask_temp_temp = (result1_temp == 0).astype('int')
    #                             result2_temp_shape = result2_temp.shape
    #
    #                             OUTPUT_SIZE = [3000,3000,3]
    #                             out_pos = np.array([OUTPUT_SIZE[0]/2-500,OUTPUT_SIZE[1]/2-500]).astype('int')
    #                             pano = np.zeros(OUTPUT_SIZE, np.uint8)
    #
    #                             window = pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp.shape[1],:]
    #                             if window.shape[0] != mask2_temp.shape[0] or window.shape[1] != mask2_temp.shape[1]:
    #                                 return result1,result2,mask1,new_mask, shift, trans_matrix
    #
    #                             pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp.shape[1],:] = 0+result2_temp
    #                             result2_temp = 0+pano[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]
    #                             #cv2.imshow('pano_2', result2_temp)
    #                             _,mask,_,_, _, _ = sti.applyHomography(main_view_frame, np.stack((side_view_object_mask,side_view_object_mask,side_view_object_mask), axis=2), tempH)
    #
    #                             pano_mask = np.zeros(OUTPUT_SIZE, np.uint8)
    #                             pano_mask[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp_shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp_shape[1],:] = 0+mask
    #                             mask = 0+pano_mask[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]
    #
    #                             pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+mask2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+mask2_temp.shape[1],:] = 0+mask2_temp
    #                             mask2_temp = 0+pano[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]
    #
    #
    #
    #                             mask = mask.astype('uint8') * mask2_temp
    #                             #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening_closing)
    #                             mask = mask * (result2 > 0).astype('uint8')
    #                             #cv2.imshow('mask',mask*255)
    #
    #                             _,mask_object_transformed,_,_, _, _ = sti.applyHomography(main_view_frame, np.stack((side_view_object_mask,side_view_object_mask,side_view_object_mask), axis=2),self.homography_list[idx])
    #                             #cv2.imshow('mask_object_transformed',mask_object_transformed*255)
    #                             Background_mask = new_mask * (mask_object_transformed == 0).astype('uint8')
    #                             #cv2.imshow('Background_mask',Background_mask.astype('uint8')*255)
    #                             color1 = np.sum(result2[:,:,0] * Background_mask[:,:,0]) / np.sum(Background_mask[:,:,0])
    #                             color2 = np.sum(result2[:,:,1] * Background_mask[:,:,1]) / np.sum(Background_mask[:,:,1])
    #                             color3 = np.sum(result2[:,:,2] * Background_mask[:,:,2]) / np.sum(Background_mask[:,:,2])
    #                             temp = np.ones(mask.shape, np.uint8)
    #                             temp[:,:,0] = color1
    #                             temp[:,:,1] = color2
    #                             temp[:,:,2] = color3
    #
    #                             result2 = Background_mask.astype('uint8') * result2 + (result2 > 0).astype('uint8') * mask_object_transformed * temp
    #                             #cv2.imshow('result2',result2)
    #                             result2 = result2 * np.logical_not(mask) + result2_temp * mask
    #                             #cv2.imshow('result2_2',result2)
    #                             #cv2.imshow('result2 * np.logical_not(mask)',result2 * np.logical_not(mask))
    #                             #cv2.waitKey(0)
    #                             #mask_temp_temp = (result1 == 0).astype('int')
    #                             #temp = (result2*mask_temp_temp + result1).astype('uint8')
    #                             #############################################################################
    #     return result1,result2,mask1,new_mask, shift, trans_matrix

class calibration_thread(threading.Thread):
    # A thread for performing stitching calibration. When run, this thread generates the fundamental matrix F
    # used for line alignment, the homography H used for image alignment, and the coordinate shift needed to ensure that the 
    # full panorama is visible.
    def __init__(self, main_frame, side_frame):
        # The thread must be initialized with the main frame and side frame.
        threading.Thread.__init__(self)
        self.main_frame = main_frame;
        self.side_frame = side_frame;

    def run(self):
        sti = stitcher2.Stitcher();
        (_, _, H, _, _, coord_shift) = sti.stitch([self.main_frame, self.side_frame], showMatches=True)
        self.fundamental_matrix = la.calcF(self.main_frame, self.side_frame)
        self.H = H;
        self.coord_shift = coord_shift

class warpping_thread(threading.Thread):
    # The warping thread takes a set of Homographies and uses them to warp and blend together new images into a panorama. 
    def __init__(self, idx, H):
        # The thread must be initialized with a thread id which signifies a row of the warping queue (one pair of cameras), and a homography H.
        threading.Thread.__init__(self)
        self.idx = idx;                             # Which row of the warping queue is used
        self.sti = stitcher2.Stitcher();            # Stitcher object (performs stitching)
        self.H = H;                                 # Transformation to be applied
        self.is_end = False;                        # Flag to check when to stop thread

    def run(self):
        global warpping_queue;
        while True:
            if ((not warpping_queue[self.idx][0].empty()) and (not warpping_queue[self.idx][1].empty())):
                try:
                    main_frame = warpping_queue[self.idx][0].get()
                    side_frame = warpping_queue[self.idx][1].get()
                    (_, transformed_side_view, _, _, _, _) = self.sti.applyHomography(main_frame, side_frame, self.H)
                    warpping_queue[self.idx][2].put(transformed_side_view)
                    #print 'warp complete', self.idx
                except:
                    print 'warp error', self.idx
                    os._exit(1)

            if self.is_end:
                break;
