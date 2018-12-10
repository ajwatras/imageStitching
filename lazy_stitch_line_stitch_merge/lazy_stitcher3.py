import cv2
import urllib
import numpy as np
import stitcher
import time
from numpy.linalg import inv
import line_align as la
from ulti import kmean
from ulti import find_pairs


class lazy_stitcher:
    # Lazy Stitcher serves to perform all operations required for video stitching, and panorama generation. 
    def __init__(self, main_view_cap, side_view_caps):
        # Initialize and calibrate the lazy stitcher.
        self.main_view_cap = main_view_cap
        self.side_view_caps = side_view_caps
        self.num_side_cams = len(self.side_view_caps)

        self.calibrate(main_view_cap,side_view_caps)
        caps = [[]] * (self.num_side_cams + 1)
        caps[0] = main_view_cap
        for i in range(self.num_side_cams):
            caps[i+1] = side_view_caps[i]

        ret, self.background_models = la.modelBackground(caps,1)

    def calibrate(self, main_view_cap,side_view_caps):
        # Loads default stitcher from stitcher.py
        sti = stitcher.Stitcher();

        ##################################################### Read Calibration Frames ####################################################################
        ret, main_view_frame = self.main_view_cap.read()
        side_view_frames = [[]] * self.num_side_cams
        for i in range(self.num_side_cams):
            _, side_view_frames[i] = self.side_view_caps[i].read()        

        ##################################################### Store image Shapes ##########################################################################
        main_view_cal_image = main_view_frame
        self.main_view_image_shape = main_view_cal_image.shape

        side_view_cal_images = side_view_frames
        self.side_view_image_shape = [];
        for i in range(len(side_view_frames)):
            self.side_view_image_shape.append(side_view_cal_images[i].shape)

        ############################################## Compute Initial Homographies ##########################################################################
        self.homography_list = [];
        self.coord_shift_list = [];
        self.fundamental_matrices_list = [];
        for i in range(len(side_view_frames)):
            (_, _, H, _, _, coord_shift) = sti.stitch([main_view_cal_image, side_view_cal_images[i]], showMatches=True)

            self.homography_list.append(H)
            self.coord_shift_list.append(coord_shift)
            self.fundamental_matrices_list.append(la.calcF(main_view_cal_image, side_view_cal_images[i],i+1))
            #self.fundamental_matrices_list.append(la.calcF(main_view_cal_image, side_view_cal_images[i]))

        ############################################## Initial Pano ##########################################################################
        seams_main_view_list = [];
        transformed_image_shapes_list = [];
        trans_matrices_list = [];
        shift_list = [];
        pano = np.zeros((5000, 5000, 3), np.uint8)
        out_pos = np.array([2500-self.main_view_image_shape[0]/2,2500-self.main_view_image_shape[1]/2]).astype('int')
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

        ############################################## Coordinate Shifts and Pano Shape ##############################################################################
        pts = np.nonzero(pano)
        pts = np.asarray(pts)
        left_most = np.min(pts[1,:])-1
        right_most = np.max(pts[1,:])+1
        up_most = np.min(pts[0,:])-1
        down_most = np.max(pts[0,:])+1
        self.main_view_upleft_coord = [out_pos[0] - up_most, out_pos[1] - left_most]
        self.final_pano = np.zeros((pano[up_most:down_most, left_most:right_most, :]).shape, np.uint8)

        ############################################## View Masks #####################################################################################################
        self.transformed_mask_side_view = [];
        self.masks_side_view = [];
        kernel_opening_closing = np.ones((5,5),np.uint8)
        for i in range(len(side_view_frames)):
            transformed_mask = pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_image_shapes_list[i][0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_image_shapes_list[i][1], :]
            transformed_mask = cv2.morphologyEx((transformed_mask == (i + 2)).astype('uint8'), cv2.MORPH_OPEN, kernel_opening_closing)
            self.transformed_mask_side_view.append(transformed_mask)
            self.masks_side_view.append(cv2.warpPerspective(self.transformed_mask_side_view[i], inv(trans_matrices_list[i]), (self.side_view_image_shape[i][1], self.side_view_image_shape[i][0])))

        ############################################## Seam Locations ##################################################################################################
        self.seam = np.zeros((self.main_view_image_shape[0], self.main_view_image_shape[1]))
        kernel = np.ones((50,50),np.uint8)
        for i in range(len(side_view_frames), 0, -1):
            self.seam = self.seam + (self.seam == 0) * (cv2.dilate(seams_main_view_list[i-1], kernel, iterations = 1) * i)

        self.crossing_edges_main_view_list = [];
        self.crossing_edges_side_views_list = [];
        kernel_gradient = np.mat([[0, 1, 0],[1, -4, 1],[0, 1, 0]])
        #Added by Alex
        #crossing_edge = np.ones((self.main_view_image_shape[0],self.main_view_image_shape[1]))
        #crossing_edge[1:crossing_edge.shape[0]-1,1:crossing_edge.shape[1]-1] = np.zeros((self.main_view_image_shape[0] - 2, self.main_view_image_shape[1] - 2))
        for i in range(len(side_view_frames)):
            #crossing_edge = (self.seam == (i+1)).astype('uint8')
            #crossing_edge = cv2.filter2D(crossing_edge, -1, kernel_gradient)
            #crossing_edge = (crossing_edge > 0).astype('uint8')
            #pts = np.nonzero(crossing_edge)
            #pts = np.asarray(pts)
            #for j in range(pts.shape[1]):
            #    if pts[0,j] + 24 > self.main_view_image_shape[0] or pts[0,j] - 24 < 0 or pts[1,j] + 24 > self.main_view_image_shape[1] or pts[1,j] - 24 < 0:
            #        crossing_edge[pts[0,j],pts[1,j]] = 0
            temp_seam_main_view = sti.mapMainView(self.homography_list[i],main_view_frame,side_view_frames[i])
            self.crossing_edges_main_view_list.append(temp_seam_main_view)

            #temp_seam_side_view = np.zeros(transformed_image_shapes_list[i][0:2])
            #temp_seam_side_view[shift_list[i][1]:shift_list[i][1]+self.main_view_image_shape[0], shift_list[i][0]:shift_list[i][0]+self.main_view_image_shape[1]] = crossing_edge #(temp_seam == (i+1)).astype('uint8')
            #temp_seam_side_view = cv2.warpPerspective(temp_seam_side_view, inv(trans_matrices_list[i]), (self.side_view_image_shape[i][1], self.side_view_image_shape[i][0]), flags=cv2.INTER_CUBIC)
            #pts = np.nonzero(temp_seam_side_view)
            #pts = np.asarray(pts)
            #for j in range(pts.shape[1]):
            #    if pts[0,j] + 24 > self.side_view_image_shape[i][0] or pts[0,j] - 24 < 0 or pts[1,j] + 24 > self.side_view_image_shape[i][1] or pts[1,j] - 24 < 0:
            #        temp_seam_side_view[pts[0,j],pts[1,j]] = 0

            #self.crossing_edges_side_views_list.append(temp_seam_side_view)


            #Added by Alex
            #print i, self.homography_list[i]
            temp_seam_side_view = sti.mapSeams(self.homography_list[i],main_view_frame,side_view_frames[i])
            self.crossing_edges_side_views_list.append(temp_seam_side_view)

        ##############################################
        self.diff_buffer = [];
        self.buffer_current_idx = [];
        for i in range(len(side_view_frames)):
            self.diff_buffer.append(np.zeros((self.side_view_image_shape[i][0], self.side_view_image_shape[i][1], 2), np.int))
            self.buffer_current_idx.append(False)
            self.diff_buffer[i][:,:,int(self.buffer_current_idx[i])] = side_view_cal_images[i][:,:,1];

        self.diff_buffer.append(np.zeros((self.main_view_image_shape[0], self.main_view_image_shape[1], 2), np.int))
        self.buffer_current_idx.append(False)
        self.diff_buffer[len(side_view_frames)][:,:,int(self.buffer_current_idx[len(side_view_frames)])] = main_view_cal_image[:,:,1];

        ##############################################
        self.background_models = []
        self.intensity_weights = [[]] * (len(side_view_frames) + 1)
        self.object_texture = []
        self.object_loc = []

        ##############################################


    def read_next_frame(self, main_view_frame, side_view_frames):
        intensity_diff_threshold = 20;
        pixel_diff_threshold = 30;
        pixel_seam_diff_threshold = 2;

        for i in range(len(side_view_frames)):
            side_view_frames[i] = side_view_frames[i] + 1

        side_view_has_motion = [];
        for i in range(len(side_view_frames)):
            motion_detect = ((np.absolute(self.diff_buffer[i][:,:,int(self.buffer_current_idx[i])].astype('int') - side_view_frames[i][:,:,1]).astype('int')) >= intensity_diff_threshold).astype('int') * self.masks_side_view[i][:,:,0]
            side_view_has_motion.append(np.sum(motion_detect) > pixel_diff_threshold)
            if side_view_has_motion[i]:
                self.buffer_current_idx[i] = not self.buffer_current_idx[i]
                self.diff_buffer[i][:,:,int(self.buffer_current_idx[i])] = side_view_frames[i][:,:,1];

        seam_has_motion = [];
        motion_detect = (np.absolute(self.diff_buffer[len(side_view_frames)][:,:,int(self.buffer_current_idx[len(side_view_frames)])] - main_view_frame[:,:,1]) >= intensity_diff_threshold).astype('uint8') * self.seam
        self.buffer_current_idx[len(side_view_frames)] = not self.buffer_current_idx[len(side_view_frames)]
        self.diff_buffer[len(side_view_frames)][:,:,int(self.buffer_current_idx[len(side_view_frames)])] = main_view_frame[:,:,1];
        for i in range(len(side_view_frames)):
            seam_has_motion.append(np.sum(motion_detect == (i+1)).astype('uint8') >= pixel_seam_diff_threshold)

        return side_view_has_motion, seam_has_motion


    def stitch(self,main_view_frame,side_view_frames,models):
        out_pos = self.main_view_upleft_coord
        sti = stitcher.Stitcher()
        out_pano = np.copy(self.final_pano)

        if len(np.nonzero(self.final_pano)[0]) == 0:
            self.main_mask = np.zeros(self.final_pano.shape,dtype = 'uint8')
            self.main_seam = np.zeros(self.final_pano.shape,dtype = 'uint8')
            self.main_mask[out_pos[0]:out_pos[0]+main_view_frame.shape[0],out_pos[1]:out_pos[1]+main_view_frame.shape[1]] = 1

            for i in range(len(side_view_frames)):
                result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frames[i],self.homography_list[i])
                temp_result_window = self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :]
                self.final_pano [out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = result2 * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])
                self.main_seam[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = np.maximum((result2 > 0).astype('uint8'),self.main_seam[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :])
            
            self.pano_mask = cv2.dilate(255*(np.sum(self.final_pano,2) > 0).astype('uint8'),np.ones((5,5),np.uint8),iterations=1)

            outer_edge = np.zeros(self.pano_mask.shape)
            outer_edge[0,:] = 1
            outer_edge[:,0] = 1
            outer_edge[outer_edge.shape[0]-1,:] = 1
            outer_edge[:,outer_edge.shape[1]-1] = 1
            outer_edge = outer_edge * self.pano_mask

            self.pano_mask = np.maximum(cv2.Canny(self.pano_mask,100,200)*(1-self.main_mask[:,:,0]),outer_edge)
            self.main_seam = self.main_seam * self.main_mask
            
            kernel = np.ones((5,5),np.uint8)
            self.main_seam[:,:,0] = cv2.dilate(self.main_seam[:,:,0],kernel,iterations=1)



        else:
            if (len(self.object_loc) > 2):
                 ### Perform Object detection and save timing info ###
                file = open("obj_det_timing.txt", "a")
                t = time.time()
                outer_edge = np.zeros(main_view_frame.shape)
                outer_edge[0,:] = 1
                outer_edge[:,0] = 1
                outer_edge[outer_edge.shape[0]-1,:] = 1
                outer_edge[:,outer_edge.shape[1]-1] = 1
                obj_detected,pts1,main_view_object_mask = la.genMainMask(main_view_frame,models[0],self.main_seam[out_pos[0]:out_pos[0]+main_view_frame.shape[0],out_pos[1]:out_pos[1]+main_view_frame.shape[1]]*outer_edge)

                detect_time = time.time() - t
                print "Object Detection: ", detect_time
                file.write("Object Detection: ")
                file.write(str(detect_time))
                file.write("\n")
                file.close()

                if obj_detected:
                    ### Perform Alignment and save timing info ###
                    file = open("obj_align_timing.txt","a")
                    t = time.time()

                    tempH,post_obj_mask = la.lineAlignWithModel(0,pts1.astype('int'),255*main_view_object_mask,self.object_loc,self.object_texture,self.main_seam,self.pano_mask,[out_pos[1],out_pos[0]])
                    print "Object Location: ",self.object_loc
                    print "Object Texture: ",len(self.object_texture)
                    align_time = time.time() - t 
                    print "Object_Alignment: ", align_time
                    file.write("Object Alignment: ")
                    file.write(str(align_time))
                    file.write("\n")
                    file.close()

                    ### Perform warping and save timing info ###
                    file = open("obj_warp_timing.txt","a")
                    t = time.time()
                    #tempH = la.lineAlign(pts1,main_view_frame,pts2,side_view_frame,self.fundamental_matrices_list[idx])
                    #result1,result2,mask1,new_mask, shift, trans_matrix = la.warpObject(main_view_frame, side_view_frame, side_view_object_mask, side_view_background, tempH, self.homography_list[idx], sti,result1,mask1,result2,shift, new_mask, trans_matrix)
                    #result1,result2,mask1,new_mask, shift, trans_matrix = la.warpObject(main_view_frame, side_view_frame, side_view_object_mask, tempH, self.homography_list[idx], sti,result1,mask1,result2,shift, new_mask, trans_matrix)
                    trans_obj = cv2.warpPerspective(self.object_texture,tempH,(out_pano.shape[1],out_pano.shape[0]))
                    #cv2.imshow("Obj",post_obj_mask)
                    #cv2.waitKey(0)
                    out_pano = trans_obj * post_obj_mask + out_pano * (1-post_obj_mask)
                    #result1,result2,mask1,new_mask,shift,trans_mat = stitch.applyHomography(main_frame,side_frame,np.linalg.inv(tempH))

                    warping_time = time.time() - t 
                    print "Object Warping: ", warping_time
                    file.write("Object Warping: ")
                    file.write(str(warping_time))
                    file.write("\n")
                    file.close()


            else:
                transformed_side_obj = [[]] * len(side_view_frames)
                side_view_has_motion, seam_has_motion = self.read_next_frame(main_view_frame, side_view_frames)
                print side_view_has_motion, seam_has_motion
                for i in range(len(side_view_frames)):
                    if side_view_has_motion[i]:
                        (_, transformed_side_bg,transformed_side_obj[i], _, side_mask, line_shift, _) = self.line_stitch(main_view_frame, side_view_frames[i], i,models[0],models[i+1])
                        # Comment when updating to line_align_3
                        temp_result_window = out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_bg.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_bg.shape[1], :]
                        out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_bg.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_bg.shape[1], :] = transformed_side_bg * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])

                        ## Uncomment when updating to line_align_3
                        #temp_shift = [0,0]
                        #temp_shift[0] = out_pos[0] - self.coord_shift_list[i][0] + line_shift[0]
                        #temp_shift[1] = out_pos[1] - self.coord_shift_list[i][1] + line_shift[1]
                        #self.final_pano = la.placeFrame(self.final_pano, transformed_side_view, side_mask, temp_shift )
                        #if True: #seam_has_motion[i]:
                            #self.line_stitch(main_view_frame, side_view_frames[i], i)
                    else:
                        result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frames[i],self.homography_list[i])
                        temp_result_window = out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :]
                        out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = result2 * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])


                for i in range(len(side_view_frames)):
                    if side_view_has_motion[i]:
                        if len(transformed_side_obj[i]) > 0:
                            temp_result_window = out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_obj[i].shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_obj[i].shape[1], :]
                            obj_mask = (transformed_side_obj[i] > 0).astype('uint8')
                            out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_obj[i].shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_obj[i].shape[1], :] = transformed_side_obj[i] * obj_mask + temp_result_window * (1- obj_mask)



        print out_pos
        print self.main_view_image_shape
        out_pano[out_pos[0]:out_pos[0]+self.main_view_image_shape[0], out_pos[1]:out_pos[1]+self.main_view_image_shape[1],:] = main_view_frame

        return out_pano.astype('uint8'),main_view_frame,side_view_frames


    def stitch2(self, main_view_frame, side_view_frames,models):
        transformed_side_obj = [[]] * len(side_view_frames)
        sti = stitcher.Stitcher();
        out_pos = self.main_view_upleft_coord
        side_view_has_motion, seam_has_motion = self.read_next_frame(main_view_frame, side_view_frames)
        print side_view_has_motion, seam_has_motion
        for i in range(len(side_view_frames)):
            if side_view_has_motion[i]:
                (_, transformed_side_bg,transformed_side_obj[i], _, side_mask, line_shift, _) = self.line_stitch(main_view_frame, side_view_frames[i], i,models[0],models[i+1])
                # Comment when updating to line_align_3
                temp_result_window = self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_bg.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_bg.shape[1], :]
                self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_bg.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_bg.shape[1], :] = transformed_side_bg * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])

                ## Uncomment when updating to line_align_3
                #temp_shift = [0,0]
                #temp_shift[0] = out_pos[0] - self.coord_shift_list[i][0] + line_shift[0]
                #temp_shift[1] = out_pos[1] - self.coord_shift_list[i][1] + line_shift[1]
                #self.final_pano = la.placeFrame(self.final_pano, transformed_side_view, side_mask, temp_shift )
                #if True: #seam_has_motion[i]:
                    #self.line_stitch(main_view_frame, side_view_frames[i], i)
            else:
                result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frames[i],self.homography_list[i])
                temp_result_window = self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :]
                self.final_pano [out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = result2 * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])


        for i in range(len(side_view_frames)):
            if side_view_has_motion[i]:
                if len(transformed_side_obj[i]) > 0:
                    temp_result_window = self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_obj[i].shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_obj[i].shape[1], :]
                    obj_mask = (transformed_side_obj[i] > 0).astype('uint8')
                    self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_obj[i].shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_obj[i].shape[1], :] = transformed_side_obj[i] * obj_mask + temp_result_window * (1- obj_mask)

        self.final_pano[out_pos[0]:out_pos[0]+self.main_view_image_shape[0], out_pos[1]:out_pos[1]+self.main_view_image_shape[1],:] = main_view_frame

        return self.final_pano, main_view_frame, side_view_frames

    def line_stitch(self, main_view_frame, side_view_frame, idx,main_view_background,side_view_background):
        sti = stitcher.Stitcher()
        result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frame,self.homography_list[idx])
        new_mask = (result2 > 0).astype('uint8')
        file = open("obj_det_timing.txt", "a")


        ### Perform Object detection and save timing info ###
        t = time.time()
        obj_detected,pts1,pts2,main_view_object_mask,side_view_object_mask = la.genObjMask(idx,main_view_frame, main_view_background,side_view_frame, side_view_background, self.crossing_edges_main_view_list, self.crossing_edges_side_views_list, self.homography_list)
        detect_time = time.time() - t
        print "Object Detection: ", detect_time
        file.write("Object Detection: ")
        file.write(str(detect_time))
        file.write("\n")
        file.close()

        #if np.mat(side_view_object_mask).size > 0:
        #    cv2.imshow("side mask "+str(idx),255*side_view_object_mask.astype('uint8'))
        
        bg_image = result2
        obj_image = []
        if obj_detected:
                        if len(self.object_loc) > 2:
                            ### Perform Alignment and save timing info ###
                            file = open("obj_align_timing.txt","a")
                            t = time.time()

                            main_seam, side_seam, side_border,transformed_side_border = la.genBorderMasks(main_view_frame, side_view_frame, mask1,new_mask,self.homography_list[idx],shift)
                            tempH,post_obj_mask = la.lineAlignWithModel(idx,pts1,255*main_view_object_mask,self.object_loc,main_seam,transformed_side_border,shift,self.homography_list[idx])
                            print "Object Location: ",self.object_loc
                            print "Object Texture: ",len(self.object_texture)
                            align_time = time.time() - t 
                            print "Object_Alignment: ", align_time
                            file.write("Object Alignment: ")
                            file.write(str(align_time))
                            file.write("\n")
                            file.close()


                        else:
                            ### Perform Alignment and save timing info ###
                            file = open("obj_align_timing.txt","a")
                            t = time.time()
                            # Detect Main View location in side view
                            side_view_main_mask = la.mapCoveredSide(self.homography_list[idx],main_view_frame,side_view_frame)
                            main_seam, side_seam, side_border,transformed_side_border = la.genBorderMasks(main_view_frame, side_view_frame, mask1,new_mask,self.homography_list[idx],shift)
                            #tempH = la.lineAlign(pts1,main_view_frame,pts2,side_view_frame,self.fundamental_matrices_list[idx])
                            tempH,post_obj_mask,self.object_loc = la.lineAlign(idx,pts1,255*main_view_object_mask,pts2,255*side_view_object_mask,self.fundamental_matrices_list[idx],main_seam, side_seam, side_border,transformed_side_border,shift,self.homography_list[idx])
                            self.object_texture = side_view_frame

                            align_time = time.time() - t 
                            print "Object_Alignment: ", align_time
                            file.write("Object Alignment: ")
                            file.write(str(align_time))
                            file.write("\n")
                            file.close()


                        ### Perform warping and save timing info ###
                        file = open("obj_warp_timing.txt","a")
                        t = time.time()
                        bg_image,obj_image = la.warpObject(idx,side_view_object_mask,post_obj_mask,tempH,result2,self.object_texture,side_view_background,self.homography_list[idx],shift)
                        warping_time = time.time() - t 
                        print "Object Warping: ", warping_time
                        file.write("Object Warping: ")
                        file.write(str(warping_time))
                        file.write("\n")
                        file.close()
                            
        return result1,bg_image,obj_image,mask1,new_mask, shift, trans_matrix


if __name__ == "__main__":
    cap_main = cv2.VideoCapture('output1.avi')
    cap_side = [cv2.VideoCapture('output2.avi'), cv2.VideoCapture('output3.avi'), cv2.VideoCapture('output4.avi')]
    _, main_view_frame = cap_main.read()
    _, side_view_frame_1 = cap_side[0].read()
    _, side_view_frame_2 = cap_side[1].read()
    _, side_view_frame_3 = cap_side[2].read()
    #main_view_frame = main_view_frame[2:465, 1:630, :]
    #side_view_frame_1 = side_view_frame_1[3:473, 3:633, :]
    #side_view_frame_2 = side_view_frame_2[4:478, 7:600, :]
    #side_view_frame_3 = side_view_frame_3[5:460, 3:635, :]
    print main_view_frame.shape, side_view_frame_1.shape, side_view_frame_2.shape, side_view_frame_3.shape

    a = lazy_stitcher(main_view_frame, [side_view_frame_1, side_view_frame_2, side_view_frame_3])
    while True:
        _, main_view_frame = cap_main.read()
        _, side_view_frame_1 = cap_side[0].read()
        _, side_view_frame_2 = cap_side[1].read()
        _, side_view_frame_3 = cap_side[2].read()
        #main_view_frame = main_view_frame[2:465, 1:630, :]
        #side_view_frame_1 = side_view_frame_1[3:473, 3:633, :]
        #side_view_frame_2 = side_view_frame_2[4:478, 7:600, :]
        #side_view_frame_3 = side_view_frame_3[5:460, 3:635, :]

        pano, main_view_frame, side_view_frames = a.stitch(main_view_frame, [side_view_frame_1, side_view_frame_2, side_view_frame_3])
        pano = cv2.resize(pano,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
        cv2.imshow('pano',pano)

        if cv2.waitKey(0) == ord('q'):
            exit(0)
