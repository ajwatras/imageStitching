import cv2
import urllib
import numpy as np
import stitcher
import time
from numpy.linalg import inv
from ulti import kmean
from ulti import find_pairs


class lazy_stitcher:
    # Lazy Stitcher serves to perform all operations required for video stitching, and panorama generation. It includes functions for running calibration
    # and stitching on video frames, as well as storing parameters that will be used from frame to frame so that they don't need to be re-computed. 


    def __init__(self, caps):
        ############################ Parameters computed during calibration #################################################################################
        self.num_side_cams = len(caps) - 1                              # The number of cameras, excluding the main view camera
        self.main_view_image_shape = []                                 # The shape of the incoming frames from the main view
        self.side_view_image_shape = []                                 # The shapes of the incoming frames from the side views
        self.homography_list =[]                                        # For each side view, the 3x3 H matrix used to align them
        self.coord_shift_list = []                                      # For each side view, the coord shift needed to align the main view with the transformed side view.
        self.fundamental_matrices_list = []                             # For each side view, the fundamental matrix F from the main view to the side view (for EpiMatch)
        self.main_view_upleft_coord = []                                # The location of the point (0,0) from the untransformed main view in the final panorama
        self.final_pano = []                                            # The canvas for the final panorama to be displayed on
        self.transformed_mask_side_view = []                            # For each side view, stores the pixel locations of that view in the final panorama
        self.masks_side_view = []                                       ### ATTENTION: Used in read_next_frame for motion detection, unsure of purpose
        self.seam = []                                                  # For each side view, a mask showing the seam location between the main view and that transformed side view 
        self.crossing_edges_main_view_list = []                         # For each side view, a mask showing the border points where the main view will overlap that side view (used in genObjMask)
        self.crossing_edges_side_views_list = []                        # For each side view, a mask showing the border points where that side view will overlap the main view (Used in genObjMask)
        self.diff_buffer = [];                                          # For each side view, stores the last frame to detect whether the new frame shows any movement.
        self.buffer_current_idx = [];                                   ### ATTENTION: Used in read_next_frame for motion detection, unsure of purpose

        self.background_models = [[]] * (self.num_side_cams + 1)                                     # For each side view, an image whose pixel values are estimates of the pure background image for that camera
        self.intensity_weights = [[]] * (self.num_side_cams + 1)        # For each camera, stores the relative shift in intensity needed to normalize intensities
        self.max_weight = [[]] * 3                                      # The average pixel value for the background after normalization. 
        self.pano_finished = False                                      # A flag for determining whether final_pano has been constructed yet.
        ############################## Parameters stored for view changing #####################################################################################
        self.view_idx = 0                                               #
        self.alt_pano = [[]] * (self.num_side_cams + 1)                 #
        self.alt_view_homographies = [[]] * (self.num_side_cams + 1)    #
        self.alt_main_up_left = [[]] * (self.num_side_cams + 1)         #
        self.alt_coord_shift_list = [[]] * (self.num_side_cams + 1)     # 
        self.alt_view_shift = [[]] * (self.num_side_cams + 1)           #
        self.alt_view_masks = [[]] * (self.num_side_cams + 1)           #
        self.alt_view_seams = [[]] * (self.num_side_cams + 1)           #
        self.alt_main_view_crossing = [[]] * (self.num_side_cams + 1)   #
        self.alt_side_view_crossing = [[]] * (self.num_side_cams + 1)   #
        ############################ Parameters computed during first self.stitch ##############################################################################
        self.pano_mask = []                                             # A binary image where 1 denotes the outer boundaries of the side views in the final mosaic
        self.main_mask = []                                             # A binary image where 1 denotes a location where the main view is located in the final mosaic
        self.main_seam = []                                             # A binary image where 1 denotes the overlapping section between the main view and any side view in the final mosaic
        ########################### Parameters computed once object is detected ################################################################################
        self.object_texture = []                                        # An image containing the object we wish to align
        self.object_loc = []                                            # A quadrilateral showing where the object is located in object_texture
        self.object_edge_intersect = [False] * 4                        # A set of four boolean values depicting whether the object intersected each of the four edges. 
        self.main_view_tracking_pts = [[]] * 4                          # A set of four points which can be used to define the main view lines of the object
        self.line1 = []                                                 # A line [rho, theta] detailing one edge of the surgical tool
        self.line2 = []                                                 # A line [rho, theta] detailing one edge of the surgical tool
        self.line1_tracking_pts = []                                    # A 2x2 matrix detailing the tracking points to be used in the line tracker. Each row is a point.
        self.line2_tracking_pts = []                                    # A 2x2 matrix detailing the tracking points to be used in the line tracker. Each row is a point.
        ########################## Parameters for timing info ###################################################################################################
        self.bg_sub_time = 0
        self.line_fit_time = 0
        self.line_match_time = 0
        self.feat_gen_time = 0

        ## Run Calibration
        self.calibrate(caps)

    def calibrate(self, caps):
        # Performs the calibration step of our visualization system (See *Paper location TBD*). 
        side_view_caps = list(caps)
        main_view_cap =  side_view_caps.pop(self.view_idx)

        # Loads default stitcher from stitcher.py
        sti = stitcher.Stitcher();

        ##################################################### Read Calibration Frames ####################################################################

        # read incoming frames and store total number of cameras
        ret, main_view_frame = main_view_cap.read()
        side_view_frames = [[]] * self.num_side_cams
        for i in range(self.num_side_cams):
            _, side_view_frames[i] = side_view_caps[i].read()        

        ##################################################### Store image Shapes ##########################################################################
        # Store the dimensions of the video coming from each of the cameras
        main_view_cal_image = main_view_frame
        self.main_view_image_shape = main_view_cal_image.shape

        side_view_cal_images = side_view_frames
        self.side_view_image_shape = [];
        for i in range(len(side_view_frames)):
            self.side_view_image_shape.append(side_view_cal_images[i].shape)

        ############################################## Compute Initial Homographies ##########################################################################
        #Homography computation is performed with SURF feature matching (See *Paper Location TBD) fed into a RANSAC (see *Paper Lovation TBD), which is used 
        # to solve for an optimal Homography (*See *Paper Location TBD). As the mat format for images can only store pixel values located in positive locations, 
        # coord_shift stores the coordinate shift required in order to ensure that all nonzero pixels will be located in positive locations after transformation.
        for i in range(len(side_view_frames)):
            (_, _, H, _, _, coord_shift) = sti.stitch([main_view_cal_image, side_view_cal_images[i]], showMatches=True)

            self.homography_list.append(H)
            self.coord_shift_list.append(coord_shift)
            self.fundamental_matrices_list.append(calcF(main_view_cal_image, side_view_cal_images[i],i+1))
            #self.fundamental_matrices_list.append(la.calcF(main_view_cal_image, side_view_cal_images[i]))

        ############################################## Coordinate shifts and Pano Shape #########################################################################
        # This is a wall of text and should be documented / Optimized / checked for legacy variables.
        seams_main_view_list = [];                                              #
        transformed_image_shapes_list = [];                                     # Saves the shape of our post transform images for each homography
        trans_matrices_list = [];                                               # Each trans_matrix is the combination of the homography with the coordinate transformation.
        shift_list = [];                                                        # the required coord shift required for each homography
        pano = np.zeros((5000, 5000, 3), np.uint8)                              # An overlarge image to place the images on for determining required panorama size.


        out_pos = np.array([2500-self.main_view_image_shape[0]/2,2500-self.main_view_image_shape[1]/2]).astype('int')   # Find the center of the panorama
        for i in range(len(side_view_frames)):
            # Apply the computed homography to generate a mask of the frame location in the final panorama.
            (transformed_main_view, transformed_side_view, mask_main_view, mask_side_view, shift, trans_matrix) = sti.applyHomography(np.ones(self.main_view_image_shape, np.uint8), (i + 2) * np.ones(self.side_view_image_shape[i], np.uint8), self.homography_list[i])

            # detect seam locations in the main view. 
            seam = sti.locateSeam(mask_main_view[:,:,0], mask_side_view[:,:,0]) #
            seams_main_view_list.append(seam[shift[1]:shift[1]+self.main_view_image_shape[0], shift[0]:shift[0]+self.main_view_image_shape[1]])
            
            #
            transformed_image_shapes_list.append(transformed_main_view.shape)
            trans_matrices_list.append(trans_matrix)
            shift_list.append(shift)

            #Place side view in panorama
            temp_result = (transformed_side_view * np.logical_not(mask_main_view) + transformed_main_view).astype('uint8')
            temp_result_window = pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_main_view.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_main_view.shape[1], :]
            pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_main_view.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_main_view.shape[1], :] = temp_result * mask_side_view + temp_result_window * np.logical_not(mask_side_view)

        # Place main view in panorama
        pano[out_pos[0]:out_pos[0]+self.main_view_image_shape[0], out_pos[1]:out_pos[1]+self.main_view_image_shape[1],:] = np.ones(self.main_view_image_shape, np.uint8)


        # determine the necessary size for our backgroung panorama.
        pts = np.nonzero(pano)
        pts = np.asarray(pts)
        left_most = np.min(pts[1,:])-1
        right_most = np.max(pts[1,:])+1
        up_most = np.min(pts[0,:])-1
        down_most = np.max(pts[0,:])+1

        

        self.main_view_upleft_coord = [out_pos[0] - up_most, out_pos[1] - left_most]                                    #The location of the main view (0,0) in the final coord system
        self.final_pano = np.zeros((pano[up_most:down_most, left_most:right_most, :]).shape, np.uint8)                  #The empty canvas that will be used for the final panorama.

        ############################################## View Masks #####################################################################################################
        # Not certain what these are used for, need to check.
        kernel_opening_closing = np.ones((5,5),np.uint8)
        for i in range(len(side_view_frames)):
            transformed_mask = pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_image_shapes_list[i][0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_image_shapes_list[i][1], :]
            transformed_mask = cv2.morphologyEx((transformed_mask == (i + 2)).astype('uint8'), cv2.MORPH_OPEN, kernel_opening_closing)
            self.transformed_mask_side_view.append(transformed_mask)
            self.masks_side_view.append(cv2.warpPerspective(self.transformed_mask_side_view[i], inv(trans_matrices_list[i]), (self.side_view_image_shape[i][1], self.side_view_image_shape[i][0])))

        ############################################## Seam Locations ##################################################################################################
        # Should be migrated to geometric seams rather than seam masks.
        self.seam = np.zeros((self.main_view_image_shape[0], self.main_view_image_shape[1]))
        kernel = np.ones((50,50),np.uint8)
        for i in range(len(side_view_frames), 0, -1):
            self.seam = self.seam + (self.seam == 0) * (cv2.dilate(seams_main_view_list[i-1], kernel, iterations = 1) * i)

        kernel_gradient = np.mat([[0, 1, 0],[1, -4, 1],[0, 1, 0]])

        for i in range(len(side_view_frames)):
            temp_seam_main_view = sti.mapMainView(self.homography_list[i],main_view_frame,side_view_frames[i])
            self.crossing_edges_main_view_list.append(temp_seam_main_view)

            temp_seam_side_view = sti.mapSeams(self.homography_list[i],main_view_frame,side_view_frames[i])
            self.crossing_edges_side_views_list.append(temp_seam_side_view)

        ##############################################
        for i in range(len(side_view_frames)):
            self.diff_buffer.append(np.zeros((self.side_view_image_shape[i][0], self.side_view_image_shape[i][1], 2), np.int))
            self.buffer_current_idx.append(False)
            self.diff_buffer[i][:,:,int(self.buffer_current_idx[i])] = side_view_cal_images[i][:,:,1];

        self.diff_buffer.append(np.zeros((self.main_view_image_shape[0], self.main_view_image_shape[1], 2), np.int))
        self.buffer_current_idx.append(False)
        self.diff_buffer[len(side_view_frames)][:,:,int(self.buffer_current_idx[len(side_view_frames)])] = main_view_cal_image[:,:,1];

        ############################################## Background Modeling ###############################################################################################
        caps = [[]] * (self.num_side_cams + 1)
        caps[0] = main_view_cap
        for i in range(self.num_side_cams):
            caps[i+1] = side_view_caps[i]

        ret, self.background_models = modelBackground(caps,1)
        ################################# Compute Intensity Correction ###################################################################################################
        # NOTE: Method is incomplete, adjust for each color channel or brightness alternate color space. 
        ret, main_view_frame = main_view_cap.read()
        side_view_frames = [[]] * self.num_side_cams
        for i in range(self.num_side_cams):
            _, side_view_frames[i] = side_view_caps[i].read() 

        ave_intensity_b = [[]] * (len(side_view_frames) + 1)
        ave_intensity_g = [[]] * (len(side_view_frames) + 1)
        ave_intensity_r = [[]] * (len(side_view_frames) + 1)
        #self.intensity_weights = [[]] * (len(side_view_frames) + 1)
        #self.max_weight = [[]] * 3

        for i in range(len(ave_intensity_b)):
            ave_intensity_b[i] = [np.mean(self.background_models[i][:,:,0])]
            ave_intensity_g[i] = [np.mean(self.background_models[i][:,:,1])]
            ave_intensity_r[i] = [np.mean(self.background_models[i][:,:,2])]


        self.max_weight[0] = np.max(ave_intensity_b)
        self.max_weight[1] = np.max(ave_intensity_g)
        self.max_weight[2] = np.max(ave_intensity_r)
        
        for i in range(len(ave_intensity_b)):
            #self.intensity_weights[i] = [ave_intensity_b[i]/self.max_weight[0],ave_intensity_g[i]/self.max_weight[1],ave_intensity_r[i]/self.max_weight[2]]
            self.intensity_weights[i] = [self.max_weight[0] - ave_intensity_b[i], self.max_weight[1] - ave_intensity_g[i],self.max_weight[2] - ave_intensity_r[i]]


        #print "Intensity Weights: ", self.intensity_weights

        ################################# Compute blending weights ######################################################################################################
        #blending_window = 25

        # Compute main view blending mask
        #main_mask = np.ones([self.main_view_image_shape[0],self.main_view_image_shape[1]]).astype('uint8')
        #main_mask = np.pad(main_mask,[(1,1),(1,1)],mode='constant')
        #dist_mask = cv2.distanceTransform(main_mask,cv2.DIST_L1,3)/blending_window
        #print "dist: ",dist_mask.shape
        #print "main: ",main_mask.shape
        #main_mask = np.minimum(main_mask,dist_mask)
        #main_mask = main_mask[1:-1,1:-1]
        #np.repeat(main_mask[:,:,np.newaxis], 3,axis=2)


        #for i in range(self.num_side_cams):
        #    side_mask = np.amax(self.transformed_mask_side_view[i],axis=2)
        #    print side_mask
        #    distance_mask = cv2.distanceTransform(side_mask.astype('uint8'),cv2.DIST_L1,3)/blending_window
        #    side_mask = np.minimum(side_mask,distance_mask)
        #    self.transformed_mask_side_view[i] = np.repeat(side_mask[:,:,np.newaxis], 3,axis=2)


        ################################# pre-Compute alternative main views ############################################################################################
        # This section precomputes 
        self.alt_view_homographies[0] = self.homography_list
        self.alt_view_shift[0] = self.coord_shift_list
        self.alt_view_masks[0] = self.transformed_mask_side_view
        self.alt_main_up_left[0] = self.main_view_upleft_coord
        self.alt_view_seams[0] = self.seam
        self.alt_main_view_crossing[0] = self.crossing_edges_main_view_list
        self.alt_side_view_crossing[0] = self.crossing_edges_side_views_list
        self.alt_pano[0] = self.final_pano


        #for i in range(self.num_side_cams):
        #    self.alt_view_homographies[i] = self.homography_list
        #    self.alt_view_shift[i] = self.coord_shift_list
        #    self.alt_view_masks[i] = self.transformed_mask_side_view
        #    self.alt_main_up_left[i] = self.main_view_upleft_coord
        #    self.alt_view_seams[i] = self.seam

        #self.changeMainView(2)

        #return 
        temp_H = []
        temp_H.append(np.eye(3))
        for i in range(len(self.homography_list)):
            temp_H.append(self.homography_list[i])

        for i in range(self.num_side_cams):
            H_list = [[]] * (self.num_side_cams + 1)
            # Generate transformation to make side view i the main view
            H_inv = np.linalg.inv(self.homography_list[i])

            # Generate new homographies
            for j in range(self.num_side_cams + 1):
                H_list[j] = np.dot(H_inv,temp_H[j])
                
            self.alt_view_homographies[i+1] = H_list
            self.alt_view_homographies[i+1].pop(i+1)



            # Compute required shift in order to keep panorama positive
            main_view_image_shape = self.side_view_image_shape[i]
            side_view_image_shape = [[]] * self.num_side_cams
            for j in range(i):
                side_view_image_shape[j] = self.side_view_image_shape[j]
            side_view_image_shape[i] = self.main_view_image_shape
            for j in range(i+1,self.num_side_cams):
                side_view_image_shape[j] = self.side_view_image_shape[j]

            self.alt_main_up_left[i+1],self.alt_view_shift[i+1], self.alt_view_masks[i+1],self.alt_main_view_crossing[i+1],self.alt_side_view_crossing[i+1],self.alt_view_seams[i+1],self.alt_pano[i+1] = computePanoShift(H_list,main_view_frame,main_view_image_shape,side_view_frames,side_view_image_shape,self.num_side_cams)



            # generate new seams


        ############ DEBUG alt views ###################################
        #frames = [[]] * (self.num_side_cams + 1)
        #for i in range(len(caps)):
        #    _,frames[i] = caps[i].read()

        #for j in range(self.num_side_cams):

        #    self.changeMainView(j,frames)
        #    side_view_frames = list(frames)
        #    main_view_frame = side_view_frames.pop(j)

        #    for i in range(len(side_view_frames)):
        #        print self.homography_list[i]
        #        result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frames[i],self.homography_list[i])
                
        #        cv2.imshow("main view",result1)
        #        cv2.imshow("side view",result2)
        #        cv2.waitKey(0)


        ##########################################################################################################################################

    def changeMainView(self,new_main_view_idx,frames):
        # Set new index
        self.view_idx = new_main_view_idx
        # reset panorama
        self.final_pano = self.alt_pano[new_main_view_idx]
        self.pano_finished = False


        # Apply changes to self.
        self.crossing_edges_main_view_list = self.alt_main_view_crossing[new_main_view_idx]
        self.crossing_edges_side_views_list = self.alt_side_view_crossing[new_main_view_idx]
        self.main_view_image_shape = frames[new_main_view_idx].shape
        self.homography_list = self.alt_view_homographies[new_main_view_idx]
        self.transformed_mask_side_view = self.alt_view_masks[new_main_view_idx]
        self.main_view_upleft_coord = self.alt_main_up_left[new_main_view_idx]
        self.seam = self.alt_view_seams[new_main_view_idx]
        self.coord_shift_list = self.alt_view_shift[new_main_view_idx]

    def correctIntensity(self,main_view_frame,side_view_frames,calibration_type = 0):
        # Corrects for the fact that the translation between light intensity and pixel values may vary between cameras
        # INPUTS:
        # main_view_frame - The camera frame associated with the main view cameras
        # side_view_frames - The list of camera frames associated with each side view camera
        # calibration_type - a type of 0 means the weight will be added to the image frame, and a type of 1 means the weight will multiply
        # intensity_weights - The intensity weights computed during calibration.
        # OUTPUTS: 
        # main_view_frame - the intensity corrected main view frame
        # side_view_frames - the intensity corrected side view frames

        frame_list = [[]] * (len(side_view_frames) + 1)
        frame_list[0] = main_view_frame
        for i in range(len(side_view_frames)):
            frame_list[i+1] = side_view_frames[i]

        if calibration_type == 0:
            for i in range(len(frame_list)):
                frame_shape = frame_list[i][:,:,0].shape
                frame_list[i][:,:,0] = (self.intensity_weights[i][0] * np.ones(frame_shape) + frame_list[i][:,:,0]).astype('uint8')
                frame_list[i][:,:,1] = (self.intensity_weights[i][1] * np.ones(frame_shape) + frame_list[i][:,:,1]).astype('uint8')
                frame_list[i][:,:,2] = (self.intensity_weights[i][2] * np.ones(frame_shape) + frame_list[i][:,:,2]).astype('uint8')
        elif calibration_type == 1:
            for i in range(len(frame_list)):
                frame_shape = frame_list[i][:,:,0].shape
                frame_list[i][:,:,0] = (self.intensity_weights[i][0]*frame_list[i][:,:,0]).astype('uint8')
                frame_list[i][:,:,1] = (self.intensity_weights[i][1]*frame_list[i][:,:,1]).astype('uint8')
                frame_list[i][:,:,2] = (self.intensity_weights[i][2]*frame_list[i][:,:,2]).astype('uint8')

        main_view_frame = frame_list[0]
        side_view_frames = frame_list[1:]

        return main_view_frame, side_view_frames

    def quantizeAlignmentError(self, side_idx, main_view, side_view):
        # Estimate line parameters
        main_view_top_line,main_view_bot_line,a,b = trackObject2(main_view, self.background_models[0],[0,0],[0,0])
        side_view_top_line,side_view_bot_line,a,b = trackObject2(side_view, self.background_models[side_idx],[0,0],[0,0])


        print "main top: "+ str(main_view_top_line)
        print "main bot: "+ str(main_view_bot_line)
        print "side top: "+ str(side_view_top_line)
        print "side bot: "+ str(side_view_bot_line)

        # Compare line parameters
        top_slope_diff = main_view_top_line[1] - side_view_top_line[1]
        top_offset_diff = main_view_top_line[0] - side_view_top_line[0]

        bot_slope_diff = main_view_bot_line[1] - side_view_bot_line[1]
        bot_offset_diff = main_view_bot_line[0] - side_view_bot_line[0]

        # Write results to file. 
        file = open("line_comparison.txt",'a')
        file.write(str(top_slope_diff) + "," + str(top_offset_diff) + ",")
        file.write(str(bot_slope_diff) + "," + str(bot_offset_diff) + "\n")
        file.close()
        
        return


    def read_next_frame(self, main_view_frame, side_view_frames):
        # Checks whether there has been sufficient motion between subsequent frames 
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


    def stitch(self,frames,models):
        # uses incoming frames to update final panorama. 
        # INPUTS: 
        #     main_view_frame - A mat object containing the frame from the main view camera, must be in format 'uint8'
        #     side_view_frames - A list, for each side view contains the frame from that side view, must be format 'uint8'
        #     models - A list, for each side view, contains the frame gathered during background modeling. must be format 'uint8'
        # OUTPUTS:
        #     out_pano - The resulting image mosaic from our visualization algorithm. 

        out_pos = self.main_view_upleft_coord                               # Note main view shift
        sti = stitcher.Stitcher()                                           # To apply transformation
        out_pano = np.copy(self.final_pano)                                 # Canvas

        side_view_frames = list(frames)
        main_view_frame =  side_view_frames.pop(self.view_idx)


        ################################### If background hasn't been drawn yet #############################################################################################################
        if not self.pano_finished:
            self.pano_finished = True                                                                                               # Mark that background is being drawn
            self.main_mask = np.zeros(self.final_pano.shape,dtype = 'uint8')                                                        # Initialize mask of main view location
            self.main_seam = np.zeros(self.final_pano.shape,dtype = 'uint8')                                                        # Initialize mask for main view overlap with side views

            # Correct intensity
            main_view_frame,side_view_frames = self.correctIntensity(main_view_frame,side_view_frames)
            
            # Generate main view mask
            self.main_mask[out_pos[0]:out_pos[0]+main_view_frame.shape[0],out_pos[1]:out_pos[1]+main_view_frame.shape[1]] = 1

            # Apply Background homographies       
            for i in range(len(side_view_frames)):
                result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frames[i],self.homography_list[i])
                

                #Place transformed background on canvas
                print "result2",result2.shape
                print "result1",result1.shape
                for j in range(len(self.transformed_mask_side_view)):
                    print "Alt view",j, ":",self.alt_view_masks[self.view_idx][j].shape
                    print "side_view",j,":",self.transformed_mask_side_view[j].shape

                print "window: ",self.final_pano [out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :].shape
                temp_result_window = self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :]
                self.final_pano [out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = result2 * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])
                #self.final_pano [out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = result2 * mask2_original + temp_result_window * np.logical_not(mask2_original)
                # Set main_seam = 1 for any nonzero pixels in the transformed side view.
                self.main_seam[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = np.maximum((result2 > 0).astype('uint8'),self.main_seam[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :])

            # generate mask of side view locations (expand by two pixels to ensure border is contained.)
            self.pano_mask = cv2.dilate(255*(np.sum(self.final_pano,2) > 0).astype('uint8'),np.ones((5,5),np.uint8),iterations=1)

            # Generate a border mask for the final canvas. (this ensures that the outer edge is detected as an edge by the edge detector)
            outer_edge = np.zeros(self.pano_mask.shape)
            outer_edge[0,:] = 1
            outer_edge[:,0] = 1
            outer_edge[outer_edge.shape[0]-1,:] = 1
            outer_edge[:,outer_edge.shape[1]-1] = 1
            outer_edge = outer_edge * self.pano_mask

            # Apply Edge detector to identify side view borders
            self.pano_mask = np.maximum(cv2.Canny(self.pano_mask,100,200)*(1-self.main_mask[:,:,0]),outer_edge)
            #self.pano_mask = outer_edge
            self.main_seam = self.main_seam * self.main_mask
            
            kernel = np.ones((5,5),np.uint8)
            self.main_seam[:,:,0] = cv2.dilate(self.main_seam[:,:,0],kernel,iterations=1)



            # Perform background fill with average background color
            self.final_pano[:,:,0] = self.final_pano[:,:,0] + self.max_weight[0] * (self.final_pano[:,:,0] == 0).astype('uint8')
            self.final_pano[:,:,1] = self.final_pano[:,:,1] + self.max_weight[1] * (self.final_pano[:,:,1] == 0).astype('uint8')
            self.final_pano[:,:,2] = self.final_pano[:,:,2] + self.max_weight[2] * (self.final_pano[:,:,2] == 0).astype('uint8')


        ###################################################################################################################################################################
        else:
            ####################################### If Object has been modeled ############################################################################################ 
            if (len(self.object_loc) > 2):
                 ### Perform Object detection and save timing info ###
                t = time.time()  # Begin timing for object detection

                # generate mask of main view border
                outer_edge = np.zeros(main_view_frame.shape)
                outer_edge[0,:] = 1
                outer_edge[:,0] = 1
                outer_edge[outer_edge.shape[0]-1,:] = 1
                outer_edge[:,outer_edge.shape[1]-1] = 1

                # detect motion in main view
                obj_detected,pts1,main_view_object_mask = genMainMask(main_view_frame,models[self.view_idx],self.main_seam[out_pos[0]:out_pos[0]+main_view_frame.shape[0],out_pos[1]:out_pos[1]+main_view_frame.shape[1]]*outer_edge)

                detect_time = time.time() - t
                print "Object Detection: ", detect_time
                # Save timing info
                file = open("obj_det_timing.txt", "a")
                file.write("Object Detection: ")
                file.write(str(detect_time))
                file.write("\n")
                file.close()

                if obj_detected:
                    ### Perform Alignment and save timing info ###
                    t = time.time()

                    # Compute aligning transformation for object.
                    # Track Lines
                    #self.line1,self.line2,self.line1_tracking_pts,self.line2_tracking_pts = trackObject(main_view_frame,self.background_models[0],self.line1,self.line1_tracking_pts,self.line2,self.line2_tracking_pts)
                    self.line1,self.line2,a = trackObject2(main_view_frame,self.background_models[self.view_idx],self.line1,self.line2, pts1,self.final_pano.shape,self.main_view_upleft_coord)
                    #line_frame = np.copy(main_view_frame)
                    #drawLines(self.line1[0],self.line1[1],line_frame)
                    #drawLines(self.line2[0],self.line2[1],line_frame)
                    #cv2.imshow("lines",line_frame)
                    #cv2.waitKey(0)

                    #a = np.mat([a[:,0],a[1,:]])
                    t2 = time.time()
                    (tempH,status) = cv2.findHomography(self.object_loc,a)
                    post_obj_mask = np.zeros([self.main_seam.shape[0],self.main_seam.shape[1],3])
                    pts = a.reshape((-1,1,2))
                    post_obj_mask = cv2.fillPoly(post_obj_mask, [pts.astype('int32')], (1,1,1))
                    #tempH,post_obj_mask = lineAlignWithModel(0,pts1.astype('int'),255*main_view_object_mask,self.object_loc,self.object_texture,self.main_seam,self.pano_mask,[out_pos[1],out_pos[0]],self.line1,self.line2)
                    compH_time = time.time() - t

                    file = open('comp_H_timing.txt','a')
                    file.write("Compute H: ")
                    file.write(str(compH_time))
                    file.write("\n")
                    file.close()

                    #print "Points found originally: ", pts1
                    #print "New Points: ", a
                    #cv2.imshow("mask1", post_obj_mask1)
                    #cv2.imshow("mask",post_obj_mask)
                    #cv2.waitKey(0)

                    align_time = time.time() - t 
                    print "Object_Alignment: ", align_time
                    # Save timing info
                    file = open("obj_align_timing.txt","a")
                    file.write("Object Alignment: ")
                    file.write(str(align_time))
                    file.write("\n")
                    file.close()

                    ### Perform warping and save timing info ###
                    t = time.time()

                    # Apply transformation to object and blend
                    t2 = time.time()
                    trans_obj = cv2.warpPerspective(self.object_texture,tempH,(out_pano.shape[1],out_pano.shape[0]))
                    H_warp_time = time.time() - t2
                    t2 = time.time()

                    #Apply Blending.
                    out_pano[post_obj_mask > 0] = trans_obj[post_obj_mask > 0]
                    blend_time = time.time() - t2


                    file = open("apply_H_timing.txt",'a')
                    file.write("Apply H timing: ")
                    file.write(str(H_warp_time))
                    file.write("\n")

                    file = open("blend_timing.txt",'a')
                    file.write("Blending Time: ")
                    file.write(str(blend_time))
                    file.write("\n")

                    warping_time = time.time() - t 
                    print "Object Warping: ", warping_time
                    #Save timing info
                    file = open("obj_warp_timing.txt","a")
                    file.write("Object Warping: ")
                    file.write(str(warping_time))
                    file.write("\n")
                    file.close()

            ################################# If object hasn't been modeled ###########################################################################################################
            else:

                transformed_side_obj = [[]] * len(side_view_frames)
                
                side_view_has_motion, seam_has_motion = self.read_next_frame(main_view_frame, side_view_frames)
                
                for i in range(len(side_view_frames)):
                    if side_view_has_motion[i]:
                        # Perform object alignment
                        if i < self.view_idx:
                            (_, transformed_side_bg,transformed_side_obj[i], _, side_mask, line_shift, _) = self.line_stitch(main_view_frame, side_view_frames[i], i,models[self.view_idx],models[i])

                        else:
                            (_, transformed_side_bg,transformed_side_obj[i], _, side_mask, line_shift, _) = self.line_stitch(main_view_frame, side_view_frames[i], i,models[self.view_idx],models[i+1])

                        #Place side view background in pano
                        temp_result_window = out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_bg.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_bg.shape[1], :]
                        out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_bg.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_bg.shape[1], :] = transformed_side_bg * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])

                    else:
                        # Warp frame
                        result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frames[i],self.homography_list[i])

                        # Place side view in Pano
                        temp_result_window = out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :]
                        out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = result2 * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])

                # Place object on top of background
                for i in range(len(side_view_frames)):
                    if side_view_has_motion[i]:
                        if len(transformed_side_obj[i]) > 0:
                            temp_result_window = out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_obj[i].shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_obj[i].shape[1], :]
                            obj_mask = (transformed_side_obj[i] > 0).astype('uint8')
                            out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_obj[i].shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_obj[i].shape[1], :] = transformed_side_obj[i] * obj_mask + temp_result_window * (1- obj_mask)


        # Place Main View 
        out_pano[out_pos[0]:out_pos[0]+self.main_view_image_shape[0], out_pos[1]:out_pos[1]+self.main_view_image_shape[1],:] = main_view_frame

        # Update background model
        #for i in range(self.num_side_cams+1):
        #    self.background_models[i] = (.99*self.background_models[i] + .01*frames[i]).astype('uint8')


        return out_pano.astype('uint8'), models

    def stitch2(self,frames,models):
    # uses incoming frames to update final panorama. 
        # INPUTS: 
        #     main_view_frame - A mat object containing the frame from the main view camera, must be in format 'uint8'
        #     side_view_frames - A list, for each side view contains the frame from that side view, must be format 'uint8'
        #     models - A list, for each side view, contains the frame gathered during background modeling. must be format 'uint8'
        # OUTPUTS:
        #     out_pano - The resulting image mosaic from our visualization algorithm. 

        out_pos = self.main_view_upleft_coord                               # Note main view shift
        sti = stitcher.Stitcher()                                           # To apply transformation
        out_pano = np.copy(self.final_pano)                                 # Canvas

        side_view_frames = list(frames)
        main_view_frame =  side_view_frames.pop(self.view_idx)

        if not self.pano_finished:
            self.pano_finished = True
            self.main_mask = np.zeros(self.final_pano.shape,dtype = 'uint8')                                                        # Initialize mask of main view location
            self.main_seam = np.zeros(self.final_pano.shape,dtype = 'uint8')                                                        # Initialize mask for main view overlap with side views

            # Generate main view mask
            self.main_mask[out_pos[0]:out_pos[0]+main_view_frame.shape[0],out_pos[1]:out_pos[1]+main_view_frame.shape[1]] = 1

            # Apply Background homographies       
            for i in range(len(side_view_frames)):
                result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frames[i],self.homography_list[i])      

                #Place transformed background on canvas
                print "window: ",self.final_pano [out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :].shape
                temp_result_window = self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :]
                self.final_pano [out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = result2 * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])
                #self.final_pano [out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = result2 * mask2_original + temp_result_window * np.logical_not(mask2_original)
                # Set main_seam = 1 for any nonzero pixels in the transformed side view.
                self.main_seam[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = np.maximum((result2 > 0).astype('uint8'),self.main_seam[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :])

            # generate mask of side view locations (expand by two pixels to ensure border is contained.)
            self.pano_mask = cv2.dilate(255*(np.sum(self.final_pano,2) > 0).astype('uint8'),np.ones((5,5),np.uint8),iterations=1)

            # Generate a border mask for the final canvas. (this ensures that the outer edge is detected as an edge by the edge detector)
            outer_edge = np.zeros(self.pano_mask.shape)
            outer_edge[0,:] = 1
            outer_edge[:,0] = 1
            outer_edge[outer_edge.shape[0]-1,:] = 1
            outer_edge[:,outer_edge.shape[1]-1] = 1
            outer_edge = outer_edge * self.pano_mask
            # Apply Edge detector to identify side view borders
            self.pano_mask = np.maximum(cv2.Canny(self.pano_mask,100,200)*(1-self.main_mask[:,:,0]),outer_edge)
            #self.pano_mask = outer_edge
            self.main_seam = self.main_seam * self.main_mask
          
            kernel = np.ones((5,5),np.uint8)
            self.main_seam[:,:,0] = cv2.dilate(self.main_seam[:,:,0],kernel,iterations=1)

        #################################################################################################################
        transformed_side_obj = [[]] * len(side_view_frames)
                
        side_view_has_motion, seam_has_motion = self.read_next_frame(main_view_frame, side_view_frames)
                
        for i in range(len(side_view_frames)):
                result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frames[i],self.homography_list[i])

                # Place side view in Pano
                temp_result_window = out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :]
                out_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = result2 * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])

        # Place Main View 
        out_pano[out_pos[0]:out_pos[0]+self.main_view_image_shape[0], out_pos[1]:out_pos[1]+self.main_view_image_shape[1],:] = main_view_frame


        return out_pano



    def line_stitch(self, main_view_frame, side_view_frame, idx,main_view_background,side_view_background):
        # 
        # INPUTS:
        #     main_view_frame - A mat object containing the frame from the main view camera, must be in format 'uint8'
        #     idx - the id of the side view camera to be stitched.
        #     side_view_frame - A mat containing the frame from the side view camera, must be in format 'uint8'
        #     main_view_background - a mat containing the model for the background of the main view camera
        #     side_view_background - a mat containing the model for the background of the side view camera of index idx
        # OUTPUTS:
        #     self.object_texture - a mat containing the frame which holds our detected object. 
        #     self.object_loc - a set of 4 tuples, each identifying a corner of the detected object in object_texture
        #     result1 - the padded main_view after applying the background homography
        #     bg_image - the transformed side view with the object removed and replaced with 
        #     obj_image - 
        #     mask1 - 
        #     new_mask - 
        #     shift -
        #     trans_matrix -
        sti = stitcher.Stitcher()
        result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frame,self.homography_list[idx])
        new_mask = (result2 > 0).astype('uint8')
        file = open("obj_det_timing.txt", "a")


        ### Perform Object detection and save timing info ###
        t = time.time()
        obj_detected,pts1,pts2,main_view_object_mask,side_view_object_mask = genObjMask(idx,main_view_frame, main_view_background,side_view_frame, side_view_background, self.crossing_edges_main_view_list, self.crossing_edges_side_views_list, self.homography_list)
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

                            self.line1,self.line2,self.line1_tracking_pts,self.line2_tracking_pts = trackObject(main_view_frame,self.background_models[self.view_idx],self.line1,self.line1_tracking_pts,self.line2,self.line2_tracking_pts)
                            main_seam, side_seam, side_border,transformed_side_border = genBorderMasks(main_view_frame, side_view_frame, mask1,new_mask,self.homography_list[idx],shift)
                            tempH,post_obj_mask = lineAlignWithModel(idx,pts1,255*main_view_object_mask,self.object_loc,main_seam,transformed_side_border,shift,self.homography_list[idx],self.line1,self.line2)
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
                            side_view_main_mask = mapCoveredSide(self.homography_list[idx],main_view_frame,side_view_frame)
                            main_seam, side_seam, side_border,transformed_side_border = genBorderMasks(main_view_frame, side_view_frame, mask1,new_mask,self.homography_list[idx],shift)
                            #tempH = la.lineAlign(pts1,main_view_frame,pts2,side_view_frame,self.fundamental_matrices_list[idx])
                            tempH,post_obj_mask,self.object_loc,self.line1_tracking_pts,self.line2_tracking_pts,self.line1,self.line2 = lineAlign(idx,pts1,255*main_view_object_mask,pts2,255*side_view_object_mask,self.fundamental_matrices_list[idx],main_seam, side_seam, side_border,transformed_side_border,shift,self.homography_list[idx])
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
                        bg_image,obj_image = warpObject(idx,side_view_object_mask,post_obj_mask,tempH,result2,self.object_texture,side_view_background,self.homography_list[idx],shift)
                        warping_time = time.time() - t 
                        print "Object Warping: ", warping_time
                        file.write("Object Warping: ")
                        file.write(str(warping_time))
                        file.write("\n")
                        file.close()
                            
        return result1,bg_image,obj_image,mask1,new_mask, shift, trans_matrix

#####################################################################3 Other Calibration Code ######################################################################################################


def computePanoShift(H_list,main_view_frame,main_view_image_shape,side_view_frames,side_view_image_shape,num_side_cams):
        seams_main_view_list = [];
        transformed_image_shapes_list = [];
        trans_matrices_list = [];
        shift_list = [];
        crossing_edges_main_view_list = []
        crossing_edges_side_views_list = []
        diff_buffer = []
        buffer_current_idx = []
        sti = stitcher.Stitcher()
        transformed_mask_side_view = []
        masks_side_view = []

        pano = np.zeros((5000, 5000, 3), np.uint8)
        out_pos = np.array([2500-main_view_image_shape[0]/2,2500-main_view_image_shape[1]/2]).astype('int')


        for i in range(num_side_cams):
            (transformed_main_view, transformed_side_view, mask_main_view, mask_side_view, shift, trans_matrix) = sti.applyHomography(np.ones(main_view_image_shape, np.uint8), (i + 2) * np.ones(side_view_image_shape[i], np.uint8), H_list[i])
            seam = sti.locateSeam(mask_main_view[:,:,0], mask_side_view[:,:,0])

            seams_main_view_list.append(seam[shift[1]:shift[1]+main_view_image_shape[0], shift[0]:shift[0]+main_view_image_shape[1]])
            
            transformed_image_shapes_list.append(transformed_main_view.shape)
            trans_matrices_list.append(trans_matrix)
            shift_list.append([shift[1],shift[0]])

            temp_result = (transformed_side_view * np.logical_not(mask_main_view) + transformed_main_view).astype('uint8')
            temp_result_window = pano[out_pos[0]-shift_list[i][0]:out_pos[0]-shift_list[i][0]+transformed_main_view.shape[0], out_pos[1]-shift_list[i][1]:out_pos[1]-shift_list[i][1]+transformed_main_view.shape[1], :]
            pano[out_pos[0]-shift_list[i][0]:out_pos[0]-shift_list[i][0]+transformed_main_view.shape[0], out_pos[1]-shift_list[i][1]:out_pos[1]-shift_list[i][1]+transformed_main_view.shape[1], :] = temp_result * mask_side_view + temp_result_window * np.logical_not(mask_side_view)


        pano[out_pos[0]:out_pos[0]+main_view_image_shape[0], out_pos[1]:out_pos[1]+main_view_image_shape[1],:] = np.ones(main_view_image_shape, np.uint8)

        pts = np.nonzero(pano)
        pts = np.asarray(pts)
        left_most = np.min(pts[1,:])-1
        right_most = np.max(pts[1,:])+1
        up_most = np.min(pts[0,:])-1
        down_most = np.max(pts[0,:])+1
        main_view_upleft_coord = [out_pos[0] - up_most, out_pos[1] - left_most]                                    #The location of the main view (0,0) in the final coord system
        final_pano = np.zeros((pano[up_most:down_most, left_most:right_most, :]).shape, np.uint8)                  #The empty canvas that will be used for the final panorama.
        
        kernel_opening_closing = np.ones((5,5),np.uint8)
        for i in range(len(side_view_frames)):
            transformed_mask = pano[out_pos[0]-shift_list[i][0]:out_pos[0]-shift_list[i][0]+transformed_image_shapes_list[i][0], out_pos[1]-shift_list[i][1]:out_pos[1]-shift_list[i][1]+transformed_image_shapes_list[i][1], :]
            transformed_mask = cv2.morphologyEx((transformed_mask == (i + 2)).astype('uint8'), cv2.MORPH_OPEN, kernel_opening_closing)
            transformed_mask_side_view.append(transformed_mask)
            masks_side_view.append(cv2.warpPerspective(transformed_mask_side_view[i], inv(trans_matrices_list[i]), (side_view_image_shape[i][1], side_view_image_shape[i][0])))



        # Should be migrated to geometric seams rather than seam masks.
        seam = np.zeros((main_view_image_shape[0], main_view_image_shape[1]))
        kernel = np.ones((50,50),np.uint8)
        for i in range(len(side_view_frames), 0, -1):
            seam = seam + (seam == 0) * (cv2.dilate(seams_main_view_list[i-1], kernel, iterations = 1) * i)

        kernel_gradient = np.mat([[0, 1, 0],[1, -4, 1],[0, 1, 0]])

        for i in range(len(side_view_frames)):
            temp_seam_main_view = sti.mapMainView(H_list[i],main_view_frame,side_view_frames[i])
            crossing_edges_main_view_list.append(temp_seam_main_view)

            temp_seam_side_view = sti.mapSeams(H_list[i],main_view_frame,side_view_frames[i])
            crossing_edges_side_views_list.append(temp_seam_side_view)

        ##############################################
        #for i in range(len(side_view_frames)):
        #    diff_buffer.append(np.zeros((side_view_image_shape[i][0], side_view_image_shape[i][1], 2), np.int))
        #    buffer_current_idx.append(False)
        #    diff_buffer[i][:,:,int(buffer_current_idx[i])] = side_view_cal_images[i][:,:,1];

        #diff_buffer.append(np.zeros((main_view_image_shape[0], main_view_image_shape[1], 2), np.int))
        #buffer_current_idx.append(False)
        #diff_buffer[len(side_view_frames)][:,:,int(buffer_current_idx[len(side_view_frames)])] = main_view_cal_image[:,:,1];



        return main_view_upleft_coord, shift_list,transformed_mask_side_view,crossing_edges_main_view_list,crossing_edges_side_views_list,seam,final_pano



###################################################################### Line Alignment Code #########################################################################################################



def calcF(image1,image2,label=0, ratio=.75):
    # Calculates the fundamental matrix between two cameras using matched feature points. This will give bad results if all of the detected 
    # feature points are coplanar. This script is also currently 
    if (label is 0):
        (kpsA, featuresA) = detectAndDescribe(image1)
        (kpsB, featuresB) = detectAndDescribe(image2)

        #matcher = cv2.DescriptorMatcher_create("BruteForce-L1")     # replaces "BruteForce" because L1 seems to give slightly better results. 
        matcher = cv2.BFMatcher(cv2.NORM_L1,crossCheck=False)

        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        #rawMatches = matcher.match(featuresA,featuresB)
                
        matches = []

        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
                print len(featuresA)
                print len(featuresB)
                    
                print len(matches), "Matches found"
        # computing a homography requires at least 4 matches, we use 10 to ensure a robust stitch.
        if len(matches) > 20:
                            
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
     
        (F, mask) = cv2.findFundamentalMat(ptsA,ptsB)

    else:
        #_,_,_,_,F = loadnpz('calibration_7_14_17/output.npz')
        #_,_,_,_,F = loadnpz('../tools/matlab_calibration.npz')
        _,_,_,_,F = loadnpz('../tools/camera_parameters.npz')
        F = F[label]

    #elif (label is 1):
        #F = np.loadtxt('fundamentalMatrices/F1.txt',delimiter=',')
    #elif (label is 2):
        #F = np.loadtxt('fundamentalMatrices/F2.txt',delimiter=',')
#   elif (label is 3):
        #F = np.loadtxt('fundamentalMatrices/F3.txt',delimiter=',')
    
    return F

def checkHalfPlane(rho,theta,x):
    w = np.mat([np.cos(theta),np.sin(theta)])
    b = -rho

    r = (np.dot(w,x) + b)/np.linalg.norm(w,2)

    return r


def checkLine(line, n):
# This function checks to see if the lines are vertical or horizontal.
# it is used to ensure that the image edges are not detected. but it may result 
# in 
# the dropping of desired lines. 

    rho = line[0,0]
    theta = line[0,1]
    #print "CHECKLINE: ", rho, n[0], n[1], theta
    #print (np.abs(rho - n[0]) < 5), (np.abs(theta - np.pi/2) < .001)
    if (np.abs(rho - n[1]) < 5) and (np.abs(theta) < .001):
        return False
    if (np.abs(rho - n[0]) < 5) and (np.abs(theta - np.pi/2) < .001):
        return False

    return True

def computeTrackingPoints(line1,line2,main_view_object_mask,shift,threshold = 5,line_gap = 5):
    # Compute and store the points used by our tracking algorithm.
    object_side_loc = [False,False,False,False]
    main_view_tracking_pts = []
    line1_chosen_pts = []
    line2_chosen_pts = []

    x = len(main_view_object_mask[0,:])
    y = len(main_view_object_mask[:,0])

    # Check if object intersects top side
    intersection_pts = np.nonzero(main_view_object_mask[0,:])
    if len(intersection_pts[0]) > 0:
        object_side_loc[0] = True

        min_pt = min(intersection_pts[0]) + shift[0,0]
        max_pt = max(intersection_pts[0]) + shift[0,0]

        line1_edge_pt = (int)((line1[0] - shift[0,1]*np.sin(line1[1]))/np.cos(line1[1])) #- shift[0,0]
        line2_edge_pt = (int)((line2[0] - shift[0,1]*np.sin(line2[1]))/np.cos(line2[1])) #- shift[0,0]

        if (max_pt - min_pt) < threshold:
            ave_pt = (max_pt + min_pt)/2
            if np.abs(ave_pt - line1_edge_pt) < np.abs(ave_pt - line2_edge_pt):
                main_view_tracking_pts.append([0+shift[0,1],line1_edge_pt])
                line1_chosen_pts.append([0+shift[0,1],line1_edge_pt])
            else:
                main_view_tracking_pts.append([0+shift[0,1],line2_edge_pt])
                line2_chosen_pts.append([[0+shift[0,1],line2_edge_pt]])
        else:
            if (line1_edge_pt >= shift[0,0]) and (line1_edge_pt <= x+shift[0,0]):
                main_view_tracking_pts.append([0+shift[0,1],line1_edge_pt])
                line1_chosen_pts.append([0+shift[0,1],line1_edge_pt])
            if (line2_edge_pt >= shift[0,0]) and (line2_edge_pt <= x+shift[0,0]):
                main_view_tracking_pts.append([0+shift[0,1],line2_edge_pt])
                line2_chosen_pts.append([0+shift[0,1],line2_edge_pt])



    # Check if object intersects left side
    intersection_pts = np.nonzero(main_view_object_mask[:,0])
    if len(intersection_pts[0]) > 0:
        object_side_loc[1] = True

        min_pt = min(intersection_pts[0]) + shift[0,1]
        max_pt = max(intersection_pts[0]) + shift[0,1]

        line1_edge_pt = (int)((line1[0] - shift[0,0]*np.cos(line1[1]))/np.sin(line1[1])) #- shift[0,1]
        line2_edge_pt = (int)((line2[0] - shift[0,0]*np.cos(line2[1]))/np.sin(line2[1])) #- shift[0,1]

        if (max_pt - min_pt) < threshold:
            ave_pt = (max_pt + min_pt/2)
            if np.abs(ave_pt - line1_edge_pt) < np.abs(ave_pt - line2_edge_pt):
                main_view_tracking_pts.append([line1_edge_pt,0+shift[0,0]])
                line1_chosen_pts.append([line1_edge_pt,0+shift[0,0]])
            else:
                main_view_tracking_pts.append([line2_edge_pt,0+shift[0,0]])
                line2_chosen_pts.append([line2_edge_pt,0+shift[0,0]])
        else:
            if (line1_edge_pt >= shift[0,1]) and (line2_edge_pt <= y+shift[0,1]):
                main_view_tracking_pts.append([line1_edge_pt,0+shift[0,0]])
                line1_chosen_pts.append([line1_edge_pt,0+shift[0,0]])
            if (line2_edge_pt >= shift[0,1]) and (line2_edge_pt <= y+shift[0,1]):
                main_view_tracking_pts.append([line2_edge_pt,0+shift[0,0]])
                line2_chosen_pts.append([line2_edge_pt,0+shift[0,0]])           

    # Check if object intersects bottom side
    intersection_pts = np.nonzero(main_view_object_mask[-1,:])
    if len(intersection_pts[0]) > 0:
        object_side_loc[2] = True
        
        min_pt = min(intersection_pts[0]) + shift[0,0]
        max_pt = max(intersection_pts[0]) + shift[0,0]

        line1_edge_pt = (int)((line1[0] - (y + shift[0,1])*np.sin(line1[1]))/np.cos(line1[1])) #- shift[0,0]
        line2_edge_pt = (int)((line2[0] - (y + shift[0,1])*np.sin(line2[1]))/np.cos(line2[1])) #- shift[0,0]        

        if (max_pt - min_pt) < threshold:
            ave_pt = (max_pt + min_pt)/2
            if np.abs(ave_pt - line1_edge_pt) < np.abs(ave_pt - line2_edge_pt):
                main_view_tracking_pts.append([y+shift[0,1],line1_edge_pt])
                line1_chosen_pts.append([y+shift[0,1],line1_edge_pt])
            else:
                main_view_tracking_pts.append([y+shift[0,1],line2_edge_pt])
                line2_chosen_pts.append([y+shift[0,1],line2_edge_pt])
        else:
            if (line1_edge_pt >= shift[0,0]) and (line1_edge_pt <= x+shift[0,0]):
                main_view_tracking_pts.append([y+shift[0,1],line1_edge_pt])
                line1_chosen_pts.append([y+shift[0,1],line1_edge_pt])
            if (line2_edge_pt >= shift[0,0]) and (line2_edge_pt <= x+shift[0,0]):
                main_view_tracking_pts.append([y+shift[0,1],line2_edge_pt])
                line2_chosen_pts.append([y+shift[0,1],line2_edge_pt])

    # Check if object intersects right side
    intersection_pts = np.nonzero(main_view_object_mask[:,-1])
    if len(intersection_pts[0]) > 0:
        object_side_loc[3] = True

        min_pt = min(intersection_pts[0]) + shift[0,1]
        max_pt = max(intersection_pts[0]) + shift[0,1]

        line1_edge_pt = (int)((line1[0] - (x+shift[0,0])*np.cos(line1[1]))/np.sin(line1[1])) #- shift[0,1]
        line2_edge_pt = (int)((line2[0] - (x+shift[0,0])*np.cos(line2[1]))/np.sin(line2[1])) #- shift[0,1]

        if (max_pt - min_pt) < threshold:
            ave_pt = (max_pt + min_pt/2)
            if np.abs(ave_pt - line1_edge_pt) < np.abs(ave_pt - line2_edge_pt):
                main_view_tracking_pts.append([line1_edge_pt,x+shift[0,0]])
                line1_chosen_pts.append([line1_edge_pt,x+shift[0,0]])
            else:
                main_view_tracking_pts.append([line2_edge_pt,x+shift[0,0]])
                line2_chosen_pts.append([line2_edge_pt,x+shift[0,0]])
        else:
            if (line1_edge_pt >= shift[0,1]) and (line1_edge_pt <= y+shift[0,1]):
                main_view_tracking_pts.append([line1_edge_pt,x+shift[0,0]])
                line1_chosen_pts.append([line1_edge_pt,x+shift[0,0]])
            if (line2_edge_pt >= shift[0,1])  and (line2_edge_pt <= y+shift[0,1]):
                main_view_tracking_pts.append([line2_edge_pt,x+shift[0,0]])
                line2_chosen_pts.append([line2_edge_pt,x+shift[0,0]])  

    return np.mat(line1_chosen_pts), np.mat(line2_chosen_pts)







def correctPoint(x, y, line):
#Adjust the point so that it lies on the desired line. This is used so that
#Neighborhoods only need to be approximate. 
    rho = line[0]
    theta = line[1]

    if (theta == 0):
        x = rho
    elif (np.abs(theta) < np.pi/4) or (np.abs(theta) > 3*np.pi/4): 
        x = -np.sin(theta)/np.cos(theta)*y + rho/np.cos(theta)
    else:
        y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)
    
    print "CorrectPoint: ", x,y,rho,theta
    return x,y

def detectTip(image):
    return


def detectAndDescribe(image):
#This function computes the SURF features of the image. 

    # detect and extract features from the image
    #descriptor = cv2.xfeatures2d.SIFT_create()
    descriptor = cv2.xfeatures2d.SURF_create()
        
    (kps, features) = descriptor.detectAndCompute(image, None)
                
 
    # convert the keypoints from KeyPoint objects to NumPy
    # arrays
    kps = np.float32([kp.pt for kp in kps])
    # return a tuple of keypoints and features
    return (kps, features)


def drawEpilines(img1, img2, linesA,linesB, pts1, pts2):
#A function for drawing the epipolar geometry onto a set of images. 

    r,c,z = img1.shape
    #r,c,z = img1.shape
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r1,r2,pt1,pt2 in zip(linesA,linesB,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())

        x0,y0 = map(int, [0, -r1[2]/r1[1] ])
        x1,y1 = map(int, [c, -(r1[2]+r1[0]*c)/r1[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)

        x0,y0 = map(int, [0, -r2[2]/r2[1] ])
        x1,y1 = map(int, [c, -(r2[2]+r2[0]*c)/r2[1] ])
        img2 = cv2.line(img2, (x0,y0), (x1,y1), color,1)
        #print r1,r2,pt1,pt2,color
        #img1 = cv2.circle(img1,(int(pt1[0]),int(pt1[1])),5,color,-1)
        #img2 = cv2.circle(img2,(int(pt2[0]),int(pt2[1])),5,color,-1)
    return img1,img2

def drawLines(rho,theta,image,color=(0,0,255),width=2):
#Copied code for drawing the lines on an image given rho and theta
#It doesn't appear to be functioning correctly, hopefully that's on the code 
# and not on the line transform, but if you run into issues with this, it might be 
# worth debugging. 
    if np.isnan(rho) or np.isnan(theta):
        return

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    #print "X0,b", x0,b,rho
    x1 = int(x0 + 10000*(-b))
    y1 = int(y0 + 10000*(a))
    x2 = int(x0 - 10000*(-b))
    y2 = int(y0 - 10000*(a))
    
    #print (x1,y1)
    #print (x2,y2)
    #print color
    #print width
    #print image.shape

    cv2.line(image,(x1,y1),(x2,y2),color,width)

def epiMatch(point, line, F, D = 1):
#This function uses the epipolar geometry of a scene to find the corresponding feature
#matches for a point which we know lies along a given line. 
#point - The point for which we want to find a feature match.
#line - the line on which the point will be found in the other image. 
#F - The fundamental matrix for the images
#D - the direction in which the epipolar lines should be computed. (should F get inverted.)
    homPoint = np.mat([point[0],point[1],1]) #change point to homogeneous coordinates
    #print homPoint,F
    #Calculate epipolar lines and change formatting of both sets of lines.
    epiline = cv2.computeCorrespondEpilines(homPoint, D,F)
    A = epiline[0,0,0]
    B = epiline[0,0,1]
    C = epiline[0,0,2]
    rho = line[0]
    theta = line[1]

    #Intersect two lines
    if (theta == 0):
        x = rho
        y2 = A/B*x + C/B
    else: 
        x = (rho/np.sin(theta) +C/B)/(np.cos(theta)/np.sin(theta) - A/B)
        y1 = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)
        y2 = -A/B*x - C/B

    #Determine error in the intersection (used for debugging. )
    #err1 = A*x + B*y2 + C
    #err2 = -x*np.cos(theta) - y1*np.sin(theta) + rho
    #epiline2 = cv2.computeCorrespondEpilines(homPoint, 2,F)
    #print "POINT: ", point
    #print "EPILINE: ", epiline, epiline2
    #print "OBJECT LINE: ", line
    #print "MATCHED POINT: ", x,y1
    #print "ERROR FROM LINES: ", err1, err2

    return (int(x),int(y2))

def findOuterEdge(image,seam,line):
    # TESTING: Code for finding the intersection of LINE with the outer edge of the IMAGE.
    # The code must detect which edge of the image is the outer edge, and then perform the intersection. 

    # Calculate image dimensions
    n = image.shape

    # Check where seam intersects with edges, our seam should be a mask of where the other views are located on top of the image. 
    # edge1, x = 0

    edge1 = np.zeros(n)
    edge1[:,0] = 1
    flag = [True, True, True, True]

    flag[0] = np.any(np.logical_and(edge1,seam))

    edge2 = np.zeros(n)
    edge2[0,:] = 1
    flag[1] = np.any(np.logical_and(edge2,seam))

    edge3 = np.zeros(n)
    edge3[n[0]-1,:] = 1
    flag[2] = np.any(np.logical_and(edge3,seam))

    edge4 = np.zeros(n)
    edge4[:,n[1]-1] = 1
    flag[3] = np.any(np.logical_and(edge4,seam))

    # Save the edge lines in Hough Line transform format. 
    edges = np.mat([[np.pi/2,0],[0,0],[0,n[0]],[np.pi/2,n[1]]])


    # intersect line with edge, 
    point = np.zeros([4,2])
    for i in range(0,4):
        if not flag[i]:
            #If a horizontal edge
            if edges[i,0] is 0:
                point[i,1] = np.int(edges[i,1])
                point[i,0] = np.int(-np.tan(line[0])*point[i,1] + line[1]/np.cos(line[0]))
            # If a vertical edge 
            else:
                point[i,0] = np.int(edges[i,1])
                point[i,1] = np.int(-point[i,0]/np.tan(line[0]) + line[1]/np.sin(line[0]))

            # Check if point is in image.
            if (point[i,0] <= n[0]) and (point[i,1] <= n[1]) and (point[i,0] >= 0) and (point[i,1] >= 0):
                return point[i,:]

    
    # return edge point
    return point[0,:]


def findOuterPts(point,vec,window):
    # Intersects the line defined by the equation [x,y] = a*vec + point with the outer edge of the image
    # check x = 0

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    vec = vec/np.linalg.norm(vec)

    if vec[0] is not 0:
        a1 = -point[0]/vec[0]
        if a1 >= 0:
            y1 = point[1] + a1 * vec[1]
            if y1 >= 0 and y1 <= window[0]:
                return [0, int(y1)]

    # check y = 0 
    if vec[1] is not 0:
        a2 = -point[1]/vec[1]
        if a2 >= 0:
            x1 = point[0] + a2 * vec[0]
            if x1 >= 0 and x1 <= window[1]:
                inter_found = True
                print [int(x1),0]
    # check x = x_max
    if vec[0] is not 0:
        a3 = (window[1]-point[0])/vec[0]
        if a3 >= 0:
            y2 = point[1] + a3 * vec[1]
            if y2 >= 0 and y2 <= window[0]:
                return [window[1],int(y2)]

    # check y = y_max
    if vec[1] is not 0:
        a4 = (window[0]-point[1])/vec[1]
        if a4 >= 0:
            x2 = point[0] + a4 * vec[0]
            if x2 >= 0 and x2 <= window[1]:
                return [int(x2),window[0]]

    print "Error: No Intersection Found in findOuterPts()"
    print a1,a2,a3,a4
    print y1,x1,y2,x2
    return [-1,-1]


def genBorderMasks(main_frame,side_frame, main_mask,side_mask,H,shift,edgeThresh = 1):
    # generate initial masks
    side_border = np.zeros(side_frame.shape)
    side_border[0,:,:] = 1
    side_border[:,0,:] = 1
    side_border[side_frame.shape[0]-1,:,:] = 1
    side_border[:,side_frame.shape[1]-1,:] = 1
    padded_side_border = np.pad(side_border,((shift[0]+1,0),(shift[1]+1,0),(0,0)),'constant',constant_values = 0)

    main_seam = main_mask * side_mask
    side_seam = mapCoveredSide(H,main_frame,side_frame)
    transformed_side_border = cv2.warpPerspective(padded_side_border,H,(side_mask.shape[1],side_mask.shape[0]))

    return main_seam, side_seam, side_border, transformed_side_border

def genMainMask(main_view_frame,main_view_background,crossing_edges_main_view_list):

    t = time.time()
    # Generate main view foreground mask
    main_view_object_mask = identifyFG(main_view_frame, main_view_background).astype('uint8')

    diff_frame_time = time.time() - t
    #Print timing info
    print diff_frame_time
    
    t = time.time()
    main_view_edge = cv2.Canny(main_view_object_mask, 50, 150)
    main_view_object_mask = (main_view_object_mask/255).astype('uint8')
    kernel_gradient = np.mat([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    main_view_edge = cv2.filter2D(main_view_object_mask,-1,kernel_gradient)
    #main_view_edge = (main_view_edge > 0).astype('uint8')
    main_view_edge = (main_view_edge > 0).astype('bool')
    pts = np.nonzero(np.logical_and(main_view_object_mask,crossing_edges_main_view_list[:,:,0].astype('bool')))

    #pts = np.nonzero(main_view_object_mask * crossing_edges_main_view_list[:,:,0])
    pts = np.asarray(pts)
    if pts.shape[1] >= 2:
        clusters_main_view, cluster_err = kmean(pts, 2)
        pts1 = np.mat([[clusters_main_view[1,0], clusters_main_view[0,0]], [clusters_main_view[1,1], clusters_main_view[0,1]]])
        file = open("check_seam_timing.txt",'a')
        file.write("Check Seam Overlap Timing: ")
        file.write(str(time.time() - t))
        file.write('\n')

        return True,pts1,main_view_object_mask

    file = open("check_seam_timing.txt",'a')
    file.write("Check Seam Overlap Timing: ")
    file.write(str(time.time() - t))
    file.write('\n')

    return False, [], []

def genObjMask(idx,main_view_frame, main_view_background, side_view_frame,side_view_background, crossing_edges_main_view_list, crossing_edges_side_views_list, homography_list ):
    # genObjMask detects a moving object which crosses the seam line between the main view and the side view. It returns a sub image containing the object,
    # and a point declaring the location of the 0.0 pixel of the mask in the original image.

    main_view_object_mask = identifyFG(main_view_frame, main_view_background).astype('uint8')
    main_view_edge = cv2.Canny(main_view_object_mask, 50, 150)
    main_view_object_mask = (main_view_object_mask/255).astype('uint8')
    kernel_gradient = np.mat([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    main_view_edge = cv2.filter2D(main_view_object_mask,-1,kernel_gradient)
    main_view_edge = (main_view_edge > 0).astype('uint8')

    side_view_object_mask = identifyFG(side_view_frame, side_view_background).astype('uint8')
    side_view_edge = cv2.Canny(side_view_object_mask,50,150)
    side_view_object_mask = (side_view_object_mask/255).astype('uint8')
    side_view_edge = cv2.filter2D(side_view_object_mask,-1,kernel_gradient)
    side_view_edge = (side_view_edge > 0).astype('uint8')

    # DISPLAY DETECTED OBJECTS
    #cv2.imshow("Side View Object"+str(idx),side_view_object_mask*255)
    #cv2.imshow("side View mask"+str(idx), np.logical_or(self.crossing_edges_side_views_list[idx], side_view_object_mask).astype('uint8')*255 ) 
    #cv2.imwrite("side_view_object"+str(idx)+".jpg",side_view_object_mask*255)
    #cv2.imwrite("side_view_mask"+str(idx)+".jpg",np.logical_or(self.crossing_edges_side_views_list[idx], side_view_object_mask).astype('uint8')*255 )
    #cv2.waitKey(10)

    pts = np.nonzero(main_view_edge * crossing_edges_main_view_list[idx])
    pts = np.asarray(pts)

    if pts.shape[1] >= 2:
            print "Object detected in main view"
            clusters_main_view, cluster_err = kmean(pts, 2)
            #if cluster_err <= 10:
            if True:    
                side_view_object_mask = identifyFG(side_view_frame, side_view_background).astype('uint8')
                side_view_edge = cv2.Canny(side_view_object_mask,50,150)
                side_view_object_mask = (side_view_object_mask/255).astype('uint8')
                side_view_edge = cv2.filter2D(side_view_object_mask,-1,kernel_gradient)
                side_view_edge = (side_view_edge > 0).astype('uint8')  
                
                #cv2.imshow("side View mask"+str(idx), side_view_object_mask*255 )              
                #cv2.imshow("Side View Frame " +str(idx), side_view_frame)
                #cv2.waitKey(10)
                #print np.nonzero(side_view_edge)

                pts = np.nonzero(side_view_edge * crossing_edges_side_views_list[idx])
                pts = np.asarray(pts)
                if pts.shape[1] >= 2:
                    print "Object Detected in side view"
                    clusters_side_view, cluster_err = kmean(pts, 2)
                    #if cluster_err <= 10:
                    if True:
                        dist_main = np.sum(np.square(clusters_main_view[:, 0] - clusters_main_view[:, 1]))
                        dist_side = np.sum(np.square(clusters_side_view[:, 0] - clusters_side_view[:, 1]))
                        print dist_main, dist_side
                        if True:
                        #if dist_main >= 100 and dist_main <= 5000 and dist_side >= 100 and dist_side <= 5000:

                            clusters_side_view = find_pairs(clusters_main_view, clusters_side_view, homography_list[idx])

                            ##############################
                            pts1 = np.mat([[clusters_main_view[1,0], clusters_main_view[0,0]], [clusters_main_view[1,1], clusters_main_view[0,1]]])
                            pts2 = np.mat([[clusters_side_view[1,0], clusters_side_view[0,0]], [clusters_side_view[1,1], clusters_side_view[0,1]]])
                            pts1 = np.mat([pts1[0,0], pts1[0,1]])
                            pts2 = np.mat([pts2[0,0], pts2[0,1]])

                            return True,pts1,pts2,main_view_object_mask,side_view_object_mask



    return False, [], [], [], []

def genTrackerPoints(line1,line2,obj_edge_mask,seam_mask,thresh = 1, line_resolution = 10):
    line1_points = [[],[]]
    line2_points = [[],[]]

    ## Find intersection between line1 and seam mask.

    # Check each edge individually to speed up computation time. 
    left_edge = np.nonzero(seam_mask[0,:]*obj_edge_mask[0,:])
    top_edge = np.nonzero(seam_mask[:,0]*obj_edge_mask[:,0])
    right_edge = np.nonzero(seam_mask[-1,:]*obj_edge_mask[-1,:])
    bottom_edge = np.nonzero(seam_mask[:,-1]*obj_edge_mask[:,-1])

    # identify line points
    if (len(left_edge[0]) > 0):
        temp_edge = left_edge
    elif (len(top_edge[0] > 0)):
        temp_edge = top_edge
    elif (len(right_edge[0] > 0)):
        temp_edge = right_edge
    elif (len(bottom_edge[0]) > 0):
        temp_edge = bottom_edge

    for i in range(len(temp_edge[0])):
        x = temp_edge[0][i]
        y = temp_edge[1][i]

        # Check to see if x and y fit line 1 equation 
        if np.abs(line1[0] - (x*np.cos(line1[1]) + y*np.sin(line1[1]))) < thresh:
            line1_border_point = [x,y]
        # Check to see if x and y fit line 2 equation
        if np.abs(line2[0] - (x*np.cos(line2[1]) + y*np.sin(line2[1]))) < thresh:
            line2_border_point = [x,y]

    # Use line 1 orientation to pick second line 1 point. 
    if np.abs(line1[1]) < (np.pi/4):
        if line1_border_point[0] == 0:
            x = line_resolution
        else:
            x = line1_border_point[0] - line_resolution
        y = (line1[0] - x*np.cos(line1[1]))/np.sin(line1[1])
    else:
        if line1_border_point[0] == 0:
            y = line_resolution
        else:
            y = line1_border_point - line_resolution
        x = (line1[0] - y*np.sin(line1[1]))/np.cos(line1[1])

    line1_second_point = [x,y]


    # Use line 2 orientation to pick second line 2 point. 
    if np.abs(line2[1]) < (np.pi/4):
        if line2_border_point[0] == 0:
            x = line_resolution
        else:
            x = line2_border_point[0] - line_resolution
        y = (line1[0] - x*np.cos(line2[1]))/np.sin(line2[1])
    else:
        if line1_border_point[0] == 0:
            y = line_resolution
        else:
            y = line2_border_point - line_resolution
        x = (line2[0] - y*np.sin(line2[1]))/np.cos(line2[1])

    line2_second_point = [x,y]


    return [line1_border_point,line1_second_point,line2_border_point,line2_second_point]



def identifyFG(frame, model,thresh = 40):
    #Uses the model from modelBackground to separate foreground objects from background objects
    # 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)

    diff_frame = np.abs(frame.astype('float') - model.astype('float'))
    binary = cv2.threshold(np.uint8(diff_frame),thresh,255,cv2.THRESH_BINARY)
    return binary[1]

def loadnpz(filename):
    #Loads a .npz file containing calibration information. 
    file_array = np.load(filename)
    (dst,K,R,t,F) = file_array['arr_0']

    return dst,K,R,t,F

def lineDetect(image, edgeThresh = 20,N_size = 200 ):
    # DEBUG: Sometimes detects lines incorrectly and has divide by zero errors for vertical lines.  
    # LineDetect detects whether there is a single prominent line in the image or 2. If there are 2, then 
    # the function will return the equation for each. Needs to be debugged, sometimes will only detect one 
    # line even though two are present. 

    # Generate Lines and points of intersection for image 1
    t = time.time()
    edges1 = cv2.Canny(image,edgeThresh,edgeThresh*3)
    edge_time = time.time() - t

    t = time.time()
    lines1 = lineSelect(edges1,edgeThresh,N_size)
    select_time = time.time() - t

    t = time.time()
    A_pts = np.nonzero(edges1[0,:])
    B_pts = np.nonzero(edges1[:,0])
    C_pts = np.nonzero(edges1[edges1.shape[0]-1,:])
    D_pts = np.nonzero(edges1[:,edges1.shape[1]-1])

    # Check whether multiple lines have been detected. 
    if len(A_pts)+len(B_pts)+len(C_pts)+len(D_pts) < 2:
        print "Not enough edge points: ",len(A_pts)+len(B_pts)+len(C_pts)+len(D_pts)
        return []
        
    if len(A_pts)+len(B_pts)+len(C_pts)+len(D_pts) == 2:
        outline = lines1[0]
        return outline

    # Check for distance between detected line and detected points
    A_dist = []
    B_dist = []
    C_dist = []
    D_dist = []
    pts = []
    distances = []
    temp_line = lines1[0]
    #print "temp_line: ",temp_line[0,0],temp_line[0,1]

    # A points distance
    if any(map(len,A_pts)):
        y = 0
        x = -np.sin(temp_line[0,1])/np.cos(temp_line[0,1])*y + temp_line[0,0]/np.cos(temp_line[0,1])
        A_dist = np.abs(A_pts - x)
        #print "A Distance: ",A_dist,x,y
        for i in range(0,len(A_pts[0])):
            #print A_pts[0][i],A_dist[0][i]
            pts.append([A_pts[0][i],y])
            distances.append(A_dist[0][i])

    # B points distance
    if any(map(len,B_pts)):
        x = 0
        y = -np.cos(temp_line[0,1])/np.sin(temp_line[0,1])*x + temp_line[0,0]/np.sin(temp_line[0,1])
        B_dist = np.abs(B_pts - y)
        #print "B Distance: ",B_dist,x,y
        for i in range(0,len(B_pts[0])):
            #print B_pts[0][i],B_dist[0][i]
            pts.append([x,B_pts[0][i]])
            distances.append(B_dist[0][i])
    
    # C points distance
    if any(map(len,C_pts)):
        y = edges1.shape[0]-1
        x = -np.sin(temp_line[0,1])/np.cos(temp_line[0,1])*y + temp_line[0,0]/np.cos(temp_line[0,1])
        C_dist = np.abs(C_pts - x)
        #print "C Distance: ",C_dist,x,y
        for i in range(0,len(C_pts[0])):
            #print C_pts[0][i],C_dist[0][i]
            pts.append([C_pts[0][i],y])
            distances.append(C_dist[0][i])
    
    # D points distance
    if any(map(len,D_pts)):
        x = edges1.shape[1]-1
        y = -np.cos(temp_line[0,1])/np.sin(temp_line[0,1])*x + temp_line[0,0]/np.sin(temp_line[0,1])
        D_dist = np.abs(D_pts - y)
        #print "D Distance: ",D_dist,x,y
        for i in range(0,len(D_pts[0])):
            #print D_pts[0][i],D_dist[0][i]
            pts.append([x,D_pts[0][i]])
            distances.append(D_dist[0][i])

    #print pts
    #print distances
    idx = np.argsort(np.mat(distances))
    #print idx
    #print pts[idx[0,0]], pts[idx[0,1]]


    chosen_pts = np.array(pts)[idx]
    chosen_pts = [[chosen_pts[0][0][0],chosen_pts[0][0][1]],[chosen_pts[0][1][0],chosen_pts[0][1][1]]]
    distances = np.array(distances)[idx]
    first_line = temp_line
    second_line = first_line

    

    #print chosen_pts
    #print distances

    # detect distance to subsequent lines
    for i in range(1, len(lines1)):
        print "Detecting subsequent lines..."
        pts = []
        distances = []
        temp_line = lines1[i]

        # A points distance
        if any(map(len,A_pts)):
            y = 0
            x = -np.sin(temp_line[0,1])/np.cos(temp_line[0,1])*y + temp_line[0,0]/np.cos(temp_line[0,1])
            A_dist = np.abs(A_pts - x)
            #print "A Distance: ",A_dist,x,y
            for i in range(0,len(A_pts[0])):
                #print A_pts[0][i],A_dist[0][i]
                pts.append([A_pts[0][i],y])
                distances.append(A_dist[0][i])

        # B points distance
        if any(map(len,B_pts)):
            x = 0
            y = -np.cos(temp_line[0,1])/np.sin(temp_line[0,1])*x + temp_line[0,0]/np.sin(temp_line[0,1])
            B_dist = np.abs(B_pts - y)
            #print "B Distance: ",B_dist,x,y
            for i in range(0,len(B_pts[0])):
                #print B_pts[0][i],B_dist[0][i]
                pts.append([x,B_pts[0][i]])
                distances.append(B_dist[0][i])
    
        # C points distance
        if any(map(len,C_pts)):
            y = edges1.shape[0]-1
            x = -np.sin(temp_line[0,1])/np.cos(temp_line[0,1])*y + temp_line[0,0]/np.cos(temp_line[0,1])
            C_dist = np.abs(C_pts - x)
            #print "C Distance: ",C_dist,x,y
            for i in range(0,len(C_pts[0])):
                #print C_pts[0][i],C_dist[0][i]
                pts.append([C_pts[0][i],y])
                distances.append(C_dist[0][i])
    
        # D points distance
        if any(map(len,D_pts)):
            x = edges1.shape[1]-1
            y = -np.cos(temp_line[0,1])/np.sin(temp_line[0,1])*x + temp_line[0,0]/np.sin(temp_line[0,1])
            D_dist = np.abs(D_pts - y)
            #print "D Distance: ",D_dist,x,y
            for i in range(0,len(D_pts[0])):
                #print D_pts[0][i],D_dist[0][i]
                pts.append([x,D_pts[0][i]])
                distances.append(D_dist[0][i])

        idx = np.argsort(np.mat(distances))
        line_pts = np.array(pts)[idx]
        line_pts = [[line_pts[0][0][0],line_pts[0][0][1]],[line_pts[0][1][0],line_pts[0][1][1]]]


        line_found = False

        for i in range(0,2):
            match_flag = False
            for j in range(0,2):
                if np.linalg.norm(np.mat(line_pts[i]) - np.mat(chosen_pts[j])) < 5:
                    print "Match Found: ",line_pts,chosen_pts
                    match_flag = True

            if not match_flag:
                second_line = temp_line
                line_found = True
                break



        if line_found:
            break
        

    match_time = time.time() - t
    file = open('subtest.txt','a')
    file.write("Edge Timing: " + str(edge_time) + "\n")
    file.write("Line Timing: " + str(select_time) + "\n")
    file.write("Matching Timing" + str(match_time) + "\n")
    file.close()    
    return [first_line[0,0],first_line[0,1]],[second_line[0,0],second_line[0,1]]

def lineEdge(idx,line, frame_mask, border_mask):
    # INDEVELOPMENT: lineEdge finds intersections between line and frame_edge. 
    # line=(rho,theta) stores the line information 
    # frame_mask shows a mask of the frame, with a mask denoting where the other image will be located

    frame_mask = frame_mask.astype('bool')
    border_mask = border_mask.astype('bool')

    if (len(frame_mask.shape) == 2):
        frame_mask = np.repeat(frame_mask[:,:,np.newaxis],3,axis=2)
    if (len(border_mask.shape) == 2):
        border_mask = np.repeat(border_mask[:,:,np.newaxis],3,axis=2)

    line_mask = np.zeros(frame_mask.shape)
    drawLines(line[0],line[1],line_mask,(1,1,1),1)

    line_mask = line_mask.astype('bool')

    #kernel = np.ones((5,5),np.uint8)
    #frame_mask[:,:,0] = cv2.dilate(frame_mask[:,:,0],kernel,iterations=1)

    possible_points = np.nonzero(np.logical_and(line_mask[:,:,0].astype('bool'),np.logical_and(border_mask[:,:,0].astype('bool'),np.logical_not(frame_mask[:,:,0]))))
    #possible_points = np.nonzero(line_mask[:,:,0]*border_mask[:,:,0]*(1-frame_mask[:,:,0]))

    if(len(possible_points[0]) > 1):
        if (max(possible_points[0]) - min(possible_points[0]) > 6) or (max(possible_points[0]) - min(possible_points[0]) > 6): 
            print "Warning: line intersects with too many edges in lineEdge. Total number of edges ",len(possible_points[0])

    elif (len(possible_points[0]) == 0):
        print "Warning: no point found."
        return (-10,-10)

    print"Possible points: ", possible_points
    point = (possible_points[1][0],possible_points[0][0])


    return point


    


def lineSelect(image, edgeThresh, N_size):
# Computes the line corresponding to the most prominent edge in the image.

    n = image.shape
    outlines = (0,0)

    edges = cv2.Canny(image,edgeThresh,edgeThresh*3)

    #cv2.imshow("Temp",image)
    #cv2.waitKey(0)

    lines = cv2.HoughLines(edges[1:edges.shape[0]-1,1:edges.shape[1]-1],1,np.pi/180,N_size/2)

    temp_image = np.repeat(edges[:, :, np.newaxis]/2, 3, axis=2)
    temp_image = 100*temp_image.astype('uint8')
    if lines is None:
        print "ERROR: No line found in lineSelect"
        return np.mat([0,0])

    for line in lines:
        #print line[0,0], line[0,1]
        drawLines(line[0,0],line[0,1],temp_image)

    cc = 0;
    outlines = []
    for line in lines:
        #print "LINE: ",line, np.abs(line[0,1] - np.pi/2) > .001
        #Removing vertical and horizontal lines, this is probably not a great way of selecting lines. 
        if checkLine(line,n):
            #print "Not 0: ", cc,line[0,0], line[0,1]
            outlines.append([line[0,0],line[0,1]])
            cc = cc + 1

    outlines = sortLines(outlines, edges)

    # Grab line crossing points

    # Identify 2 dominant line crossings
    # Check for secondary line crossings
    #  
    #print "FINAL LINE: ", outlines

    return outlines

def matchLines(point1, matpoint1, point2, matpoint2):
# 

    #Define variables use by the first point
    x1 = point1[0]
    y1 = point1[1]
    x2 = matpoint2[0]
    y2 = matpoint2[1]

    #Define variables used by the second point
    mx1 = matpoint1[0]
    my1 = matpoint1[1]
    mx2 = point2[0] 
    my2 = point2[1]

    #Parameterize first line
    m1 = (y2 - y1)/(x2 - x1)
    b1 = y1 - m1*x1

    #Parameterize second line
    m2 = (my2 - my1)/(mx2 - mx1)
    b2 = my1 - m2*mx1

    #Calculate rotation required
    theta  = np.arctan2(y2 - y1, x2 - x1) - np.arctan2(my2 - my1, mx2 - mx1) 
    R = [[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]

    #Calculate scaling required
    d1 = np.sqrt(pow(y2 - y1,2)) + np.sqrt(pow(x2 - x1,2))
    d2 = np.sqrt(pow(my2 - my1,2)) + np.sqrt(pow(mx2 - mx1,2))
    s = d1/d2
    S = [[s,0,0],[0,s,0],[0,0,1]]

    #Calculate translation to center rotation around origin requirement.
    t1 = [-mx1,-my1]
    tx2 = x2 - t1[0]
    ty2 = y2 - t1[1]

    T1 = [[1,0,t1[0]],[0,1,t1[1]],[0,0,1]]

    #Calculate secondary translation
    A =  np.dot(np.dot(S,R),T1)
    t2 = np.transpose([x1,y1,1] - np.dot(A,np.transpose([mx1,my1,1])))
    T2 = [[1,0,t2[0]],[0,1,t2[1]],[0,0,1]]

    #return the output transformation. 
    output = np.dot(T2,A)
    return output

def mapCoveredSide(H, main_view, side_view):
        main_view_mask = (main_view > 0).astype('uint8')
        backwards_H = np.linalg.inv(H)

        transformed_mask = cv2.warpPerspective(main_view_mask,backwards_H,(side_view.shape[1],side_view.shape[0]))

        return transformed_mask

def modelBackground(caps, CAL_LENGTH = 10):
    # This function runs the image captures until the user inputs p. Then it uses the next
    # CAL_LENGTH frames to generate a model for the background. 

    frames = []
    output = []
    n = len(caps)


    for k in range(0,n):
        frames.append([])
        output.append([])   
        ret, frames[k] = caps[k].read()

    print "Press P to start estimating background, press q to quit."
    while ret:
        for k in range(0,n):
            ret,frames[k] = caps[k].read()
            cv2.imshow("frame " + str(k), frames[k])

        if cv2.waitKey(10) == ord('q'):
            return False, []

        if cv2.waitKey(10) == ord('p'):
            for i in range(0,n):
                cv2.destroyWindow("frame "+str(i))
            #cv2.destroyWindow("frame 0")
            #cv2.destroyWindow("frame 1")
            #cv2.destroyWindow("frame 2")
            #cv2.destroyWindow("frame 3")
            cv2.waitKey(1)

            break

    for k in range(0,n):
        output[k] = frames[k]/CAL_LENGTH
        for m in range(1,CAL_LENGTH):
            ret, frames[k] = caps[k].read()
            output[k] = output[k] + frames[k]/CAL_LENGTH


        output[k] = output[k].astype('uint8')
    return True, output


def pairLines(lines1,lines2,H,main_view_seam):
    #
    line1 = lines1[0]
    line2 = lines1[1]
    mat1 = lines2[0]
    mat2 = lines2[1]

    
    # Transform line to line segment
    y1 = lines2[0][0]/np.sin(lines2[0][1]) # x = 0
    y2 = -np.cos(lines2[0][1])/np.sin(lines2[0][1]) + y1 # x = 1
    pts1 = np.mat([[0,1],[y1,y2],[1,1]])

    y1 = lines2[1][0]/np.sin(lines2[1][1]) # x = 0
    y2 = -np.cos(lines2[1][1])/np.sin(lines2[1][1]) + y1 # x = 1
    pts2 = np.mat([[0,1],[y1,y2],[1,1]])

    # Apply homography to line segment
    transformed_pts1 = np.dot(H,pts1)
    transformed_pts2 = np.dot(H,pts2)

    # correct for disparity
    transformed_pts1 = np.mat([[transformed_pts1[0,0]/transformed_pts1[2,0],transformed_pts1[1,0]/transformed_pts1[2,1]],[transformed_pts1[0,1]/transformed_pts1[2,0],transformed_pts1[1,1]/transformed_pts1[2,1]]])
    transformed_pts2 = np.mat([[transformed_pts2[0,0]/transformed_pts2[2,0],transformed_pts2[1,0]/transformed_pts2[2,1]],[transformed_pts2[0,1]/transformed_pts2[2,0],transformed_pts2[1,1]/transformed_pts2[2,1]]])

    # Convert segment back to line.
    if transformed_pts1[0,0] == transformed_pts1[1,0]:
        transformed_lines1 = [transformed_pts1[0,0], 0]
    else:
        transformed_lines1 = [0, np.pi/2 - np.arctan((transformed_pts1[0,1]-transformed_pts1[1,1])/(transformed_pts1[0,0]-transformed_pts1[1,0]))]
        transformed_lines1[0] = transformed_pts1[0,1]*np.sin(transformed_lines1[1]) + transformed_pts1[0,0]*np.cos(transformed_lines1[1])

    if transformed_pts2[0,0] == transformed_pts2[1,0]:
        transformed_lines2 = [transformed_pts2[0,0], 0]
    else:
        transformed_lines2 = [0,np.pi/2 - np.arctan((transformed_pts2[0,1]-transformed_pts2[1,1])/(transformed_pts2[0,0]-transformed_pts2[1,0]))]
        transformed_lines2[0] = transformed_pts2[0,1]*np.sin(transformed_lines2[1]) + transformed_pts2[0,0]*np.cos(transformed_lines2[1])

    # Detect seam crossing points
    mask1 = np.zeros(main_view_seam.shape)
    mask2 = np.zeros(main_view_seam.shape)
    mask3 = np.zeros(main_view_seam.shape)
    mask4 = np.zeros(main_view_seam.shape)
    drawLines(transformed_lines1[0],transformed_lines1[1],mask1,width = 1)
    drawLines(transformed_lines2[0],transformed_lines2[1],mask2, width = 1)
    drawLines(lines1[0][0],lines1[0][1],mask3)
    drawLines(lines1[1][0],lines1[1][1],mask4)
    main_view_line1_pt = np.nonzero(mask3*main_view_seam)
    main_view_line2_pt = np.nonzero(mask4*main_view_seam)
    side_view_line1_pt = np.nonzero(mask1*main_view_seam)
    side_view_line2_pt = np.nonzero(mask2*main_view_seam)

    if len(main_view_line1_pt[0]) == 0 or len(main_view_line2_pt[0]) == 0:
        print "ERROR: Line does not intersect main view"
        return lines1[0],lines2[0],lines1[1],lines2[1]
    if len(side_view_line1_pt[0]) == 0 or len(side_view_line2_pt[0]) == 0:
        print "ERROR: Line does not intersect side view"
        return lines1[0],lines2[0],lines1[1],lines2[1]

    main_view_line1_pt = [main_view_line1_pt[1][0],main_view_line1_pt[0][0]]
    main_view_line2_pt = [main_view_line2_pt[1][0],main_view_line2_pt[0][0]]
    side_view_line1_pt = [side_view_line1_pt[1][0],side_view_line1_pt[0][0]]
    side_view_line2_pt = [side_view_line2_pt[1][0],side_view_line2_pt[0][0]]

    main_view_pt = np.transpose((np.mat(main_view_line1_pt) + np.mat(main_view_line2_pt))/2)
    side_view_pt = np.transpose((np.mat(side_view_line2_pt) + np.mat(side_view_line1_pt))/2)

    #print lines1[1][0],lines1[1][1], main_view_pt
    #print transformed_lines1,transformed_lines2, side_view_pt

    # Calculate distances
    r1 = np.sign(checkHalfPlane(transformed_lines1[0],transformed_lines1[1],side_view_pt))
    r2 = np.sign(checkHalfPlane(transformed_lines2[0],transformed_lines2[1],side_view_pt))
    r3 = np.sign(checkHalfPlane(lines1[0][0],lines1[0][1],main_view_pt))
    r4 = np.sign(checkHalfPlane(lines1[1][0],lines1[1][1],main_view_pt))

    # Match lines
    if r1 == r3:
        line1 =  lines1[0]
        line2 = lines1[1]
        mat1 = lines2[0]
        mat2 = lines2[1]

    elif r1 == r4:
        line1 = lines1[0]
        line2 = lines1[1]
        mat1 = lines2[1]
        mat2 = lines2[0]

    else:
        print "ERROR: line match not found"

    # Return matched lines
    return line1,mat1,line2,mat2

def setRight(line):
    # Shifts the point to the right to avoid overlapping features.
    x = 400
    y = 400
    rho = line[0]
    theta = line[1]

    if (theta == 0):
        x = rho
    else: 
        y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)

    return x,y
def setLeft(line):
    # Shifts the point to the right to avoid overlapping features.

    x = 5
    y = 5
    rho = line[0]
    theta = line[1]

    if (theta == 0):
        x = rho
    else: 
        y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)

    return x,y

def shiftRight(x,y,line,dist = 5):
    # Shifts the point to the right to avoid overlapping features.
    x = x+dist
    y = y + dist
    rho = line[0]
    theta = line[1]

    if (theta == 0):
        x = rho
    else: 
        y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)

    return x,y
def shiftLeft(x,y,line,dist = 5):
    # Shifts the point to the right to avoid overlapping features.

    x = x - dist
    y = y - dist
    rho = line[0]
    theta = line[1]

    if (theta == 0):
        x = rho
    else: 
        y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)

    return x,y


def sortLines(lines, edge):
# Sort the lines by how well they cover the edge. Still needs to bwe adjusted for multi-edge
    #print "Pre-sort:", lines
    H = np.eye(3)
    lines = np.mat(lines)

    # Transform the lines based on H

    # Evaluate for the seam 

    coverage = []
    for line in lines:
        #print line
        temp= np.zeros((edge.shape[0], edge.shape[1],3)).astype('uint8')
        drawLines(line[0,0],line[0,1],temp)
        temp = np.nonzero(temp[:,:,0]*edge)
        coverage.append(len(temp[0]))
        #print temp,coverage

    #idx = coverage.index(max(coverage))
    idx = np.argsort(-np.mat(coverage))
    np.array(lines)[idx]
    outlines = lines
    #outlines = lines[idx]  
    #outlines = lines[lines[:,1].argsort()]
    #print "Post-sort: ", outlines



    return outlines

def surgSeg(image, model,N_size = 20, edgeThresh = 20): 
    # IN DEVELOPMENT: Code to detect the location of  a surgical tool and generate a mask to cover it. 

    # generate a blank mask of the correct shape
    mask = np.uint8(np.zeros(image.shape))

    # find the difference image between the current frame and the background model. 
    diff_img = np.uint8(abs(np.int16(image) - np.int16(model)))

    # Apply canny edge detector to the difference image
    edges = cv2.Canny(diff_img,edgeThresh,edgeThresh*3) 
    #cv2.imshow('edges', edges)
    #cv2.waitKey(0)
    #cv2.destroyWindow("edges")

    # Apply hough line transform to detect dominant edge lines
    #lines = cv2.HoughLines(edges,1,np.pi/180,N_size*2)
    lines = cv2.HoughLinesP(edges,rho = 1,theta = np.pi/180, threshold = 2, minLineLength = 5)



    # If DRAW_LINES, then we should draw our detected lines on the image (used for testing)
    lines2 = np.zeros((len(lines),1,2))
    if True and lines.any() != None:
        tmp_image = image
        for i in range(0, len(lines)):
            
            #drawLines(lines[i,0,0],lines[i,0,1], tmp_image,width = 1)

            lines2[i,0,1] = np.arctan2(lines[i,0,3] - lines[i,0,1], lines[i,0,2] - lines[i,0,0])
            lines2[i,0,0] = lines[i,0,1]*np.cos(lines2[i,0,1]) + lines[i,0,0]*np.sin(lines2[i,0,1])
            drawLines(lines2[i,0,0],lines2[i,0,1], tmp_image,width = 1)


            cv2.line(tmp_image, (lines[i,0,0],lines[i,0,1]),(lines[i,0,2],lines[i,0,3]),[0,255,0],2)


        
        #print lines
        #print lines2
        #cv2.imshow('surg_mask', tmp_image)
        #cv2.waitKey(0)
        lines = lines2
    # Bound the surgical tool with a set of bounding half planes
    lineBound(mask,lines)

    # Calculate mask using intersection of half planes.

    return mask


def trackObject(frame,model,line1,previous_line1_points,line2,previous_line2_points,padding = 5,threshold=40):
    # utilize the information about where the object was in the previous frame to 
    # compute the lines in the current frame. 

    print "Previous Line 1: ", line1
    print "Previous Points 1: ",previous_line1_points
    print "Previous Line 2: ", line2
    print "Previous Points 2: ",previous_line2_points


    # Convert frames to Greyscale (For optimization, move down to transform scanline)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    model = cv2.cvtColor(model,cv2.COLOR_BGR2GRAY)
    diff_img = np.abs(frame.astype('int') - model.astype('int')).astype('uint8')
    edges = cv2.Canny(diff_img, 100,200)

    # search vertically for points (May want to adjust to search only a neighborhood)
    # detect scan lines associated with previous points.
    x1 = previous_line1_points[0,1]
    x2 = previous_line1_points[1,1]

    scanline1 = edges[:,x1]
    scanline1_pts = np.nonzero(scanline1)
    scanline2 = edges[:,x2]
    scanline2_pts = np.nonzero(scanline2)
    #scanline1 = np.abs(frame[:,x1].astype('int') - model[:,x1].astype('int'))
    #scanline2 = np.abs(frame[:,x2].astype('int') - model[:,x2].astype('int'))

    #search scanlines for edges.
    #scanline1_pts = (scanline1[1:] - scanline1[0:-1]) > threshold               # Checks edges, but moves index by 1
    #scanline1_pts = np.nonzero(scanline1_pts.astype('uint8'))

    #scanline2_pts = (scanline2[1:] - scanline2[0:-1]) > threshold               # Checks edges but moves index by 1
    #scanline2_pts = np.nonzero(scanline2_pts.astype('uint8'))


    # find edges nearest old points
    vertical_line1_min_dist1 = len(scanline1)
    vertical_line2_min_dist1 = len(scanline1)
    vertical_line1_min_dist2 = len(scanline2)
    vertical_line2_min_dist2 = len(scanline2)

    vertical_line1_pt = [-1,-1]
    vertical_line2_pt = [-1,-1]

    for i in range(len(scanline1_pts[0])):
        line1_dist = np.abs(scanline1_pts[0][i] + 1 - previous_line1_points[0,0])     # Note we need to move scanline_pts over by 1
        line2_dist = np.abs(scanline1_pts[0][i] + 1 - previous_line2_points[0,0])     # Note we need to move scanline_pts over by 1
        if line1_dist < vertical_line1_min_dist1:
            vertical_line1_min_dist1 = line1_dist
            vertical_line1_pt[0] = scanline1_pts[0][i] + 1

        if line2_dist < vertical_line2_min_dist1:
            vertical_line2_min_dist1 = line2_dist
            vertical_line2_pt[0] = scanline1_pts[0][i] + 1

    for i in range(len(scanline2_pts[0])):
        line1_dist = np.abs(scanline2_pts[0][i] + 1 - previous_line1_points[0,0])     # Note we need to move scanline_pts over by 1
        line2_dist = np.abs(scanline2_pts[0][i] + 1 - previous_line2_points[0,0])     # Note we need to move scanline_pts over by 1
        if line1_dist < vertical_line1_min_dist2:
            vertical_line1_min_dist2 = line1_dist
            vertical_line1_pt[1] = scanline2_pts[0][i] + 1

        if line2_dist < vertical_line2_min_dist2:
            vertical_line2_min_dist2 = line2_dist
            vertical_line2_pt[1] = scanline2_pts[0][i] + 1



    # Search Horizontally for points
    # detect scan lines associated with previous points.
    y1 = previous_line1_points[0,0]
    y2 = previous_line1_points[1,0]


    scanline1 = edges[y1,:]
    scanline1_pts = np.nonzero(scanline1)
    scanline2 = edges[y2,:]
    scanline2_pts = np.nonzero(scanline2)


    #scanline1 = np.abs(frame[y1,:].astype('int') - model[y1,:].astype('int'))
    #scanline2 = np.abs(frame[y2,:].astype('int') - model[y2,:].astype('int'))

    #search scanlines for edges.
    #scanline1_pts = (scanline1[1:] - scanline1[0:-1]) > threshold               # Checks edges, but moves index by 1
    #scanline1_pts = np.nonzero(scanline1_pts.astype('uint8'))

    #scanline2_pts = (scanline2[1:] - scanline2[0:-1]) > threshold               # Checks edges but moves index by 1
    #scanline2_pts = np.nonzero(scanline2_pts.astype('uint8'))


    # find edges nearest old points
    horizontal_line1_min_dist1 = len(scanline1)
    horizontal_line2_min_dist1 = len(scanline1)
    horizontal_line1_min_dist2 = len(scanline2)
    horizontal_line2_min_dist2 = len(scanline2)

    horizontal_line1_pt = [-1,-1]
    horizontal_line2_pt = [-1,-1]

    for i in range(len(scanline1_pts[0])):
        line1_dist = np.abs(scanline1_pts[0][i] + 1 - previous_line1_points[0,1])     # Note we need to move scanline_pts over by 1
        line2_dist = np.abs(scanline1_pts[0][i] + 1 - previous_line2_points[0,1])     # Note we need to move scanline_pts over by 1
        if line1_dist < horizontal_line1_min_dist1:
            horizontal_line1_min_dist1 = line1_dist
            horizontal_line1_pt[0] = scanline1_pts[0][i] + 1

        if line2_dist < horizontal_line2_min_dist1:
            horizontal_line2_min_dist1 = line2_dist
            horizontal_line2_pt[0] = scanline1_pts[0][i] + 1

    for i in range(len(scanline2_pts[0])):
        line1_dist = np.abs(scanline2_pts[0][i] + 1 - previous_line1_points[0,1])     # Note we need to move scanline_pts over by 1
        line2_dist = np.abs(scanline2_pts[0][i] + 1 - previous_line2_points[0,1])     # Note we need to move scanline_pts over by 1
        if line1_dist < horizontal_line1_min_dist2:
            horizontal_line1_min_dist2 = line1_dist
            horizontal_line1_pt[1] = scanline2_pts[0][i] + 1

        if line2_dist < horizontal_line2_min_dist2:
            horizontal_line2_min_dist2 = line2_dist
            horizontal_line2_pt[1] = scanline2_pts[0][i] + 1


    # Choose vertical or Horizontal points
    points1 = np.zeros([2,2])
    points2 = np.zeros([2,2])

    if (horizontal_line1_min_dist1 + horizontal_line1_min_dist2) < (vertical_line1_min_dist1 + vertical_line1_min_dist2):
        points1 = np.mat([[y1,horizontal_line1_pt[0]],[y2,horizontal_line1_pt[1]]])
    else:
        points1 = np.mat([[vertical_line1_pt[0],x1],[vertical_line1_pt[1],x2]])

    if (horizontal_line2_min_dist1 + horizontal_line2_min_dist2) < (vertical_line2_min_dist1 + vertical_line2_min_dist2):
        points2 = np.mat([[y1,horizontal_line2_pt[0]],[y2,horizontal_line2_pt[1]]])
    else:
        points2 = np.mat([[vertical_line2_pt[0],x1],[vertical_line2_pt[1],x2]])    

    # generate line information
    line1 = [0,0]
    line2 = [0,0]

    # Compute Theta
    line1[1] = np.pi/2  + np.arctan((float)(points1[1,0] - points1[0,0])/(float)(points1[1,1] - points1[0,1]))
    line2[1] = np.pi/2  + np.arctan((float)(points2[1,0] - points2[0,0])/(float)(points2[1,1] - points2[0,1]))

    # compute rho offset from theta (rho = xcos(theta) + ysin(theta)) for all x,y
    line1[0] = points1[0,1] * np.cos(line1[1]) + points1[0,0] * np.sin(line1[1])
    line2[0] = points2[0,1] * np.cos(line2[1]) + points2[0,0] * np.sin(line2[1])


    frame = cv2.cvtColor(frame,cv2.COLOR_GRAY2BGR)
    cv2.circle(frame, (points1[0,1],points1[0,0]), 7, (255,0,0),3)
    cv2.circle(frame, (points1[1,1],points1[1,0]), 7, (255,0,0),3)
    cv2.circle(frame, (points2[0,1],points2[0,0]), 7, (0,0,255),3)
    cv2.circle(frame, (points2[1,1],points2[1,0]), 7, (0,0,255),3)



    print "Points1: ",horizontal_line1_pt,vertical_line1_pt
    print "Points2: ",horizontal_line2_pt,vertical_line2_pt
    print "Horizontal Lines Distance", horizontal_line1_min_dist1,horizontal_line1_min_dist2,horizontal_line2_min_dist1,horizontal_line2_min_dist2
    print "Vertical Lines Distance: ", vertical_line1_min_dist1, vertical_line1_min_dist2, vertical_line2_min_dist1, vertical_line1_min_dist2


    print "Tracked Points 1: ",points1
    print "Tracked Line 1: ",line1
    print "Tracked Points 2: ",points2
    print "Tracked Line 2: ",line2

    #drawLines(line1[0],line1[1],frame)
    #drawLines(line2[0],line2[1],frame)
    #cv2.imshow("Object Tracking",frame)
    #cv2.waitKey(0)



    return line1,line2,points1,points2

def trackObject2(frame,model,line1,line2,pts,window,shift = [0,0]):
    t = time.time()
    grid_height = 5
    grid_width = 5
    top_line = [0,0]
    bot_line = [0,0]
    fit_top_complete = False
    fit_bot_complete = False
    show_frame =  np.copy(frame)
    frame = np.abs(frame.astype('float') - model.astype(float))

    # Compute Tracking neighborhood. 
    min_x = 0
    min_y = 0
    max_x = (len(frame[0,:])-1)/grid_height
    max_y = (len(frame[:,0])-1)/grid_width

    #print max_x
    #print max_y

    # Generate scanlines
    x_scanlines = []
    for i in range(min_x,max_x):
        #print i
        x_scanlines.append(frame[:,i*grid_height])


    y_scanlines = []
    for i in range(min_y, max_y):
        y_scanlines.append(frame[i*grid_width,:])

    # Search Scanlines for new points
    left_pts = []
    right_pts = []
    top_pts = []
    bot_pts = []
    for i in range(len(x_scanlines)):
        min_pts,max_pts = findScanlineEdge(x_scanlines[i])
        if (min_pts > 0) and (min_pts < max_x*grid_height+1):
            left_pts.append([i*grid_height,min_pts])
        if (max_pts > 0) and (max_pts < max_x*grid_height+1):
            right_pts.append([i*grid_height,max_pts])

    for i in range(len(y_scanlines)):
        min_pts,max_pts = findScanlineEdge(y_scanlines[i])
        if (min_pts > 0) and (min_pts < max_y*grid_height+1):
            top_pts.append([min_pts,i*grid_height])
        if (max_pts > 0) and (max_pts < max_y*grid_height+1):
            bot_pts.append([max_pts,i*grid_height])

    # Fit new lines to points
    if len(left_pts) > 2:
        # Fit left and right lines
        fit_left_x = [[]] * len(left_pts)
        fit_left_y = [[]] * len(left_pts)
        for i in range(len(left_pts)):
            fit_left_x[i] = left_pts[i][0]
            fit_left_y[i] = left_pts[i][1]

        fit_right_x = [[]] * len(right_pts)
        fit_right_y = [[]] * len(right_pts)
        for i in range(len(right_pts)):
            fit_right_x[i] = right_pts[i][0]
            fit_right_y[i] = right_pts[i][1]

        #print fit_left_x, fit_left_y
        #print fit_right_x,fit_right_y

        fit_left = np.polyfit(fit_left_x,fit_left_y,1)
        fit_right = np.polyfit(fit_right_x,fit_right_y,1)

        #print fit_left
        #print fit_right

    # add top,bot points to corresponding dataset
    fit_top_x = []
    fit_top_y = []
    fit_bot_x = []
    fit_bot_y = []
    if len(left_pts) > 2:
        if fit_left[1] > fit_right[1]:
            fit_top_x = fit_left_x
            fit_top_y = fit_left_y
            fit_bot_x = fit_right_x
            fit_bot_y = fit_right_y

        else:
            fit_top_x = fit_right_x
            fit_top_y = fit_right_y
            fit_bot_x = fit_left_x
            fit_bot_y = fit_left_y

    if len(top_pts) > 2:
        for i in range(len(top_pts)):
            fit_top_x.append(top_pts[i][0])
            fit_top_y.append(top_pts[i][1])
        for i in range(len(bot_pts)):
            fit_bot_x.append(bot_pts[i][0])
            fit_bot_y.append(bot_pts[i][1])

    if len(top_pts) > 2 or len(left_pts) > 2:
        # fit full lines
        #print fit_top_x
        #print fit_top_y
        #print fit_bot_x
        #print fit_bot_y
        if len(np.unique(fit_top_x)) < 2:
            fit_top_complete = True
            top_line = [0,fit_top_x[0]]
        else:
            fit_top = np.polyfit(fit_top_x,fit_top_y,1)

        if len(np.unique(fit_bot_x)) < 2:
            fit_bot_complete = True
            bot_line = [0,fit_bot_x[0]]   
        else:
            fit_bot = np.polyfit(fit_bot_x,fit_bot_y,1)
    else:
        fit_top = [0,-1]
        fit_bot = [0,-1]

    if not fit_top_complete:
        top_line[1] = np.pi/2  + np.arctan(fit_top[0])                          # Compute Theta
        top_line[0] = fit_top[1] * np.sin(top_line[1])                          # compute rho offset from theta (rho = xcos(theta) + ysin(theta)) for all x,y
    if not fit_bot_complete:
        bot_line[1] = np.pi/2  + np.arctan(fit_bot[0])                          # Compute Theta
        bot_line[0] = fit_bot[1] * np.sin(bot_line[1])                          # compute rho offset from theta (rho = xcos(theta) + ysin(theta)) for all x,y

    # match new lines to old lines
    if line1[1] == 0:
        b1 = line1[0]
    else:
        b1 = line1[0]/np.sin(line1[1])
    if line2[1] == 0:
        b2 = line2[0]
    else:
        b2 = line2[0]/np.sin(line2[1])

    if b1 > b2:
        line2 = top_line
        line1 = bot_line
    else:
        line1 = top_line
        line2 = bot_line


    line_time = time.time() - t
    file = open('match_lines_timing.txt','a')
    file.write("Line Matching: ")
    file.write(str(line_time))
    file.write("\n")
    file.close()
    
    #generate point, vector format
    # Determine location inside the image. 
    
    t = time.time()
    # Find point on edge of image.
    pts[0,0],pts[0,1] = correctPoint(pts[0,0],pts[0,1],top_line)
    pts[1,0],pts[1,1] = correctPoint(pts[1,0],pts[1,1],bot_line)
    if pts[0,1] < pts[1,1]:
        top_in_pts = np.mat([pts[0,0],pts[0,1]]).astype('int')
        bot_in_pts = np.mat([pts[1,0],pts[1,1]]).astype('int')
    else:
        top_in_pts = np.mat([pts[1,0],pts[1,1]]).astype('int')
        bot_in_pts = np.mat([pts[0,0],pts[1,1]]).astype('int')

    ave_pt = correctPoint(np.mean(fit_top_x),np.mean(fit_top_y),top_line)
    edge_pts = top_in_pts.astype('int')
    vec = (edge_pts - ave_pt).astype('int')

    top_trans = [ave_pt[0] + shift[1],ave_pt[1]+ shift[0]]
    top_vec = [vec[0,0],vec[0,1]]

    ave_pt = correctPoint(np.mean(fit_bot_x),np.mean(fit_bot_y),bot_line) 
    edge_pts = bot_in_pts.astype('int')
    vec = (edge_pts - ave_pt).astype('int')

    bot_trans = [ave_pt[0]+shift[1],ave_pt[1]+shift[0]]
    bot_vec = [vec[0,0],vec[0,1]]



    top_out_pts = findOuterPts(top_trans,top_vec,window)
    bot_out_pts = findOuterPts(bot_trans,bot_vec,window)

    out_pts = np.zeros([4,2])
    out_pts[0,:] = top_trans
    out_pts[1,:] = bot_trans
    out_pts[2,:] = bot_out_pts
    out_pts[3,:] = top_out_pts

    det_time = time.time() - t
    file = open('det_feat_timing.txt','a')
    file.write("Feature Detection: ")
    file.write(str(det_time))
    file.write("\n")
    file.close()
    
    #out_poly = np.mat([[top_in_pts, top_out_pts],[bot_in_pts,bot_out_pts]])

    #print "Tracked Line 1: ",line1
    #print "Tracked Line 2: ",line2

    #print"Out Points: ", out_pts
    #print "Top Vec: ", top_vec
    #print "Bot Vec: ", bot_vec


    #img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
    #cv2.line(show_frame, (top_in_pts[0,0],top_in_pts[0,1]),(top_in_pts[0,0] - top_vec[0], top_in_pts[0,1] - top_vec[1]),(255,0,0),3)
    #cv2.line(show_frame, (bot_in_pts[0,0],bot_in_pts[0,1]),(bot_in_pts[0,0] - bot_vec[0], bot_in_pts[0,1] - bot_vec[1]),(255,0,0),3)
    #cv2.imshow("Tracked Lines",show_frame)
    #cv2.waitKey(0)

    # Generate neighborhood to search for next lines

    return line1, line2, out_pts

def findScanlineEdge(scanline):
    # Should find the edges of a square pulse shape in the scanline
    threshold = 75
    detected_pts =np.nonzero(scanline > threshold)
    if len(detected_pts[0]) == 0:
        return -1,-1
    min_pt = np.min(detected_pts[0])
    max_pt = np.max(detected_pts[0])

    return min_pt,max_pt


def warpObject(idx,side_view_object_mask,trans_obj_mask,tempH,result2,side_frame,background_model,H,coord_shift):
    # Applies the transformation computed by the alignment step to the detected foreground
    # object and blends it with the orignal frame using the background model to fill in the gap.
    stitch = stitcher.Stitcher()

    if (tempH is None):
        print "Error: No foreground homography"
        return result2,[]

    #side_view_object_mask = np.repeat(side_view_object_mask[:,:,np.newaxis],3,axis=2) #.astype('uint8')

    if (side_view_object_mask.shape[0] > 0) and (side_view_object_mask.shape[1] > 0 ):
        print tempH
        if (len(side_view_object_mask.shape) == 2):
            side_view_object_mask = np.repeat(side_view_object_mask[:,:,np.newaxis],3,axis=2)

        #trans_obj_mask = cv2.warpPerspective(side_view_object_mask,tempH, (result2.shape[1],result2.shape[0]))
        #trans_obj_mask = side_view_object_mask
        #trans_obj_mask = np.repeat(trans_obj_mask[:,:,np.newaxis],3,axis=2)
        trans_obj = cv2.warpPerspective(side_frame,tempH, (result2.shape[1],result2.shape[0]))

        # This does not apply correctly for frame 3
        #background_model = np.pad(background_model,((coord_shift[0],0),(coord_shift[1],0),(0,0)),'constant',constant_values = 0)

        background_fill = side_frame*(1 - side_view_object_mask) + background_model* side_view_object_mask
        trans = np.eye(3)
        trans[0,2] = coord_shift[0]
        trans[1,2] = coord_shift[1]
        background_fill = cv2.warpPerspective(background_fill,np.dot(trans,H),(result2.shape[1],result2.shape[0]))    
        #_,background_fill,_,_,_,_ = stitch.applyHomography(side_frame,side_frame,H)

        if idx == 2:
            cv2.imshow("pre-transform", 255*side_view_object_mask.astype('uint8'))

        # This line does not behave appropriately when used on side_view frame 3.
        side_view_object_mask = np.pad(side_view_object_mask,((coord_shift[0],0),(coord_shift[1],0),(0,0)),'constant',constant_values = 0)
        side_view_object_mask = cv2.warpPerspective(side_view_object_mask,H,(result2.shape[1],result2.shape[0]))    
        #_,side_view_object_mask,_,_,_,_ = stitch.applyHomography(side_view_object_mask,side_view_object_mask,H)
        
        #n = result2.shape
        #trans_obj_mask = np.pad(trans_obj_mask,((coord_shift[1],0),(coord_shift[0],0),(0,0)),'constant',constant_values = 0)
        #trans_obj_mask = trans_obj_mask[0:n[0],0:n[1],0:n[2]]
        #trans_obj = np.pad(trans_obj,((coord_shift[1],0),(coord_shift[1],0),(0,0)),'constant',constant_values = 0)
        #trans_obj = trans_obj[0:n[0],0:n[1],0:n[2]]
        
        result2 = background_fill
        #if idx == 2:
        #   cv2.imshow("Background Filled"+str(idx),result2)
        result2 = trans_obj_mask*trans_obj + (1 - trans_obj_mask)*result2
        trans_obj = trans_obj_mask*trans_obj


        if idx == 7:
            cv2.imshow("obj mask "+str(idx),255*side_view_object_mask.astype('uint8'))
            cv2.imshow('back fill '+str(idx),background_fill)
            cv2.imshow("result2 "+str(idx),result2)
            cv2.waitKey(0)

    
        #result1,result2,mask1,new_mask,shift,trans_mat = stitch.applyHomography(main_frame,side_frame,np.linalg.inv(tempH))
        #pano2 = result2*(1 - mask1.astype('uint8'))+result1*mask1.astype('uint8')

    print background_fill.shape
    print trans_obj.shape
    return background_fill, trans_obj


def lineAlignWithModel(idx,points1,image1,side_view_pts,image2,main_seam,transformed_side_border,shift,lines11,lines12,N_size = 40, edgeThresh = 70,DRAW_LINES = False):
    # Avoid error cases
    if points1[0,1] <= N_size:
        points1[0,1] = N_size
    if points1[0,0] <= N_size:
        points1[0,0] = N_size

    out_obj_mask = []

    ## Line Detection
    t = time.time()
    #Isolate Point Neighborhoods
    sub1 = image1[points1[0,1] - N_size:points1[0,1]+N_size, points1[0,0] - N_size:points1[0,0]+N_size]

    #Identify the coordinate shift for sub neighborhoods. 
    #shift1 = shift + points1[0,:] - [N_size,N_size]
    shift1 = np.mat(shift)
    points1 = points1 + shift

    # Detect dominant lines

    #print sub1.shape
    #lines11, lines12 = lineDetect(sub1,edgeThresh,N_size)
    line_time = time.time() - t


    ## Coordinate Shifting
    t = time.time()
    # If no lines are found, print error and return identity
    if (lines11 is None):
        print "Error: Lines not found", lines1
        return np.zeros((3,3)),out_obj_mask

    sub1 = np.repeat(sub1[:,:,np.newaxis],3,axis=2)
    lines11 = [np.cos(-np.arctan2(shift1[0,1],shift1[0,0]) + lines11[1])*np.sqrt(pow(shift1[0,0],2) + pow(shift1[0,1],2)) + lines11[0],lines11[1]]
    if lines12:
        lines12 = [np.cos(-np.arctan2(shift1[0,1],shift1[0,0]) + lines12[1])*np.sqrt(pow(shift1[0,0],2) + pow(shift1[0,1],2)) + lines12[0],lines12[1]]
    line2_time = time.time() - t


    ## Line Pairing
    t = time.time()
    #lines11,lines22,lines12,lines21 = pairLines([lines11,lines12],[lines21,lines22],H,main_seam)
    line3_time = time.time() - t

    print "Lines 1: ",lines11,lines12


    ## Ensure Point lies on line
    t = time.time()

    feat_time = time.time()
    points = np.zeros((2,2))
    points[0,0], points[0,1] = correctPoint(points1[0,0],points1[0,1],lines11)
    if lines12:
        print "Adjusting for lines12"
        points[1,0],points[1,1] = correctPoint(points1[0,0],points1[0,1],lines12)
    points = np.round(points).astype('int')
    correction_time = time.time() - t


    ## Choose Main View Points
    t = time.time()
    main_view_pts =  np.zeros((4,2))
    main_view_pts[0,:] = points[0,:] #+ np.mat([shift[0],shift[1]])
    main_view_pts[1,:] = points[1,:] #+ np.mat([shift[0],shift[1]])
    main_view_pts[3,:] = lineEdge(idx,lines11,main_seam,transformed_side_border) #+ np.mat([shift[0],shift[1]])
    main_view_pts[2,:] = lineEdge(idx,lines12,main_seam,transformed_side_border) #+ np.mat([shift[0],shift[1]])
    main_view_pts = main_view_pts.astype('int')
    point_time = time.time() - t

    print "Shift: ",shift
    print "Points: ",points
    print "Main View Points: ",main_view_pts
    print "Side View Points: ",side_view_pts
    print "Main Seam Shape: ",main_seam.shape
    print "DONE!"


    ## 
    pts = main_view_pts.reshape((-1,1,2))
    out_obj_mask = np.zeros((main_seam.shape[0],main_seam.shape[1],3))

    #print pts.shape, pts.dtype
    #print out_obj_mask.shape
    
    cv2.fillPoly(out_obj_mask,[pts],(1,1,1))


    # Perform Feature Matching
    points1 = (main_view_pts[0,0],main_view_pts[0,1])
    points2 = (side_view_pts[0,0],side_view_pts[0,1])
    mat1 = (main_view_pts[1,0],main_view_pts[1,1])
    mat2 = (side_view_pts[1,0],side_view_pts[1,1])


    ## Visualize Results
    if DRAW_LINES:
                ptsB = np.mat([points1,mat2])
                ptsA = np.mat([mat1,points2])
                #tmp_image1 = np.repeat(image1[:, :, np.newaxis], 3, axis=2)
                #tmp_image2 = np.repeat(image2[:, :, np.newaxis], 3, axis=2)
                tmp_image2 = np.ones(image2.shape).astype('uint8')
                #tmp_image1 = np.copy(100*main_seam).astype('uint8')
                tmp_image1 = 50*np.ones((main_seam.shape[0], main_seam.shape[1],3)).astype('uint8')
                tmp_image1[0:image1.shape[0],0:image1.shape[1],:] = np.repeat(image1[:,:,np.newaxis],3,axis=2).astype('uint8')
                #print tmp_image1.shape

                tmp_image1 = np.pad(tmp_image1,((shift[1],0),(shift[0],0),(0,0)),'constant',constant_values = 0)



                drawLines(lines11[0],lines11[1],tmp_image1)
                #drawLines(lines21[0],lines21[1],tmp_image2)
                if lines12:
                    drawLines(lines12[0],lines12[1],tmp_image1)
                #if lines22:
                #   drawLines(lines22[0],lines22[1],tmp_image2)


                
                #cv2.circle(tmp_image1, mat2, 5, (255,0,0))
                #cv2.circle(tmp_image2, mat1, 5, (0,255,0))
                #cv2.circle(tmp_image1, points1, 5, (0,255,0))
                #cv2.circle(tmp_image2, points2, 5, (255,0,0))

                cv2.circle(tmp_image1, (main_view_pts[0,0],main_view_pts[0,1]), 7, (255,0,0),3)
                cv2.circle(tmp_image1, (main_view_pts[1,0],main_view_pts[1,1]), 7, (0,255,0),3)
                cv2.circle(tmp_image1, (main_view_pts[2,0],main_view_pts[2,1]), 7, (0,0,255),3)
                cv2.circle(tmp_image1, (main_view_pts[3,0],main_view_pts[3,1]), 7, (255,0,255),3)

                cv2.circle(tmp_image2, (side_view_pts[0,0],side_view_pts[0,1]), 7, (255,0,0),3)
                cv2.circle(tmp_image2, (side_view_pts[1,0],side_view_pts[1,1]), 7, (0,255,0),3)
                cv2.circle(tmp_image2, (side_view_pts[2,0],side_view_pts[2,1]), 7, (0,0,255),3)
                cv2.circle(tmp_image2, (side_view_pts[3,0],side_view_pts[3,1]), 7, (255,0,255),3)

                #print points1,points2
                print "PtsA: ",ptsA
                print "PtsB: ",ptsB
                #linesA = cv2.computeCorrespondEpilines(ptsA, 2,F)
                #linesB = cv2.computeCorrespondEpilines(ptsB, 1,F)
                #linesA = linesA.reshape(-1,3)
                #linesB = linesB.reshape(-1,3)
                #img5,img6 = drawEpilines(tmp_image1,tmp_image2,linesA,linesB,ptsB,ptsA)
                print "tmp Image Shape: ",tmp_image2.shape
                cv2.imshow("image1 (2 edge) ",tmp_image1)
                cv2.imshow("image2 (2 edge) ",tmp_image2)
                #cv2.waitKey(0)
                #cv2.destroyWindow("image1 (2 edge)")
                #cv2.destroyWindow("image2 (2 edge)")




    if (main_view_pts[2,:] == (-10,-10)).all() or (main_view_pts[3,:] == (-10,-10)).all() or (side_view_pts[2,:] == (-10,-10)).all() or (side_view_pts[3,:] == (-10,-10)).all():
        print "ERROR: Unable to detect line edge intersection",
        print "Main view points: ", main_view_pts
        print "Side view points: ",side_view_pts
        shift_mat = np.eye(3)
        shift_mat[0,2] = shift[0]
        shift_mat[1,2] = shift[1]
        return np.eye(3), out_obj_mask


    feat_time = time.time() - feat_time
    file = open("det_feat_timing.txt",'a')
    file.write("Feature Generation: ")
    file.write(str(feat_time))
    file.write("\n")
    file.close()
    
    t = time.time()
    (LineT,status) = cv2.findHomography(side_view_pts,main_view_pts)
    trans_time = time.time() - t
    file = open("comp_H_timing.txt",'a')
    file.write("Homography Computation: ")
    file.write(str(trans_time))
    file.write("\n")
    file.close()

    # Print timing info
    #file = open("test.txt" 'a')
    #file.write("Line Detection 1: " + str(line_time) + "\n")
    #file.write("Line Detection 2: " + str(line2_time) + "\n")
    #file.write("Line Detection 3: " + str(line3_time) + "\n")
    #file.write("Point Correction: " + str(correction_time) + "\n")
    #file.write("Point Choosing: " + str(point_time) + "\n")
    #file.write("Transformation Calculation: " + str(trans_time) + "\n")

    file.close()
    #print "Proposed PtsB: ",np.dot(LineT,np.transpose(np.mat([[main_view_pts[0,0],main_view_pts[0,1],1],[main_view_pts[1,0],main_view_pts[1,1],1],[main_view_pts[2,0],main_view_pts[2,1],1],[main_view_pts[3,0],main_view_pts[3,1],1]])))
    #print "Actual Pts B: ",side_view_pts
    print "LineT: ",LineT
    return LineT,out_obj_mask


def lineAlign(idx,points1, image1,points2, image2, F, main_seam, side_seam, side_border,transformed_side_border,shift,H, N_size = 40, edgeThresh = 70,DRAW_LINES = False):
    # lineAlign performs epipolar line alignment on the detected object. 
    # the Function takes the following inputs:
    # points1: a 1x2 or 2x2 matrix of the form [x,y] which denotes the approximate location of the intersection of
    # the edges of the object with the edges of image1
    # image1:  the main view image, this image should not be transformed.
    # points2: a 1x2 or 2x2 matrix of the form [x,y] which denotes the approximate location of the intersection of
    # the edges of the object with the edges of image2
    # image2:  the side view image which will be transformed.
    # F: the fundamental matrix relating image 1 to image 2.
    # main_seam: the mask for the edge of image 2 as seen in image 1
    # side_seam: the mas, for the edge of image 1 as seen in image 2
    # side_border: A mask for the outer edge of image 2 prior to transformation
    # transformed_side_border:  A mask for the outer edge of image 2 after transformtation
    # shift: The shift applied to the image 1 when the Homography is applied to image 2.
    # N_size: the size bounding box we should use to detect the dominant line associated with each object edge.
    # Too small of a bounding box will lead to a poor estimate of the line, but too large a bounding box may lead
    # to incorrect edges being detected. 
    # edgeThresh: the threshold value for the edge detector. 
    # DRAW_LINES: a flag used for Visualizing the line alignment process. 

    # This function first detects the two lines l_1, l_2 which make up the prominent edges of the object in image 1
    # and the two lines \hat{l}_1,\hat{l}_2 which make up the prominent edges of the object in image 2. 

    # The function then detects the points p_1,p_2,p_3, and p_4. where
    # p_1 and p_2 are the intersection of l_1, l_2 respectively with the edge of image1
    # and p_3, p_4 are the intersection of l_1,l_2 respectively with the far edge of image2

    # Then the desired destination of the object transformation is chosen using our epipolar line alignment.
    # We compute a set of features \hat{p}_1, \hat{p}_2,\hat{p}_3, \hat{p}_4 that will contain the positions 
    # of the side view points which must be aligned to p_1, \ldots, p_4. 

    # \hat{p}_1 = (x,y) s.t. l_1*[x,y,1]' = 0 and  [x,y,1]*F*p_1 = 0                    
    # \hat{p}_2 = (x,y) s.t. l_2*[x,y,1]' = 0 and  [x,y,1]*F*p_2 = 0
    # \hat{p}_3 = p_3 
    # \hat{p}_4 = p_4
        file = open("test.txt",'a')

        if idx == 4:
            DRAW_LINES = True
        # The function then computes a homography transformation H_{obj} that align the \hat{p}_i to the p_i.

        if (N_size == []):
            N_size = 20
        if (edgeThresh == []):
            edgeThresh = 20


        #Error checking code. If the point is too close to the edge, this bugs out and causes the program to crash. We shift the window to prevent this from happening. 
        # This may be a source of innefficiency which may need to be changed later. 
        if points1[0,1] <= N_size:
            points1[0,1] = N_size
        if points1[0,0] <= N_size:
            points1[0,0] = N_size
        if points2[0,1] <= N_size:
            points2[0,1] = N_size
        if points2[0,0] <= N_size:
            points2[0,0] = N_size
        #End Error checking segment. 

        out_obj_mask = []

        t = time.time()
        #Isolate point Neighborhoods
        sub1 = image1[points1[0,1] - N_size:points1[0,1]+N_size, points1[0,0] - N_size:points1[0,0]+N_size]
        sub2 = image2[points2[0,1] - N_size:points2[0,1]+N_size, points2[0,0] - N_size:points2[0,0]+N_size]

        #Identify the coordinate shift for sub neighborhoods. 
        shift1 = shift + points1[0,:] - [N_size,N_size]
        shift2 = points2[0,:] - [N_size,N_size]
        points1 = points1 + shift
        points2 = points2 + shift


        # Detect dominant lines
        lines11, lines12 = lineDetect(sub1,edgeThresh,N_size)
        lines21, lines22 = lineDetect(sub2,edgeThresh,N_size)
        line_time = time.time() - t

        t = time.time()
        # If no lines are found, print error and return identity
        if (lines11 is None) or (lines21 is None):
            print "Error: Lines not found", lines1, lines2
            return np.zeros((3,3)),out_obj_mask,[]

        sub1 = np.repeat(sub1[:,:,np.newaxis],3,axis=2)
        sub2 = np.repeat(sub2[:,:,np.newaxis],3,axis=2)


        lines11 = [np.cos(-np.arctan2(shift1[0,1],shift1[0,0]) + lines11[1])*np.sqrt(pow(shift1[0,0],2) + pow(shift1[0,1],2)) + lines11[0],lines11[1]]
        lines21 = [np.cos(-np.arctan2(shift2[0,1],shift2[0,0]) + lines21[1])*np.sqrt(pow(shift2[0,0],2) + pow(shift2[0,1],2)) + lines21[0],lines21[1]]
        if lines12:
            lines12 = [np.cos(-np.arctan2(shift1[0,1],shift1[0,0]) + lines12[1])*np.sqrt(pow(shift1[0,0],2) + pow(shift1[0,1],2)) + lines12[0],lines12[1]]
        if lines22:
            lines22 = [np.cos(-np.arctan2(shift2[0,1],shift2[0,0]) + lines22[1])*np.sqrt(pow(shift2[0,0],2) + pow(shift2[0,1],2)) + lines22[0],lines22[1]]
        line2_time = time.time() - t

        t = time.time()
        lines11,lines22,lines12,lines21 = pairLines([lines11,lines12],[lines21,lines22],H,main_seam)
        line3_time = time.time() - t

        print "Lines 1: ",lines11,lines12
        print "Lines 2: ",lines21,lines22
        t = time.time()
        #Ensure points lie on a line.
        corr_pts1,corr_pts2 = computeTrackingPoints(lines11,lines12,sub1,shift1)
        print "Points1: ", points1, corr_pts1
        print "Points2: ", points2, corr_pts2
        #cv2.imshow("sub1",sub1)
        #cv2.imshow("sub2",sub2)
        #cv2.waitKey(0)

        points = np.zeros((4,2))

        print "Points: ",points
        points[0,0], points[0,1] = correctPoint(points1[0,0],points1[0,1],lines11)
        points[2,0],points[2,1] = correctPoint(points2[0,0],points2[0,1],lines21)
        #points[0,0], points[0,1] = correctPoint(lines11,transformed_side_seam)
        if lines12:
            print "Adjusting for lines12"
            points[1,0],points[1,1] = correctPoint(points1[0,0],points1[0,1],lines12)
            points[3,0],points[3,1] = correctPoint(points2[0,0],points2[0,1],lines12)
            #points[1,0], points[1,1] = correctPoint(lines12,transformed_side_seam)
            #x,y = correctPoint(points1[0,0],points1[0,1],lines12)
            #print"XY: ", x,y
        print "Points: ",points
        points = np.round(points).astype('int')
        print "Points: ",points
        #points = points.astype('uint8')
        correction_time = time.time() - t

        
        t = time.time()
        main_view_pts =  np.zeros((4,2))
        side_view_pts = np.zeros((4,2))

        main_view_pts[0,:] = points[0,:] #+ np.mat([shift[0],shift[1]])
        main_view_pts[1,:] = points[1,:] #+ np.mat([shift[0],shift[1]])
        main_view_pts[3,:] = lineEdge(idx,lines11,main_seam,transformed_side_border) #+ np.mat([shift[0],shift[1]])
        main_view_pts[2,:] = lineEdge(idx,lines12,main_seam,transformed_side_border) #+ np.mat([shift[0],shift[1]])

        side_view_pts[0,:] = epiMatch((main_view_pts[0,0],main_view_pts[0,1]),lines21,F)
        side_view_pts[1,:] = epiMatch((main_view_pts[1,0],main_view_pts[1,1]),lines22,F)
        
        #side_view_pts[0,:] = lineEdge(idx,lines21,1 - side_seam,side_border)
        #side_view_pts[1,:] = lineEdge(idx,lines22,1 - side_seam,side_border)
        #side_view_pts[0,:] = points[2,:]
        #side_view_pts[1,:] = points[3,:]
        side_view_pts[3,:] = lineEdge(idx,lines21,side_seam,side_border)
        side_view_pts[2,:] = lineEdge(idx,lines22,side_seam,side_border)

        main_view_pts = main_view_pts.astype('int')
        side_view_pts = side_view_pts.astype('int')


        # Check to ensure that epipolar matching places lines inside image
        n = side_border.shape
        # Check Point 0
        if side_view_pts[0,0] > n[0]:
            rho = lines21[0]
            theta = lines21[1]
            x = n[0] - 10
            y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)
            side_view_pts[0,:] = (x,y)
        if side_view_pts[0,1] > n[1]:
            rho = lines21[0]
            theta = lines21[1]
            y = n[1] - 10
            x =-np.sin(theta)/np.cos(theta)*y + rho/np.cos(theta)
        if side_view_pts[0,0] < 0:
            rho = lines21[0]
            theta = lines21[1]
            x = 10
            y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)
            side_view_pts[0,:] = (x,y)
        if side_view_pts[0,1] < 0:
            rho = lines21[0]
            theta = lines21[1]
            y = 10
            x =-np.sin(theta)/np.cos(theta)*y + rho/np.cos(theta)
            side_view_pts[0,:] = (x,y)

        #Check Point 1
        if side_view_pts[1,0] > n[0]:
            rho = lines22[0]
            theta = lines22[1]
            x = n[0] - 10
            y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)
            side_view_pts[1,:] = (x,y)
        if side_view_pts[1,1] > n[1]:
            rho = lines22[0]
            theta = lines22[1]
            y = n[1] - 10
            x = -np.sin(theta)/np.cos(theta)*y + rho/np.cos(theta)
            side_view_pts[1,:] = (x,y)
        if side_view_pts[1,0] < 0:
            rho = lines22[0]
            theta = lines22[1]
            x = 10
            y = -np.cos(theta)/np.sin(theta)*x + rho/np.sin(theta)
            side_view_pts[1,:] = (x,y)
        if side_view_pts[1,1] < 0:
            rho = lines22[0]
            theta = lines22[1]
            y = 10
            x =-np.sin(theta)/np.cos(theta)*y + rho/np.cos(theta)
            side_view_pts[1,:] = (x,y)


        point_time = time.time() - t
        print "Shift: ",shift
        print "Points: ",points
        print "Main View Points: ",main_view_pts
        print "Side View Points: ",side_view_pts
        print "DONE!"

        # Place object Mask Here
        pts = main_view_pts.reshape((-1,1,2))
        out_obj_mask = np.zeros((main_seam.shape[0],main_seam.shape[1],3))
        print pts.shape,pts.dtype
        print out_obj_mask.shape
        cv2.fillPoly(out_obj_mask,[pts],(1,1,1))

        # Perform Feature Matching
        points1 = (main_view_pts[0,0],main_view_pts[0,1])
        points2 = (side_view_pts[0,0],side_view_pts[0,1])
        mat1 = (main_view_pts[1,0],main_view_pts[1,1])
        mat2 = (side_view_pts[1,0],side_view_pts[1,1])

        # Display detected 
        if DRAW_LINES:
                ptsB = np.mat([points1,mat2])
                ptsA = np.mat([mat1,points2])
                #tmp_image1 = np.repeat(image1[:, :, np.newaxis], 3, axis=2)
                tmp_image2 = np.repeat(image2[:, :, np.newaxis], 3, axis=2)
                #tmp_image1 = np.copy(100*main_seam).astype('uint8')
                tmp_image1 = 50*np.ones((main_seam.shape[0], main_seam.shape[1],3)).astype('uint8')
                tmp_image1[0:image1.shape[0],0:image1.shape[1],:] = np.repeat(image1[:,:,np.newaxis],3,axis=2).astype('uint8')
                #print tmp_image1.shape

                tmp_image1 = np.pad(tmp_image1,((shift[1],0),(shift[0],0),(0,0)),'constant',constant_values = 0)



                drawLines(lines11[0],lines11[1],tmp_image1)
                drawLines(lines21[0],lines21[1],tmp_image2)
                if lines12:
                    drawLines(lines12[0],lines12[1],tmp_image1)
                if lines22:
                    drawLines(lines22[0],lines22[1],tmp_image2)


                
                #cv2.circle(tmp_image1, mat2, 5, (255,0,0))
                #cv2.circle(tmp_image2, mat1, 5, (0,255,0))
                #cv2.circle(tmp_image1, points1, 5, (0,255,0))
                #cv2.circle(tmp_image2, points2, 5, (255,0,0))

                cv2.circle(tmp_image1, (main_view_pts[0,0],main_view_pts[0,1]), 7, (255,0,0),3)
                cv2.circle(tmp_image1, (main_view_pts[1,0],main_view_pts[1,1]), 7, (0,255,0),3)
                cv2.circle(tmp_image1, (main_view_pts[2,0],main_view_pts[2,1]), 7, (0,0,255),3)
                cv2.circle(tmp_image1, (main_view_pts[3,0],main_view_pts[3,1]), 7, (255,0,255),3)

                cv2.circle(tmp_image2, (side_view_pts[0,0],side_view_pts[0,1]), 7, (255,0,0),3)
                cv2.circle(tmp_image2, (side_view_pts[1,0],side_view_pts[1,1]), 7, (0,255,0),3)
                cv2.circle(tmp_image2, (side_view_pts[2,0],side_view_pts[2,1]), 7, (0,0,255),3)
                cv2.circle(tmp_image2, (side_view_pts[3,0],side_view_pts[3,1]), 7, (255,0,255),3)

                #print points1,points2
                print "PtsA: ",ptsA
                print "PtsB: ",ptsB
                #linesA = cv2.computeCorrespondEpilines(ptsA, 2,F)
                #linesB = cv2.computeCorrespondEpilines(ptsB, 1,F)
                #linesA = linesA.reshape(-1,3)
                #linesB = linesB.reshape(-1,3)
                #img5,img6 = drawEpilines(tmp_image1,tmp_image2,linesA,linesB,ptsB,ptsA)

                cv2.imshow("image1 (2 edge) "+str(idx),tmp_image1)
                cv2.imshow("image2 (2 edge) "+str(idx),tmp_image2)
                #cv2.waitKey(0)
                #cv2.destroyWindow("image1 (2 edge)")
                #cv2.destroyWindow("image2 (2 edge)")

        t = time.time()
        if (main_view_pts[2,:] == (-10,-10)).all() or (main_view_pts[3,:] == (-10,-10)).all() or (side_view_pts[2,:] == (-10,-10)).all() or (side_view_pts[3,:] == (-10,-10)).all():
            print "ERROR: Unable to detect line edge intersection",
            print "Main view points: ", main_view_pts
            print "Side view points: ",side_view_pts
            shift_mat = np.eye(3)
            shift_mat[0,2] = shift[0]
            shift_mat[1,2] = shift[1]
            return np.dot(shift_mat,H), out_obj_mask,side_view_pts

        if lines12:
            if lines22:
                if np.all(lines12 == lines22):
                    LineT = matchLines(points1, mat1, points2, mat2)
                else:
                    (LineT,status) = cv2.findHomography(side_view_pts,main_view_pts)
        else:
            LineT = matchLines(points1, mat1, points2, mat2)
        trans_time = time.time() - t

        # Print timing info
        file.write("Line Detection 1: " + str(line_time) + "\n")
        file.write("Line Detection 2: " + str(line2_time) + "\n")
        file.write("Line Detection 3: " + str(line3_time) + "\n")
        file.write("Point Correction: " + str(correction_time) + "\n")
        file.write("Point Choosing: " + str(point_time) + "\n")
        file.write("Transformation Calculation: " + str(trans_time) + "\n")


        file.close()
        #print "Proposed PtsB: ",np.dot(LineT,np.transpose(np.mat([[main_view_pts[0,0],main_view_pts[0,1],1],[main_view_pts[1,0],main_view_pts[1,1],1],[main_view_pts[2,0],main_view_pts[2,1],1],[main_view_pts[3,0],main_view_pts[3,1],1]])))
        #print "Actual Pts B: ",side_view_pts
        print "LineT: ",LineT
        return LineT,out_obj_mask,side_view_pts,corr_pts1,corr_pts2,lines11,lines12