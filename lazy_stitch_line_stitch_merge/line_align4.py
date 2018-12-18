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

    def __init__(self, main_view_cap, side_view_caps):
        ############################ Parameters computed during calibration #################################################################################
        self.num_side_cams = len(side_view_caps)                        # The number of cameras, excluding the main view camera
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
        self.background_models = []                                     # For each side view, an image whose pixel values are estimates of the pure background image for that camera
        self.intensity_weights = [[]] * (self.num_side_cams + 1)        # For each camera, stores the relative shift in intensity needed to normalize intensities
        self.max_weight = [[]] * 3                                      # The average pixel value for the background after normalization. 
        self.pano_finished = False                                      # A flag for determining whether final_pano has been constructed yet.
        ############################ Parameters computed during first self.stitch ##############################################################################
        self.pano_mask = []                                             # A binary image where 1 denotes the outer boundaries of the side views in the final mosaic
        self.main_mask = []                                             # A binary image where 1 denotes a location where the main view is located in the final mosaic
        self.main_seam = []                                             # A binary image where 1 denotes the overlapping section between the main view and any side view in the final mosaic
        ########################### Parameters computed once object is detected ################################################################################
        self.object_texture = []                                        # An image containing the object we wish to align
        self.object_loc = []                                            # A quadrilateral showing where the object is located in object_texture



        ## Run Calibration
        self.calibrate(main_view_cap,side_view_caps)

    def calibrate(self, main_view_cap,side_view_caps):
        # Performs the calibration step of our visualization system (See *Paper location TBD*). 


        # Loads default stitcher from stitcher.py
        sti = stitcher.Stitcher();

        ##################################################### Read Calibration Frames ####################################################################
        ret, main_view_frame = main_view_cap.read()
        side_view_frames = [[]] * self.num_side_cams
        for i in range(self.num_side_cams):
            _, side_view_frames[i] = side_view_caps[i].read()        

        ##################################################### Store image Shapes ##########################################################################
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

        ##########################################################################################################################################

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


    def stitch(self,main_view_frame,side_view_frames,models):
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

        # If background hasn't been drawn yet.
        if not self.pano_finished:
            self.pano_finished = True                                                                                               # Mark that background is being drawn
            self.main_mask = np.zeros(self.final_pano.shape,dtype = 'uint8')                                                        # Initialize mask of main view location
            self.main_seam = np.zeros(self.final_pano.shape,dtype = 'uint8')                                                        # Initialize mask for main view overlap with side views
            
            # Generate main view mask
            self.main_mask[out_pos[0]:out_pos[0]+main_view_frame.shape[0],out_pos[1]:out_pos[1]+main_view_frame.shape[1]] = 1

            # Apply Background homographies       
            for i in range(len(side_view_frames)):
                result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frames[i],self.homography_list[i])
                
                #Place transformed background on canvas
                temp_result_window = self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :]
                self.final_pano [out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+result2.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+result2.shape[1], :] = result2 * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])
                
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
            self.main_seam = self.main_seam * self.main_mask
            
            kernel = np.ones((5,5),np.uint8)
            self.main_seam[:,:,0] = cv2.dilate(self.main_seam[:,:,0],kernel,iterations=1)


        # If background has been drawn. 
        else:
            # If Object has been modeled. 
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
                obj_detected,pts1,main_view_object_mask = genMainMask(main_view_frame,models[0],self.main_seam[out_pos[0]:out_pos[0]+main_view_frame.shape[0],out_pos[1]:out_pos[1]+main_view_frame.shape[1]]*outer_edge)

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
                    tempH,post_obj_mask = lineAlignWithModel(0,pts1.astype('int'),255*main_view_object_mask,self.object_loc,self.object_texture,self.main_seam,self.pano_mask,[out_pos[1],out_pos[0]])

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
                    trans_obj = cv2.warpPerspective(self.object_texture,tempH,(out_pano.shape[1],out_pano.shape[0]))
                    out_pano = trans_obj * post_obj_mask + out_pano * (1-post_obj_mask)

                    warping_time = time.time() - t 
                    print "Object Warping: ", warping_time
                    #Save timing info
                    file = open("obj_warp_timing.txt","a")
                    file.write("Object Warping: ")
                    file.write(str(warping_time))
                    file.write("\n")
                    file.close()

            # If object hasn't been modeled.
            else:

                transformed_side_obj = [[]] * len(side_view_frames)
                
                side_view_has_motion, seam_has_motion = self.read_next_frame(main_view_frame, side_view_frames)
                
                for i in range(len(side_view_frames)):
                    if side_view_has_motion[i]:
                        # Perform object alignment
                        (_, transformed_side_bg,transformed_side_obj[i], _, side_mask, line_shift, _) = self.line_stitch(main_view_frame, side_view_frames[i], i,models[0],models[i+1])

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


        return out_pano.astype('uint8')

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

                            main_seam, side_seam, side_border,transformed_side_border = genBorderMasks(main_view_frame, side_view_frame, mask1,new_mask,self.homography_list[idx],shift)
                            tempH,post_obj_mask = lineAlignWithModel(idx,pts1,255*main_view_object_mask,self.object_loc,main_seam,transformed_side_border,shift,self.homography_list[idx])
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
                            tempH,post_obj_mask,self.object_loc = lineAlign(idx,pts1,255*main_view_object_mask,pts2,255*side_view_object_mask,self.fundamental_matrices_list[idx],main_seam, side_seam, side_border,transformed_side_border,shift,self.homography_list[idx])
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
    main_view_object_mask = identifyFG(main_view_frame, main_view_background).astype('uint8')
    main_view_edge = cv2.Canny(main_view_object_mask, 50, 150)
    main_view_object_mask = (main_view_object_mask/255).astype('uint8')
    kernel_gradient = np.mat([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    main_view_edge = cv2.filter2D(main_view_object_mask,-1,kernel_gradient)
    main_view_edge = (main_view_edge > 0).astype('uint8')

    pts = np.nonzero(main_view_edge * crossing_edges_main_view_list[:,:,0])
    pts = np.asarray(pts)
    if pts.shape[1] >= 2:
        clusters_main_view, cluster_err = kmean(pts, 2)
        pts1 = np.mat([[clusters_main_view[1,0], clusters_main_view[0,0]], [clusters_main_view[1,1], clusters_main_view[0,1]]])
        return True,pts1,main_view_object_mask

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

    if (len(frame_mask.shape) == 2):
        frame_mask = np.repeat(frame_mask[:,:,np.newaxis],3,axis=2)
    if (len(border_mask.shape) == 2):
        border_mask = np.repeat(border_mask[:,:,np.newaxis],3,axis=2)

    line_mask = np.zeros(frame_mask.shape)
    drawLines(line[0],line[1],line_mask,(1,1,1),1)

    #kernel = np.ones((5,5),np.uint8)
    #frame_mask[:,:,0] = cv2.dilate(frame_mask[:,:,0],kernel,iterations=1)

    possible_points = np.nonzero(line_mask[:,:,0]*border_mask[:,:,0]*(1-frame_mask[:,:,0]))

    #if idx == 0:
    #temp = np.nonzero(line_mask[:,:,0]*border_mask[:,:,0])
    #print "line,border: ",temp
    #print "frame_mask: ", frame_mask[temp[0][0],temp[1][0],0]
        #temp_display = np.zeros(line_mask.shape)
        #temp_display[:,:,0] = line_mask[:,:,0]
        #temp_display[:,:,1] = border_mask[:,:,0]*(1-frame_mask[:,:,0])
        #temp_display[:,:,2] = (1- frame_mask[:,:,0])
    #R = np.zeros([256,256,3])
    #R[:,:,1] = 255
    #cv2.imshow("R",R)
        #cv2.imshow("temp display",temp_display)
        #cv2.waitKey(0)
    #cv2.destroyWindow("temp display")

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
            cv2.destroyWindow("frame 0")
            cv2.destroyWindow("frame 1")
            cv2.destroyWindow("frame 2")
            cv2.destroyWindow("frame 3")
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

#def warpObject(main_view_frame, side_view_frame, side_view_object_mask, tempH, main_homography, sti, result1, mask1, result2, shift, new_mask, trans_matrix):
    

#    corners = np.mat([[0, side_view_frame.shape[1], 0, side_view_frame.shape[1]], [0, 0, side_view_frame.shape[0], side_view_frame.shape[0]], [1, 1, 1, 1]])
#    transformed_corners = np.dot(tempH, corners)
#    bound = np.zeros((2, 4), np.float)
#    bound[0,0] = transformed_corners[0,0] / transformed_corners[2,0]
#    bound[1,0] = transformed_corners[1,0] / transformed_corners[2,0]
#    bound[0,1] = transformed_corners[0,1] / transformed_corners[2,1]
#    bound[1,1] = transformed_corners[1,1] / transformed_corners[2,1]
#    bound[0,2] = transformed_corners[0,2] / transformed_corners[2,2]
#    bound[1,2] = transformed_corners[1,2] / transformed_corners[2,2]
#    bound[0,3] = transformed_corners[0,3] / transformed_corners[2,3]
#    bound[1,3] = transformed_corners[1,3] / transformed_corners[2,3]

#    if (np.max(bound[0,:]) - np.min(bound[0,:])) <= 2500 and (np.max(bound[1,:]) - np.min(bound[1,:])) < 2500 and (tempH is not np.zeros((3,3))):
        #############################################################################
#        result1_temp,result2_temp,_,mask2_temp, shift_temp, _ = sti.applyHomography(main_view_frame, side_view_frame, tempH)
#        mask_temp_temp = (result1_temp == 0).astype('int')
#        result2_temp_shape = result2_temp.shape

#        OUTPUT_SIZE = [3000,3000,3]
#        out_pos = np.array([OUTPUT_SIZE[0]/2-500,OUTPUT_SIZE[1]/2-500]).astype('int')
#        pano = np.zeros(OUTPUT_SIZE, np.uint8)

#        window = pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp.shape[1],:]

#        if window.shape[0] != mask2_temp.shape[0] or window.shape[1] != mask2_temp.shape[1]:
#            return result1,result2,mask1,new_mask, shift, trans_matrix

#        print out_pos, shift_temp, result1.shape

#        pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp.shape[1],:] = 0+result2_temp
#        result2_temp = 0+pano[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]
        #cv2.imshow('pano_2', result2_temp)
        #cv2.waitKey(0)
#        _,mask,_,_, _, _ = sti.applyHomography(main_view_frame, np.stack((side_view_object_mask,side_view_object_mask,side_view_object_mask), axis=2), tempH)

#        pano_mask = np.zeros(OUTPUT_SIZE, np.uint8)
#        pano_mask[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+result2_temp_shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+result2_temp_shape[1],:] = 0+mask
#        mask = 0+pano_mask[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]

#        pano[out_pos[0]-shift_temp[1]:out_pos[0]-shift_temp[1]+mask2_temp.shape[0],out_pos[1]-shift_temp[0]:out_pos[1]-shift_temp[0]+mask2_temp.shape[1],:] = 0+mask2_temp
#        mask2_temp = 0+pano[out_pos[0]-shift[1]:out_pos[0]-shift[1]+result1.shape[0],out_pos[1]-shift[0]:out_pos[1]-shift[0]+result1.shape[1],:]


#        mask = mask.astype('uint8') * mask2_temp
        #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_opening_closing)
#        mask = mask * (result2 > 0).astype('uint8')
        #cv2.imshow('mask',mask*255)

#        _,mask_object_transformed,_,_, _, _ = sti.applyHomography(main_view_frame, np.stack((side_view_object_mask,side_view_object_mask,side_view_object_mask), axis=2), main_homography)
        #cv2.imshow('mask_object_transformed',mask_object_transformed*255)
#        Background_mask = new_mask * (mask_object_transformed == 0).astype('uint8')
        #cv2.imshow('Background_mask',Background_mask.astype('uint8')*255)
#        color1 = np.sum(result2[:,:,0] * Background_mask[:,:,0]) / np.sum(Background_mask[:,:,0])
#        color2 = np.sum(result2[:,:,1] * Background_mask[:,:,1]) / np.sum(Background_mask[:,:,1])
#        color3 = np.sum(result2[:,:,2] * Background_mask[:,:,2]) / np.sum(Background_mask[:,:,2])
#        temp = np.ones(mask.shape, np.uint8)
#        temp[:,:,0] = color1
#        temp[:,:,1] = color2
#        temp[:,:,2] = color3

#        result2 = Background_mask.astype('uint8') * result2 + (result2 > 0).astype('uint8') * mask_object_transformed * temp
        #cv2.imshow('result2',result2)
#        result2 = result2 * np.logical_not(mask) + result2_temp * mask
        #cv2.imshow('result2_2',result2)
        #cv2.imshow('result2 * np.logical_not(mask)',result2 * np.logical_not(mask))
        #cv2.waitKey(0)
        #mask_temp_temp = (result1 == 0).astype('int')
        #temp = (result2*mask_temp_temp + result1).astype('uint8')
        #############################################################################
#        return result1,result2,mask1,new_mask, shift, trans_matrix


#    return result1,[],[],[],[],[]

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

        if idx == 2:
            cv2.imshow("obj mask "+str(idx),255*side_view_object_mask.astype('uint8'))
            cv2.imshow('back fill '+str(idx),background_fill)
            cv2.imshow("result2 "+str(idx),result2)
            cv2.waitKey(0)

    
        #result1,result2,mask1,new_mask,shift,trans_mat = stitch.applyHomography(main_frame,side_frame,np.linalg.inv(tempH))
        #pano2 = result2*(1 - mask1.astype('uint8'))+result1*mask1.astype('uint8')

    print background_fill.shape
    print trans_obj.shape
    return background_fill, trans_obj

def lineAlignWithModel(idx,points1,image1,side_view_pts,image2,main_seam,transformed_side_border,shift,N_size = 40, edgeThresh = 70,DRAW_LINES = True):
    # For Timing info
    file = open("test.txt",'a')

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
    shift1 = shift + points1[0,:] - [N_size,N_size]
    points1 = points1 + shift

    # Detect dominant lines
    print sub1.shape
    lines11, lines12 = lineDetect(sub1,edgeThresh,N_size)
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

    print pts.shape, pts.dtype
    print out_obj_mask.shape
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


    t = time.time()
    if (main_view_pts[2,:] == (-10,-10)).all() or (main_view_pts[3,:] == (-10,-10)).all() or (side_view_pts[2,:] == (-10,-10)).all() or (side_view_pts[3,:] == (-10,-10)).all():
        print "ERROR: Unable to detect line edge intersection",
        print "Main view points: ", main_view_pts
        print "Side view points: ",side_view_pts
        shift_mat = np.eye(3)
        shift_mat[0,2] = shift[0]
        shift_mat[1,2] = shift[1]
        return np.eye(3), out_obj_mask

    
    (LineT,status) = cv2.findHomography(side_view_pts,main_view_pts)
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
        #print points1, correctPoint(points1[0,0],points1[0,1],lines1)
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
        return LineT,out_obj_mask,side_view_pts