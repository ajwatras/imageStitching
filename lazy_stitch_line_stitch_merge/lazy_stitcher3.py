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
    def __init__(self, main_view_frame, side_view_frames):
        sti = stitcher.Stitcher();

        main_view_cal_image = main_view_frame
        self.main_view_image_shape = main_view_cal_image.shape

        side_view_cal_images = side_view_frames
        self.side_view_image_shape = [];
        for i in range(len(side_view_frames)):
            self.side_view_image_shape.append(side_view_cal_images[i].shape)

        ##############################################
        self.homography_list = [];
        self.coord_shift_list = [];
        self.fundamental_matrices_list = [];
        for i in range(len(side_view_frames)):
            (_, _, H, _, _, coord_shift) = sti.stitch([main_view_cal_image, side_view_cal_images[i]], showMatches=True)

            self.homography_list.append(H)
            self.coord_shift_list.append(coord_shift)
            self.fundamental_matrices_list.append(la.calcF(main_view_cal_image, side_view_cal_images[i],i+1))
            #self.fundamental_matrices_list.append(la.calcF(main_view_cal_image, side_view_cal_images[i]))

        ##############################################
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

        ##############################################
        pts = np.nonzero(pano)
        pts = np.asarray(pts)
        left_most = np.min(pts[1,:])-1
        right_most = np.max(pts[1,:])+1
        up_most = np.min(pts[0,:])-1
        down_most = np.max(pts[0,:])+1
        self.main_view_upleft_coord = [out_pos[0] - up_most, out_pos[1] - left_most]
        self.final_pano = np.zeros((pano[up_most:down_most, left_most:right_most, :]).shape, np.uint8)

        ##############################################
        self.transformed_mask_side_view = [];
        self.masks_side_view = [];
        kernel_opening_closing = np.ones((5,5),np.uint8)
        for i in range(len(side_view_frames)):
            transformed_mask = pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_image_shapes_list[i][0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_image_shapes_list[i][1], :]
            transformed_mask = cv2.morphologyEx((transformed_mask == (i + 2)).astype('uint8'), cv2.MORPH_OPEN, kernel_opening_closing)
            self.transformed_mask_side_view.append(transformed_mask)
            self.masks_side_view.append(cv2.warpPerspective(self.transformed_mask_side_view[i], inv(trans_matrices_list[i]), (self.side_view_image_shape[i][1], self.side_view_image_shape[i][0])))

        ##############################################
        self.seam = np.zeros((self.main_view_image_shape[0], self.main_view_image_shape[1]))
        #temp_seam = np.zeros((self.main_view_image_shape[0], self.main_view_image_shape[1]))
        kernel = np.ones((50,50),np.uint8)
        for i in range(len(side_view_frames), 0, -1):
            self.seam = self.seam + (self.seam == 0) * (cv2.dilate(seams_main_view_list[i-1], kernel, iterations = 1) * i)
            #temp_seam = temp_seam + (temp_seam == 0) * (seams_main_view_list[i-1] * i)

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

    def stitch(self, main_view_frame, side_view_frames,models):
        sti = stitcher.Stitcher();
        out_pos = self.main_view_upleft_coord
        side_view_has_motion, seam_has_motion = self.read_next_frame(main_view_frame, side_view_frames)
        print side_view_has_motion, seam_has_motion
        for i in range(len(side_view_frames)):
            if side_view_has_motion[i]:
                (_, transformed_side_view, _, side_mask, line_shift, _) = self.line_stitch(main_view_frame, side_view_frames[i], i,models[0],models[i+1])
                # Comment when updating to line_align_3
                temp_result_window = self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_view.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_view.shape[1], :]
                self.final_pano[out_pos[0]-self.coord_shift_list[i][0]:out_pos[0]-self.coord_shift_list[i][0]+transformed_side_view.shape[0], out_pos[1]-self.coord_shift_list[i][1]:out_pos[1]-self.coord_shift_list[i][1]+transformed_side_view.shape[1], :] = transformed_side_view * self.transformed_mask_side_view[i] + temp_result_window * np.logical_not(self.transformed_mask_side_view[i])

                ## Uncomment when updating to line_align_3
                #temp_shift = [0,0]
                #temp_shift[0] = out_pos[0] - self.coord_shift_list[i][0] + line_shift[0]
                #temp_shift[1] = out_pos[1] - self.coord_shift_list[i][1] + line_shift[1]
                #self.final_pano = la.placeFrame(self.final_pano, transformed_side_view, side_mask, temp_shift )
                #if True: #seam_has_motion[i]:
                    #self.line_stitch(main_view_frame, side_view_frames[i], i)
        self.final_pano[out_pos[0]:out_pos[0]+self.main_view_image_shape[0], out_pos[1]:out_pos[1]+self.main_view_image_shape[1],:] = main_view_frame

        return self.final_pano, main_view_frame, side_view_frames

    def line_stitch(self, main_view_frame, side_view_frame, idx,main_view_background,side_view_background):
        sti = stitcher.Stitcher()
        result1,result2,mask1,mask2_original, shift, trans_matrix = sti.applyHomography(main_view_frame,side_view_frame,self.homography_list[idx])
        new_mask = (result2 > 0).astype('uint8')
        file = open("obj_det_timing.txt", "a")

        #main_view_edge = cv2.Canny(main_view_frame, 50, 150)
        #kernel_opening_closing = np.ones((30,30),np.uint8)
        #main_view_object_mask = (cv2.morphologyEx(main_view_edge, cv2.MORPH_CLOSE, kernel_opening_closing)/255).astype('uint8')
        #kernel_gradient = np.mat([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        #main_view_edge = cv2.filter2D(main_view_object_mask,-1,kernel_gradient)
        #main_view_edge = (main_view_edge > 0).astype('uint8')
        pano1 = result2*(1 - mask1.astype('uint8')) + result1*mask1.astype('uint8')
        #cv2.imshow("Main",main_view_frame)
        #cv2.imshow("Side",side_view_frame)
        #cv2.imshow("Pre-Correction",pano1)



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




        if obj_detected:

                            #cv2.imshow("Original",side_view_frame)
                            #cv2.imshow("Object",255*side_view_object_mask.astype('uint8'))
                            #cv2.imshow("Edge",cv2.Canny(255*side_view_object_mask.astype('uint8'),20,60).astype('uint8') )
                            ### Perform Alignment and save timing info ###
                            file = open("obj_align_timing.txt","a")
                            t = time.time()
                            # Detect Main View location in side view
                            side_view_main_mask = la.mapCoveredSide(self.homography_list[idx],main_view_frame,side_view_frame)
                            main_seam, side_seam, side_border,transformed_side_border = la.genBorderMasks(main_view_frame, side_view_frame, mask1,mask2_original,self.homography_list[idx])
                            #tempH = la.lineAlign(pts1,main_view_frame,pts2,side_view_frame,self.fundamental_matrices_list[idx])
                            tempH = la.lineAlign(idx,pts1,255*main_view_object_mask,pts2,255*side_view_object_mask,self.fundamental_matrices_list[idx],main_seam, side_seam, side_border,transformed_side_border,shift,self.homography_list[idx])
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
                            
                            result2 = la.warpObject(idx,side_view_object_mask,tempH,result2,side_view_frame,side_view_background,self.homography_list[idx],shift)
                            #result1,result2,mask1,new_mask,shift,trans_mat = stitch.applyHomography(main_frame,side_frame,np.linalg.inv(tempH))

                            warping_time = time.time() - t 
                            print "Object Warping: ", warping_time
                            file.write("Object Warping: ")
                            file.write(str(warping_time))
                            file.write("\n")
                            file.close()

                            #pano2 = result2*(1 - mask1.astype('uint8'))+result1*mask1.astype('uint8')
                            #cv2.imshow("Post-Correction",pano2)
                            #cv2.waitKey(0)
                            #cv2.destroyWindow("Pre-Correction")
                            #cv2.destroyWindow("Post-Correction")
                            
        return result1,result2,mask1,new_mask, shift, trans_matrix


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
