import cv2
import numpy as np
import line_align as la
import stitcher as sti
import time

# Generate backgrounds
main_frame = 100*np.ones((480,640,3)).astype('uint8')
side_frame = 100*np.ones((480,640,3)).astype('uint8')
background_model = 100*np.ones((480,640,3)).astype('uint8')

# Identify Homography relationship
corners = np.mat([[0,0],[0,side_frame.shape[1]],[side_frame.shape[0],0],[side_frame.shape[0],side_frame.shape[1]]])
#new_corners = np.mat([[150,-100],[150,side_frame.shape[1]-100],[side_frame.shape[0]+150,-100],[150+side_frame.shape[0],-100+side_frame.shape[1]]])
new_corners = np.mat([[-150,100],[-150,side_frame.shape[1]+100],[side_frame.shape[0]-150,100],[-150+side_frame.shape[0],100+side_frame.shape[1]]])
#new_corners = np.mat([[100,100],[100,side_frame.shape[1]+100],[side_frame.shape[0]+100,100],[100+side_frame.shape[0],100+side_frame.shape[1]]])

H,h_mask = cv2.findHomography(corners,new_corners)
H2 = np.mat([[0,1,200],[1,0,200],[0,0,1]])

# Identify desired object line
line1 = (-200,-np.pi)
line2 = (-200,-np.pi)
#line2 = (50,np.pi/2)

# Generate F
fundamental_matrices_list = []
fundamental_matrices_list.append(la.calcF(main_frame, side_frame,1))
print fundamental_matrices_list
# draw object
la.drawLines(line1[0],line1[1],main_frame,(255,255,255),30)
la.drawLines(line2[0],line2[1],side_frame,(255,255,255),30 )

cv2.imshow("main_frame",main_frame)
cv2.imshow("side_frame",side_frame)
cv2.waitKey(0)
cv2.destroyWindow("main_frame")
cv2.destroyWindow("side_frame")
#stitch object
stitch = sti.Stitcher()
result1,result2,mask1,mask2,shift,trans_mat = stitch.applyHomography(main_frame,side_frame,H)
pano = result2*(1 - mask1.astype('uint8'))+result1*mask1.astype('uint8')
#cv2.imshow("mask1",255*mask1.astype('uint8'))
#cv2.imshow("mask2",255*mask2.astype('uint8'))
#cv2.imshow("mask3",255*(mask2.astype('uint8')*mask1.astype('uint8')))
#cv2.waitKey(0)
#cv2.destroyWindow("mask1")
#cv2.destroyWindow("mask2")
#cv2.destroyWindow("mask3")
cv2.imshow("pano",pano)
cv2.waitKey(0)
cv2.destroyWindow("pano")

# generate main_view edge
main_view_edge = stitch.mapMainView(H,main_frame,side_frame)
side_view_edge = stitch.mapSeams(H,main_frame,side_frame)
print "Shift: ",shift
#shift = [shift[1],shift[0]]
# generate object masks
H_list = [H]
M_edge_list = [main_view_edge]
S_edge_list = [side_view_edge]
t=time.time()
obj_detected,pts1,pts2,main_view_object_mask,side_view_object_mask = la.genObjMask(0,main_frame, background_model,side_frame, background_model, M_edge_list,S_edge_list, H_list)
detect_time = time.time() - t
print "Object Detection: ", detect_time

if obj_detected:
    ### Perform Alignment and save timing info ###
    t = time.time()
    # Detect Main View location in side view
    #side_view_main_mask = la.mapCoveredSide(H,main_frame,side_frame)
    main_seam, side_seam, side_border,transformed_side_border = la.genBorderMasks(main_frame, side_frame, mask1,mask2,H,shift)            
    tempH = la.lineAlign(0,pts1,255*main_view_object_mask,pts2,255*side_view_object_mask,fundamental_matrices_list[0],main_seam, side_seam, side_border,transformed_side_border,shift,H)
    align_time = time.time() - t 
    #print "Object_Alignment: ", align_time
    print "tempH: ",tempH
    
    ### Perform warping and save timing info ###
    t = time.time()
    result2 = la.warpObject(0,side_view_object_mask,tempH,result2,side_frame,background_model,H,shift)
    #result1,result2,mask1,new_mask,shift,trans_mat = stitch.applyHomography(main_frame,side_frame,tempH)
    pano2 = result2*(1 - mask1.astype('uint8'))+result1*mask1.astype('uint8')
    #cv2.imshow("result1",result1)
    #cv2.imshow("result2",result2)
    #cv2.waitKey(0)
    #tempH = la.lineAlign(pts1,main_frame,pts2,side_frame,fundamental_matrices_list[0])
    #result1,result2,mask1,new_mask, shift, trans_matrix = la.warpObject(main_view_frame, side_view_frame, side_view_object_mask, side_view_background, tempH, self.homography_list[idx], sti,result1,mask1,result2,shift, new_mask, trans_matrix)

    #result1,result2,mask1,new_mask, shift, trans_matrix = la.warpObject(main_frame, side_frame, side_view_object_mask, tempH, H, stitch,result1,mask1,result2,shift, mask2, trans_mat)
    #print "Mask1: ",mask1.shape,result1.shape,result2.shape
    #pano2 = result2*(1 - mask1.astype('uint8'))+result1*mask1.astype('uint8')
    warping_time = time.time() - t 
    print "Object Warping: ", warping_time

cv2.imshow('pano1',pano)
cv2.imshow("final",pano2)
cv2.waitKey(0)
