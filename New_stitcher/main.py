import stitcher2 as stitcher
import cv2
import numpy as np
import line_align as la
import time
import threading
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString
from shapely.geometry import LinearRing


#class warping_thread(threading.Thread):
#	def __init__(self,idx):
#		threading.Thread.__init__(self)
#		self.idx = idx
#		self.sti = stitcher2.Stitcher()
#		self.H = self.is_end = False
#	def run(self, main_view_frame,side_view_frame, H):
#
#		(transformed_main_view, transformed_side_view[i], _, _, _, _) = self.sti.applyHomography(main_view_frame, side_view_frames[i], self.H)

#		return



class LineStitcher():
	def __init__(self, main_view_caps, side_view_caps):
		#Line Stitcher serves to perform 
		self.main_view_caps = main_view_caps
		self.side_view_caps = side_view_caps
		self.num_side_cams = len(self.side_view_caps)


		#_, main_view_frame = self.main_view_caps.read()
		#side_view_frames = [[]] * self.num_side_cams
		#for i in range(self.num_side_cams):
		#	_, side_view_frames[i] = self.side_view_caps[i].read()

		self.calibrate(main_view_caps,side_view_caps)
		cv2.destroyWindow("Pano")

		for i in range(self.num_side_cams):
			print"Seams: ", self.seams[i]
		

		self.fundamental_matrices_list = []
		for i in range(self.num_side_cams):
			self.fundamental_matrices_list.append(la.calcF(main_view_frame,side_view_frames[i],i+1))
	


	def calibrate(self,main_view_caps,side_view_caps,MODEL_LENGTH = 10):
		#Perform Calibration 
		sti = stitcher.Stitcher()	# Create image stitcher

		################## Read Calibration Frames ###########################
		ret, main_view_frame = self.main_view_caps.read()
		side_view_frames = [[]] * self.num_side_cams
		for i in range(self.num_side_cams):
			_, side_view_frames[i] = self.side_view_caps[i].read()

		################## Save Frame Shapes #################################
		self.main_view_image_shape = main_view_frame.shape
		self.side_view_image_shape = []
		for i in range(len(side_view_frames)):
			self.side_view_image_shape.append(side_view_frames[i].shape)

		################### Compute initial Homographies ####################
		self.homography_list = []
		self.coord_shift_list = []
		self.transformed_image_shapes_list = []
		for i in range(len(side_view_frames)):
			(_,_,H,_,_,coord_shift) = sti.stitch([main_view_frame,side_view_frames[i]])
			result1,result2,_,_, _, trans_matrix = sti.applyHomography(main_view_frame,side_view_frames[i],H)
			self.homography_list.append(H)
			self.coord_shift_list.append(coord_shift)
			self.transformed_image_shapes_list.append(result2.shape)
		################### Compute Seam Locations ##########################
		self.seams = []	# Stores seam between main view and side view as seen by the mosaic coordinate system 

		main_view_corners = np.ones([3,4])
		main_view_corners[0,0:4] = 0
		main_view_corners[0,1:3] = self.main_view_image_shape[1]
		main_view_corners[1,0:4] = 0
		main_view_corners[1,2:4] = self.main_view_image_shape[0]

		for i in range(self.num_side_cams):
			shiftH = np.array([[1,0,self.coord_shift_list[i][1]],[0,1,self.coord_shift_list[i][0]],[0,0,1]]) #
			
			corners = np.ones([3,4])
			corners[0,0:4] = 0 
			corners[0,1:3] = self.side_view_image_shape[i][1]
			corners[1,0:4] = 0
			corners[1,2:4] = self.side_view_image_shape[i][0]

			transformed_corners = np.dot(self.homography_list[i],corners)
			transformed_corners = np.dot(shiftH,transformed_corners)
			transformed_corners[0,:] = np.divide(transformed_corners[0,:],transformed_corners[2,:]) #+ self.coord_shift_list[i][0]
			transformed_corners[1,:] = np.divide(transformed_corners[1,:],transformed_corners[2,:]) #+ self.coord_shift_list[i][1]

			transformed_main = main_view_corners
			transformed_main[0,:] = main_view_corners[0,:] + self.coord_shift_list[i][1]
			transformed_main[1,:] = main_view_corners[1,:] + self.coord_shift_list[i][0]

			print "Transformed Corners: ", np.round(transformed_corners)
			print "Transformed Main: ", np.round(transformed_main)
			#Intersect Images
			#If using la.findSeams, comment out the following three lines and uncomment fourth. 
			side_poly = Polygon([(transformed_corners[0,0],transformed_corners[1,0]),(transformed_corners[0,1],transformed_corners[1,1]),(transformed_corners[0,2],transformed_corners[1,2]),(transformed_corners[0,3],transformed_corners[1,3])])
			main_lines = LinearRing([(transformed_main[0,0],transformed_main[1,0]),(transformed_main[0,1],transformed_main[1,1]),(transformed_main[0,2],transformed_main[1,2]),(transformed_main[0,3],transformed_main[1,3])])
			self.seams.append(main_lines.intersection(side_poly))
			#self.seams.append(la.findSeams(transformed_main,transformed_corners))

			# Compute mosaic size
			#concat_corners = np.hstack((transformed_main,transformed_corners))
			#print "concat_corners: ", np.round(concat_corners)
			#x_size = int(np.round(max(concat_corners[0,:]))) #- min(np.hstack(concat_corners[0,:]))))
			#y_size = int(np.round(max(concat_corners[1,:]))) #- min(np.hstack(concat_corners[1,:]))))
			#self.transformed_image_shapes_list.append((y_size,x_size))
			#print "Shapes: ",(y_size,x_size), self.transformed_image_shapes_list 

		################### Compute Canvas Size  ##############################
		size_x = []
		size_y = []
		shift_x = []
		shift_y = []
		self.frame_windows = [[]] * self.num_side_cams

		for i in range(self.num_side_cams):
			size_x.append(self.transformed_image_shapes_list[i][1])
			size_y.append(self.transformed_image_shapes_list[i][0])
			shift_x.append(self.coord_shift_list[i][1])
			shift_y.append(self.coord_shift_list[i][0])

		self.max_shift = np.array([max(shift_y),max(shift_x)])
		self.canvas_size = (max(size_y)+max(shift_y),max(size_x)+max(shift_x),3)

		for i in range(self.num_side_cams):
			up_left = self.max_shift - self.coord_shift_list[i]
			self.frame_windows[i] = [up_left[0], up_left[0]+self.transformed_image_shapes_list[i][0],up_left[1], up_left[1]+self.transformed_image_shapes_list[i][1]]


		#print size_x
		#print size_y
		#print shift_x
		#print shift_y
		#print self.canvas_size


		################### Compute Background Model  #####################		
		pano = np.zeros(self.canvas_size,np.uint8)		# Generate Canvas
		transformed_side_view = [[]] * self.num_side_cams
		coord_window = [0,0,0,0]
		Modeling_flag = False
		self.background_model_side = [[]] * (self.num_side_cams)
		self.background_model_main = [[]] * (self.num_side_cams)
		self.frame_masks = [np.zeros(self.canvas_size,np.uint8)] * (self.num_side_cams+1)

		cc = -1
		colors = [(255,0,0),(0,255,0),(0,0,255)]

		print "Press P when background modeling should begin or press q to quit"
		while ret:
			pano = np.zeros(self.canvas_size,np.uint8)		# Generate Canvas
			for i in range(self.num_side_cams):
				(transformed_main_view, transformed_side_view[i], _, _, _, _) = sti.applyHomography(main_view_frame, side_view_frames[i], self.homography_list[i])
				#transformed_main_view,transformed_side_view[i] = 
				#pano1 = transformed_main_view + transformed_side_view[i]*(transformed_main_view == 0)
				#drawShape(self.seams[i],pano1)
				#self.transformed_image_shapes_list[i]

				#cv2.imshow("pano1",pano1)
				#cv2.waitKey(0)

				up_left = self.max_shift - self.coord_shift_list[i]
				coord_window = [up_left[0], up_left[0]+self.transformed_image_shapes_list[i][0],up_left[1], up_left[1]+self.transformed_image_shapes_list[i][1]]
				#print "Coord window: ",coord_window
				if (cc == -1):
					self.frame_windows[i] = coord_window

				if i == 0:
					if cc == -1:
						self.frame_masks[0] = (transformed_main_view > 0).astype('uint8')
						self.frame_masks[i+1] = (transformed_main_view == 0).astype('uint8') * (transformed_side_view[i] > 0).astype('uint8')
					pano[coord_window[0]:coord_window[1],coord_window[2]:coord_window[3],:] = transformed_main_view * self.frame_masks[0] + transformed_side_view[i] * self.frame_masks[i+1]
				else: 
					if cc == -1:
						self.frame_masks[i+1] = (pano[coord_window[0]:coord_window[1],coord_window[2]:coord_window[3],:] == 0).astype('uint8') * (transformed_side_view[i] > 0).astype('uint8')
					pano[coord_window[0]:coord_window[1],coord_window[2]:coord_window[3],:] = pano[coord_window[0]:coord_window[1],coord_window[2]:coord_window[3],:] + transformed_side_view[i] * (self.frame_masks[i+1])


				if Modeling_flag == True:
					if cc == 0:
						self.background_model_main[i] = transformed_main_view/MODEL_LENGTH
						self.background_model_side[i] = transformed_side_view[i]/MODEL_LENGTH
					if cc == MODEL_LENGTH-1:
						return
					else: 
						self.background_model_main[i] = self.background_model_main[i] + transformed_main_view/MODEL_LENGTH
						self.background_model_side[i] = self.background_model_side[i] + transformed_side_view[i]/MODEL_LENGTH

				#drawShape(self.seams[i],pano,colors[i])
			
			cv2.imshow("Pano",pano)

			# Check Termination
			if cv2.waitKey(10) == ord('p'):
				print "Beginning background modeling"
				Modeling_flag = True

			if cv2.waitKey(10) == ord('q'):
				cv2.destroyWindow("Pano")
				return

			if Modeling_flag:
				print cc
				cc = cc+1

			# Read Next Frames
			ret, main_view_frame = self.main_view_caps.read()
			for i in range(self.num_side_cams):
				_, side_view_frames[i] = self.side_view_caps[i].read()

		################################# End Calibration  ###########################################	

	def stream(self):
		sti = stitcher.Stitcher()
		transformed_side_view = [[]] * self.num_side_cams
		background_time = [[]] * self.num_side_cams
		detect_time = [[]] * self.num_side_cams
		align_time = [[]] * self.num_side_cams
		warp_time = [[]] * self.num_side_cams
		blend_time = [[]] * self.num_side_cams

		# Open timing info save file
		file = open("timing.txt",'w')

		# Generate warping threads
		#sti_thread = [[]] * self.num_side_cams
		#for i in range(self.num_side_cams):
		#	sti_thread[i] = warping_thread(i)
		
		# Read initial Frames
		ret, main_view_frame = self.main_view_caps.read()
		side_view_frames = [[]] * self.num_side_cams
		for i in range(self.num_side_cams):
			_, side_view_frames[i] = self.side_view_caps[i].read()
			
		while ret:
			t_frame_start = time.time()
			canvas = np.zeros(self.canvas_size, np.uint8)
			for i in range(self.num_side_cams):
				# Apply background Homography
				t = time.time()
				(transformed_main_view, transformed_side_view[i], _, _, _, _) = sti.applyHomography(main_view_frame, side_view_frames[i], self.homography_list[i])
				background_time[i] = time.time() - t
				t = time.time()


				if i == 0:
					canvas[self.frame_windows[i][0]:self.frame_windows[i][1],self.frame_windows[i][2]:self.frame_windows[i][3]] = canvas[self.frame_windows[i][0]:self.frame_windows[i][1],self.frame_windows[i][2]:self.frame_windows[i][3]] + transformed_main_view * self.frame_masks[0]
				canvas[self.frame_windows[i][0]:self.frame_windows[i][1],self.frame_windows[i][2]:self.frame_windows[i][3]] = canvas[self.frame_windows[i][0]:self.frame_windows[i][1],self.frame_windows[i][2]:self.frame_windows[i][3]] + transformed_side_view[i] * self.frame_masks[i+1]

				blend_time[i] = time.time() - t

				pano1 = transformed_main_view + transformed_side_view[i] * (transformed_main_view == 0).astype('uint8')
				drawShape(self.seams[i],pano1)


        		### Perform Object detection and save timing info ###
				t = time.time()
				obj_detected,pts1,pts2,main_view_object_mask,side_view_object_mask = la.genObjMask(i,transformed_main_view, self.background_model_main[i], transformed_side_view[i], self.background_model_side[i], self.seams[i],[self.frame_windows[i][0],self.frame_windows[i][2]])
				detect_time[i] = time.time() - t


				
				if obj_detected: 
					print "Object Detected: ",obj_detected
					print "pts1: ",pts1[0,0]
					print "pts2: ",pts2[0,0]
					print "**********"
					cv2.circle(pano1,(pts1[0,1],pts1[0,0]),5,(255,0,0))	
					cv2.circle(pano1,(pts2[0,1],pts2[0,0]),5,(0,255,0))
					cv2.imshow("Pano "+str(i),pano1)
					cv2.waitKey(0)
					# Apply Alignment Phase
					t = time.time()
	                #side_view_main_mask = la.mapCoveredSide(self.homography_list[idx],main_view_frame,side_view_frame)
	                #main_seam, side_seam, side_border,transformed_side_border = la.genBorderMasks(main_view_frame, side_view_frame, mask1,new_mask,self.homography_list[idx],shift)
	                #tempH = la.lineAlign(pts1,main_view_frame,pts2,side_view_frame,self.fundamental_matrices_list[idx])
					#tempH = la.lineAlign(idx,pts1,255*main_view_object_mask,pts2,255*side_view_object_mask,self.fundamental_matrices_list[idx],main_seam, side_seam, side_border,transformed_side_border,shift,self.homography_list[idx])
					align_time[i] = time.time() - t

					# Apply Warping Phase
					t = time.time()
					#result2 = la.warpObject(idx,side_view_object_mask,tempH,result2,side_view_frame,side_view_background,self.homography_list[idx],shift)
					warp_time[i] = time.time() - t



			# Display Mosaic
			t = time.time()
			cv2.imshow("Main", canvas)
			# Check for termination
			if cv2.waitKey(1) == ord('q'):
				file.close()
				return
			display_time = time.time() - t

			# Read Next Frame
			t = time.time()
			ret, main_view_frame = self.main_view_caps.read()
			side_view_frames = [[]] * self.num_side_cams
			for i in range(self.num_side_cams):
				_, side_view_frames[i] = self.side_view_caps[i].read()
			read_time = time.time() - t

			frame_time = time.time() - t_frame_start


			print "******************************************************************************"
			print "Frame Reading took: " + str(1000*read_time) + " (ms)"
			print "Background warping took: " + str(1000*np.sum(background_time)) + " (ms)"
			print "Blending background took: " + str(1000*np.sum(blend_time)) + " (ms)"
			print "Object detection took: " + str(1000*np.sum(detect_time)) + " (ms)"
			print "Parallax alignment took: " + str(1000*np.sum(align_time)) + " (ms)"
			print "Final warping took: " + str(1000*np.sum(warp_time)) + " (ms)"
			print "Displaying image took: " + str(1000*display_time) + " (ms)"
			print "Frame Rate: "+ str(1/frame_time) + " fps"
			print "******************************************************************************"

			file.write("Frame Reading took: " + str(1000*read_time) + " (ms)" + "\n")
			file.write("Background warping took: " + str(1000*np.sum(background_time)) + " (ms) \n")
			file.write("Blending background took: " + str(1000*np.sum(blend_time)) + " (ms) \n")
			file.write("Object detection took: " + str(1000*np.sum(detect_time)) + " (ms) \n")
			file.write("Parallax alignment took: " + str(1000*np.sum(align_time)) + " (ms)  \n")
			file.write("Final warping took: " + str(1000*np.sum(warp_time)) + " (ms) \n")
			file.write("Displaying image took: " + str(1000*display_time) + " (ms) \n")
			file.write("Total Frame time: " + str(1000*frame_time) + " (ms) \n")
		
		file.close()		
		return


############################# Auxiliary Modular Tools #####################################################################################
def drawShape(shape,img,color = (0,255,0)):
	x_vals = []
	y_vals = []
	if shape.type == 'LineString':
			for point in list(shape.coords):
				x_vals.append(point[0])
				y_vals.append(point[1])
				if len(x_vals) > 1:
					start_point = (int(x_vals[-2]),int(y_vals[-2]))
					end_point = (int(x_vals[-1]),int(y_vals[-1]))
					#print start_point,end_point
					cv2.line(img, start_point,end_point,color)
	else:
		for section in shape:
			x_vals = []
			y_vals = []
			if section.type == 'LineString':
				for point in list(section.coords):
					x_vals.append(point[0])
					y_vals.append(point[1])
					if len(x_vals) > 1:
						start_point = (int(x_vals[-2]),int(y_vals[-2]))
						end_point = (int(x_vals[-1]),int(y_vals[-1]))
						cv2.line(img, start_point,end_point,(0,255,0))

			if section.type == 'MultiPoint':
				print "Error: Points, not Line"
			else:
				print section.type
				print "Error: Don't know what type this is"




############################# Begin Main function #########################################################################################

# Set video input locations
filepath = '../data/pi_writer/'
cap_main = cv2.VideoCapture(filepath+'output1.avi')
cap_side = [cv2.VideoCapture(filepath+'output2.avi'), cv2.VideoCapture(filepath+'output3.avi'), cv2.VideoCapture(filepath+'output4.avi')]


# Perform Initial Calibration
t = time.time()
a = LineStitcher(cap_main, cap_side)
cal_time = time.time() - t
print "Calibration took " + str(1000*cal_time) + " milliseconds"

#perform video streaming
a.stream()