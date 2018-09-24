import stitcher as sti
import line_align as la
import cv2
import numpy as np

def lineMatch(image1,image2, H, edgeThresh = 20 ):

	# Generate Lines and points of intersection for image 1
	edges1 = cv2.Canny(image1,edgeThresh,edgeThresh*3)
	lines1 = la.lineSelect(edges1,edgeThresh,200)

	A_pts = np.nonzero(edges1[0,:])
	B_pts = np.nonzero(edges1[:,0])
	C_pts = np.nonzero(edges1[edges1.shape[0]-1,:])
	D_pts = np.nonzero(edges1[:,edges1.shape[1]-1])

	print "A: ",A_pts
	print "B: ",B_pts
	print "C: ",C_pts
	print "D: ",D_pts

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
	print "temp_line: ",temp_line[0,0],temp_line[0,1]

	# A points distance
	if any(map(len,A_pts)):
		y = 0
		x = -np.sin(temp_line[0,1])/np.cos(temp_line[0,1])*y + temp_line[0,0]/np.cos(temp_line[0,1])
		A_dist = np.abs(A_pts - x)
		print "A Distance: ",A_dist,x,y
		for i in range(0,len(A_pts[0])):
			print A_pts[0][i],A_dist[0][i]
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

	print pts
	print distances
	idx = np.argsort(np.mat(distances))
	print idx
	print pts[idx[0,0]], pts[idx[0,1]]


	pts2 = np.array(pts)[idx]
	distances2 = np.array(distances)[idx]

	print pts2
	print distances2



main_frame = 100*np.ones((480,640,3)).astype('uint8')
side_frame = 100*np.ones((480,640,3)).astype('uint8')
line1 = (-250,-np.pi/2)
line2 = (-250,-np.pi/2)
la.drawLines(line1[0],line1[1], main_frame, (255,255,255),30)
la.drawLines(line2[0],line2[1], side_frame, (255,255,255),30)

corners = np.mat([[0,0],[0,side_frame.shape[1]],[side_frame.shape[0],0],[side_frame.shape[0],side_frame.shape[1]]])
new_corners = np.mat([[100,100],[100,side_frame.shape[1]+100],[side_frame.shape[0]+100,100],[100+side_frame.shape[0],100+side_frame.shape[1]]])
H,h_mask = cv2.findHomography(corners,new_corners)

lineMatch(main_frame,side_frame,H)