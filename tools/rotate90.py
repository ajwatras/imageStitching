## This script will capture video from four mobius cameras and store it to a desired location. 
import numpy as np
import cv2
import sys

# Location that files will be written
VIDEOWRITER_OUTPUT_PATH = './'

# This captures video from cameras 1-4. If your device does not have a webcam,
# this should probably be switched to cameras 0-3. If your machine is claiming 
# no cameras are connected, look here for the problem.
filename = sys.argv[1]
angle = int(sys.argv[2])
cap1 = cv2.VideoCapture(filename)
ret1, frame1 = cap1.read()
n = frame1.shape
print n

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
if (angle == 1):
	out1 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'rotated.avi',fourcc, 20.0, (n[1],n[0]),True)
else:
	out1 = cv2.VideoWriter(VIDEOWRITER_OUTPUT_PATH+'rotated.avi',fourcc, 20.0, (n[0],n[1]),True)


# We capture incoming video frames until the 'q' button is pressed or there are
# no more incoming video frames. 
while(out1.isOpened()):
	ret1, frame1 = cap1.read()
	if ret1==True:
	# uncomment this section if you need to flip the left and right of the incoming video.
		if (angle == 0):
			# Output 2
			frame1 = cv2.transpose(frame1)
			frame1 = cv2.flip(frame1,1)
		elif (angle == 1):
			# Output 3
			frame1 = cv2.flip(frame1,0)
			frame1 = cv2. flip(frame1,1)
		elif (angle == 2):
			# Output 4
			frame1 = cv2.transpose(frame1)
			frame1 = cv2.flip(frame1,0)


        # write the flipped frame
		out1.write(frame1)
		print frame1.shape, n[1],n[0]
		cv2.imshow('frame',frame1)
		if cv2.waitKey(1) & 0xFF == ord('q'): # terminate loop if 'q' is pressed.
			break
	else:
		break

# Release everything if job is finished
# Release cameras
cap1.release()

# Save and release video feeds
out1.release()

cv2.destroyAllWindows()
