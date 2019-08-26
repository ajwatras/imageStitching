import cv2
import numpy as np
import line_align4 as la
import argparse

# Set input video feed
file = "test.avi"
cap = cv2.VideoCapture(file)


# Set up Parameters
image = cap.read()
refPt = []


# Identify Main View Coordinates
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("image", image)
	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("N"):
		image = clone.copy()
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("Y"):
		break
 
 
# close all open windows
cv2.destroyAllWindows()


# Separate foreground and Background

# Generate Background Models.

# Apply Foreground Correction

# Final Blending


def clickAndDrag(event, x,y,flags, param):
	global refPt

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
 
	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
 
		# draw a rectangle around the region of interest
		cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("image", image)
