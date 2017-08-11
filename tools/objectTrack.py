import numpy as np
import cv2
import line align

def modelBackground(caps, CAL_LENGTH = 10):
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
			break

	for k in range(0,n):
		output[k] = frames[k]/CAL_LENGTH
		for m in range(1,CAL_LENGTH):
			ret, frames[k] = caps[k].read()
			output[k] = output[k] + frames[k]/CAL_LENGTH


		output[k] = output[k].astype('uint8')
	return True, output

def identifyFG(frame, model,thresh = 40):
	kernel = np.ones((30,30),np.uint8)
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	model = cv2.cvtColor(model, cv2.COLOR_BGR2GRAY)

	diff_frame = np.abs(frame.astype('float') - model.astype('float'))
	binary = cv2.threshold(diff_frame,thresh,255,cv2.THRESH_BINARY)

	output = cv2.morphologyEx(binary[1],cv2.MORPH_CLOSE,kernel)
	#output = cv2.morphologyEx(binary[1],cv2.MORPH_OPEN,kernel)

	return output

cap1 = cv2.VideoCapture('../data/7_18_17/bean1/output1.avi')
cap2 = cv2.VideoCapture('../data/7_18_17/bean1/output2.avi')
cap3 = cv2.VideoCapture('../data/7_18_17/bean1/output3.avi')
cap4 = cv2.VideoCapture('../data/7_18_17/bean1/output4.avi')

caps = (cap1,cap2,cap3,cap4)

ret, model = modelBackground(caps)

while ret:
	ret,frame = caps[0].read()
	binary = identifyFG(frame, model[0])
	cv2.imshow("foreground",binary)
	cv2.imshow('frame',frame)
	if cv2.waitKey(10) == ord('q'):
		break
