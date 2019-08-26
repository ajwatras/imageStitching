import cv2
import numpy as np


#vidcap = cv2.VideoCapture("/dev/stdin")
vidcap = cv2.VideoCapture(1)


success, frame = vidcap.read()

while success:
    cv2.imshow("vidreader",frame)
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
       	break
    
    success,frame = vidcap.read()

cv2.destroyAllWindows()
vidcap.release()