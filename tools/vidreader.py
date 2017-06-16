import cv2
import numpy as np


vidcap = cv2.VideoCapture('http://10.42.0.105:8060/?action=stream')
#vidcap = cv2.VideoCapture(0)


success, frame = vidcap.read()

while success:
    cv2.imshow("vidreader",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
       	break
    
    success,frame = vidcap.read()

cv2.destroyAllWindows()
vidcap.release()
