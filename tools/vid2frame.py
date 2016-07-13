## takes input video feed and separates out each frame into a jpg file.

import numpy as np
import math
import cv2

# input parameters
vid1 = "./output.avi"			#Video to be split
saveDest = "../data/vid2frame/"		#Location files should be saved 2
FrameLimit = 100000			#Limit on length of the video (for filenaming)

# Set up video capture environment
cap1 = cv2.VideoCapture(vid1)
cc = 0


# Read frames and then save them until q is pressed or video is over
ret1, frame1 = cap1.read()
while(ret1 == 1):
    
    cv2.imshow("Frame",frame1)
    
    filePath = saveDest+"frame"+str(cc).zfill(int(math.ceil(np.log10(FrameLimit))))+".jpg"
    print filePath

    cv2.imwrite(filePath, frame1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret1, frame1 = cap1.read()
    cc = cc + 1



#clean up environmentchc 
cv2.destroyAllWindows()
cap1.release()
