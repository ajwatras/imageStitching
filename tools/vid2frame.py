import numpy as np
import math
import cv2

vid1 = "../data/Calibration_vid4.avi"
saveDest = "../data/vid2frame/"
FrameLimit = 100000

cap1 = cv2.VideoCapture(vid1)
cc = 0



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



cv2.destroyAllWindows()
cap1.release()
