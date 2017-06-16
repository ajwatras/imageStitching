## This script will take four output video feeds and combine them into a single larger video feed. 
import numpy as np
import cv2

# Begin reading from the desired videofeeds
#cap1 = cv2.VideoCapture('../data/vidwriter/output1.avi')
#cap2 = cv2.VideoCapture('../data/vidwriter/output2.avi')
#cap3 = cv2.VideoCapture('../data/vidwriter/output3.avi')
#cap4 = cv2.VideoCapture('../data/vidwriter/output4.avi')

cap1 = cv2.VideoCapture('../data/pi_writer/output1.avi')
cap2 = cv2.VideoCapture('../data/pi_writer/output2.avi')
cap3 = cv2.VideoCapture('../data/pi_writer/output3.avi')
cap4 = cv2.VideoCapture('../data/pi_writer/output4.avi')


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output.avi',fourcc, 20.0, (1280, 960))

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()
ret3, frame3 = cap3.read()
ret4, frame4 = cap4.read()

print frame1.shape
#For each frame in the videofeeds, we combine the four feeds into a single one. 
while(ret1):

    result1 = np.concatenate((frame1,frame2),axis=1)
    result2 =np.concatenate((frame3,frame4),axis=1)
    result = np.concatenate((result1,result2),axis=0)
    if ret1==True:
        out.write(result)

        cv2.imshow('frame',result)
        print result.shape
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    
    
# Release everything if job is finished
cap1.release()
cap2.release()
cap3.release()
cap4.release()

out.release()

cv2.destroyAllWindows()
