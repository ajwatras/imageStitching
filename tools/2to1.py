## This script will take four output video feeds and combine them into a single larger video feed. 
import numpy as np
import cv2

# Begin reading from the desired videofeeds
#cap1 = cv2.VideoCapture('../data/vidwriter/output1.avi')
#cap2 = cv2.VideoCapture('../data/vidwriter/output2.avi')
#cap3 = cv2.VideoCapture('../data/vidwriter/output3.avi')
#cap4 = cv2.VideoCapture('../data/vidwriter/output4.avi')

cap1 = cv2.VideoCapture('./output.avi')
cap2 = cv2.VideoCapture('../Demo/beanDrop.avi')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output2.avi',fourcc, 20.0, (3200,1080))

ret1, frame1 = cap1.read()
ret2, frame2 = cap2.read()



#For each frame in the videofeeds, we combine the four feeds into a single one. 
while(ret1):
    frame1 = np.pad(frame1,((0,120),(0,0),(0,0)), 'constant', constant_values=0)
    print frame1.shape
    print frame2.shape
    result = np.concatenate((frame1,frame2),axis=1)
    if ret1==True:
        out.write(result)

        cv2.imshow('frame',result)
        print "result: "
        print result.shape
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    
# Release everything if job is finished
cap1.release()
cap2.release()

out.release()

cv2.destroyAllWindows()
