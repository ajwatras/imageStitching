import numpy as np
import cv2

cap1 = cv2.VideoCapture('../data/vidwriter/output1.avi')
cap2 = cv2.VideoCapture('../data/vidwriter/output2.avi')
cap3 = cv2.VideoCapture('../data/vidwriter/output3.avi')
cap4 = cv2.VideoCapture('../data/vidwriter/output4.avi')


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('./output.avi',fourcc, 20.0, (1280,960))

while(cap1.isOpened()):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()

    result1 = np.concatenate((frame1,frame2),axis=1)
    result2 =np.concatenate((frame3,frame4),axis=1)
    result = np.concatenate((result1,result2),axis=0)
    if ret1==True:
        out.write(result)

        cv2.imshow('frame',result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap1.release()
cap2.release()
cap3.release()
cap4.release()

out.release()

cv2.destroyAllWindows()
