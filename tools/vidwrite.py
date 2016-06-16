import numpy as np
import cv2

cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(2)
cap3 = cv2.VideoCapture(3)
cap4 = cv2.VideoCapture(4)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('../data/vidwriter/output1.avi',fourcc, 20.0, (640,480))
out2 = cv2.VideoWriter('../data/vidwriter/output2.avi',fourcc, 20.0, (640,480))
out3 = cv2.VideoWriter('../data/vidwriter/output3.avi',fourcc, 20.0, (640,480))
out4 = cv2.VideoWriter('../data/vidwriter/output4.avi',fourcc, 20.0, (640,480))

while(cap1.isOpened()):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()
    ret4, frame4 = cap4.read()
    if ret1==True:
        frame1 = cv2.flip(frame1,1)
        frame2 = cv2.flip(frame2,1)
        frame3 = cv2.flip(frame3,1)
        frame4 = cv2.flip(frame4,1)

        # write the flipped frame
        out1.write(frame1)
        out2.write(frame2)
        out3.write(frame3)
        out4.write(frame4)

        cv2.imshow('frame',frame1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap1.release()
cap2.release()
cap3.release()
cap4.release()

out1.release()
out2.release()
out3.release()
out4.release()
cv2.destroyAllWindows()
