## Import libraries for test.
import numpy as np
import cv2
import time

## Constants often used in testing
radial_dst = np.array([-0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628]) # Estimated Mobius camera Distortion.
#radial_dst = [-0.37801445,  0.38328306, -0.01922729, -0.01341008, -0.60047771]
mtx = np.array([[ 444.64628787,    0.,          309.40196271],[   0.,          501.63984347,  255.86111216],[   0.,            0.,            1.        ]]) # Estimated Mobius Camera matrix

## Introduce any GLOBAL constants
DISCONNECT_SIZE = 15
SLICE_LOCATION = 500
ix,iy = -1,-1


frame = cv2.imread("./testFrame.jpg")

## Begin test

#modify frame 
frame2 = frame.copy()
new_slice_size = frame[:,SLICE_LOCATION:,:].shape
frame2 = np.pad(frame2,pad_width = ((DISCONNECT_SIZE,0),(0,0),(0,0)),mode='constant',constant_values = 0)
frame[:,SLICE_LOCATION:,:] = frame2[0:new_slice_size[0],SLICE_LOCATION:,:]

cv2.imwrite('testbackground.jpg',frame)

cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# (500,89)-->(500,72),(500,107)-->(500,90),(537,86)-->(537,86),(537,105)-->same
xshift = 500
yshift = 72
src_pts = np.float32([[500-xshift,87-yshift],[500-xshift,105-yshift],[625-xshift,84-yshift],[625-xshift,99-yshift]])
dst_pts = np.float32([[500-xshift,72-yshift],[500-xshift,90-yshift],[625-xshift,84-yshift],[625-xshift,99-yshift]])
M = cv2.getPerspectiveTransform(src_pts,dst_pts)

segment = frame[72:105,500:625,:]


print frame.shape
print segment.shape
warped = cv2.warpPerspective(segment,M,(frame.shape[1] - xshift,frame.shape[0]- yshift))
mask = (warped > 0).astype('uint8')


cv2.imshow("warped",warped)
frame[72:105,500:625,:] = np.zeros([segment.shape[0],segment.shape[1],segment.shape[2]])
tmp_image = frame[yshift:,xshift:,:]
frame[yshift:,xshift:,:] = mask*warped + np.logical_not(mask)*tmp_image

cv2.imwrite('testOutput1.jpg',frame)

cv2.rectangle(frame,(500,72),(625,105),(255,0,0))

cv2.imwrite('testOutput2.jpg',frame)


def draw_circle(event,x,y,flags,param):
    global ix,iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(frame,(x,y),2,(255,0,0),-1)
        ix,iy = x,y

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',frame)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print ix,iy
    elif k == ord('q'):
        break
    
cv2.destroyAllWindows()


