## Import libraries for test.
import cv2
import numpy as np
from matplotlib import pyplot as plt


## Constants often used in testing
radial_dst = np.array([-0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628]) # Estimated Mobius camera Distortion.
#radial_dst = [-0.37801445,  0.38328306, -0.01922729, -0.01341008, -0.60047771]
mtx = np.array([[ 444.64628787,    0.,          309.40196271],[   0.,          501.63984347,  255.86111216],[   0.,            0.,            1.        ]]) # Estimated Mobius Camera matrix

## Introduce any GLOBAL constants
DISCONNECT_SIZE = 15
SLICE_LOCATION = 500
ix,iy = -1,-1


frame = cv2.imread("./frame01164.jpg")

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

img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
mask = np.zeros(sobelx.shape)
#mask[:,SLICE_LOCATION-1:SLICE_LOCATION+1] = np.ones([mask.shape[0],2])
mask[:,SLICE_LOCATION] = np.ones(mask.shape[0])
x_filt = mask*sobelx
x_filt= (np.abs(x_filt) > 250).astype('uint8')
#x_filt = sobelx*x_filt

x_loc,y_loc = np.nonzero(x_filt)
for i in range(0,len(x_loc)):
    frame[x_loc[i],y_loc[i],:] = [255,0,0]
    
    


sobely = cv2.Sobel(x_filt,cv2.CV_64F,0,1,ksize=5)
#sobely = np.abs(sobely) > 1
#print np.nonzero((np.abs(sobely) > 200))

#plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,1),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(x_filt*sobelx,cmap = 'gray')
plt.title('Filtered X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(frame,cmap = 'gray')
plt.title('Tagged Frame'), plt.xticks([]), plt.yticks([])


plt.show()

