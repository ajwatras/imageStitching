import numpy as np
import cv2
import calibrateCameras as ccam

#cap3 = cv2.VideoCapture('http://10.42.0.101:8010/?action=stream')
#cap1 = cv2.VideoCapture('http://10.42.0.102:8020/?action=stream')
#cap2 = cv2.VideoCapture('http://10.42.0.103:8030/?action=stream')
#cap4 = cv2.VideoCapture('http://10.42.0.104:8040/?action=stream')

cap3 = cv2.VideoCapture('../lazy_stitch_line_stitch_merge/sample/output3.avi')
cap1 = cv2.VideoCapture('../lazy_stitch_line_stitch_merge/sample/output1.avi')
cap2 = cv2.VideoCapture('../lazy_stitch_line_stitch_merge/sample/output2.avi')
cap4 = cv2.VideoCapture('../lazy_stitch_line_stitch_merge/sample/output4.avi')
#cap5 = cv2.VideoCapture('http://10.42.0.105:8050/?action=stream')


ret1, image1 = cap1.read()
ret2, image2 = cap2.read()
ret3, image3 = cap3.read()
ret4, image4 = cap4.read()
#ret5, image5 = cap5.read()


F1,pts11,pts12 = ccam.calcF(image1,image2)
F2,pts21,pts22 = ccam.calcF(image1,image3)
F3,pts31,pts32 = ccam.calcF(image1,image4)
#F4,pts41,pts42 = calcF(image1,image5)

print F1,F2,F3
np.savetxt('fundamentalMatrices/F1.txt',F1,delimiter=',')
np.savetxt('fundamentalMatrices/F2.txt',F2,delimiter=',')
np.savetxt('fundamentalMatrices/F3.txt',F3,delimiter=',')

image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
image3 = cv2.cvtColor(image3,cv2.COLOR_BGR2GRAY)
image4 = cv2.cvtColor(image4,cv2.COLOR_BGR2GRAY)

displayEpipolar(image1,image2,F1,pts11,pts12)
cv2.waitKey(0)
displayEpipolar(image1,image3,F2,pts21,pts22)
cv2.waitKey(0)
displayEpipolar(image1,image4,F3,pts31,pts32)
cv2.waitKey(0)