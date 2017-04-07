import urllib, cStringIO
import cv2

file = cString.StringIO(urllib.urlopen("http://10.42.0.105:8060/javascript_simple.html").read())
img = Image.open(file);

while 1:
	cv2.imshow("output",img)
	cv2.waitKey