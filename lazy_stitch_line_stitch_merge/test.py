from sympy import *
from sympy.geometry import *
import time
import numpy as np
import random

def findOuterPts(point,vec,window):
    # Intersects the line defined by the equation [x,y] = a*vec + point with the outer edge of the image
    # check x = 0

    x1 = []
    y1 = []
    x2 = []
    y2 = []

    vec = vec/np.linalg.norm(vec)

    if vec[0] is not 0:
        a1 = -point[0]/vec[0]
        if a1 >= 0:
            y1 = point[1] + a1 * vec[1]
            if y1 >= 0 and y1 <= window[0]:
                return [0,y1]

    # check y = 0 
    if vec[1] is not 0:
        a2 = -point[1]/vec[1]
        if a2 >= 0:
            x1 = point[0] + a2 * vec[0]
            if x1 >= 0 and x1 <= window[1]:
                inter_found = True
                print [x1,0]
    # check x = x_max
    if vec[0] is not 0:
        a3 = (window[1]-point[0])/vec[0]
        if a3 >= 0:
            y2 = point[1] + a3 * vec[1]
            if y2 >= 0 and y2 <= window[0]:
                return [window[1],y2]

    # check y = y_max
    if vec[1] is not 0:
        a4 = (window[0]-point[1])/vec[1]
        if a4 >= 0:
            x2 = point[0] + a4 * vec[0]
            if x2 >= 0 and x2 <= window[1]:
                return [x2,window[0]]

    print "Error: No Intersection Found in findOuterPts()"
    print a1,a2,a3,a4
    print y1,x1,y2,x2
    return [-1,-1]


x_max = 1080
y_max = 640

point = [random.uniform(0,x_max),random.uniform(0,y_max)]
vec = [random.uniform(-x_max,x_max),random.uniform(-y_max,y_max)]



image = np.zeros([y_max,x_max])


list_time = [[]] * 100
for i in range(100):
	t = time.time()
	pt = findOuterPts(point,vec,image.shape)
	list_time[i] = time.time() - t


print "Point: ",point
print "Vec: ", vec
print "Window ", image.shape
print "Output: ", pt
print "Time to Complete: ", sum(list_time)/100, " s"