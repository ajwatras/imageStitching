import numpy as np
import line_align4 as la
a = np.zeros([50,50])
b = np.zeros([50,50])
a[10:31,:] = 255
#a[31,:] = 255

#for i in range(49):
#    a[i,i+1] = 255
#    if i < 45:
#        a[i,i+5] = 255
line1,line2 = la.trackObject2(a,b,[0,0],[10,0])
print line1
print line2

c = line1[0]/np.sin(line1[1]) - np.cos(line1[1])/np.sin(line1[1])*10
d = line2[0]/np.sin(line2[1]) - np.cos(line2[1])/np.sin(line2[1])*10
print c
print d
