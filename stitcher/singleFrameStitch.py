import numpy as np
import cv2
import stitcher2
import sys



# Set up file structure
INPUT_FILE_LOCATION = "../data/singleFrame/"
FILENAME_TEMPLATE = "D"
FILE_EXTENSION = ".png"

# Variables used to change stitcher settings. ie. Tunable Parameters
CALWINDOW = 1
DILATION_KERNEL = np.ones([3,3])
EROSION_LOOPS = 1
DEBUGGING=0
DILATION_LOOPS = 4
FOV = 45
OUTPUT_SIZE = [1080,1920,3]

#Camera Coefficients
radial_dst = np.array([-0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628])
#radial_dst = [-0.37801445,  0.38328306, -0.01922729, -0.01341008, -0.60047771]
mtx = np.array([[ 444.64628787,    0.,          309.40196271],[   0.,          501.63984347,  255.86111216],[   0.,            0.,            1.        ]])
#mtx = [[ 457.59059782,    0.,          324.59076057],[   0.,          521.77053812,  294.85975196],[   0.,            0.,            1.,        ]]


# Derived Parameters
#FOV bounds
pix2Ang = np.linspace(-45,45,480)
pix2Ang2 = np.linspace(-45,45,640)
top_edge = (np.abs(pix2Ang-FOV)).argmin()
bot_edge = (np.abs(pix2Ang+FOV)).argmin()
left_edge = (np.abs(pix2Ang2 + FOV)).argmin()
right_edge = (np.abs(pix2Ang2 - FOV)).argmin()
print [top_edge,bot_edge,left_edge,right_edge]


#Initializing Needed Processes and Variables
stitch = stitcher2.Stitcher()
H1 = np.zeros([3,3])
H2 = np.zeros([3,3])
H3 = np.zeros([3,3])
output = np.zeros(OUTPUT_SIZE)
output_center = np.array([OUTPUT_SIZE[0]/2,OUTPUT_SIZE[1]/2]).astype('int')

fname1 = INPUT_FILE_LOCATION+FILENAME_TEMPLATE+"1"+FILE_EXTENSION
fname2 = INPUT_FILE_LOCATION+FILENAME_TEMPLATE+"2"+FILE_EXTENSION
fname3 = INPUT_FILE_LOCATION+FILENAME_TEMPLATE+"3"+FILE_EXTENSION
fname4 = INPUT_FILE_LOCATION+FILENAME_TEMPLATE+"4"+FILE_EXTENSION

#print fname1

img1 = cv2.imread(fname1)
img2 = cv2.imread(fname2)
img3 = cv2.imread(fname3)
img4 = cv2.imread(fname4)


#result1 = np.concatenate((img1,img2),axis=1)
#result2 =np.concatenate((img3,img4),axis=1)
#result = np.concatenate((result1,result2),axis=0)

#cv2.imshow('images',result)
#cv2.waitKey(0)

#Stitch images
(result1, vis1,H,mask11,mask12,coord_shift) = stitch.stitch([img4, img1], showMatches=True)
output_location = [output_center[0] - coord_shift[0],output_center[1] - coord_shift[1]]
if result1 is 0:
    print "Error: No match in stitch 1"
    sys.exit(0)
    
tmp_out = output[output_location[0]:output_location[0]+result1.shape[0], output_location[1]:output_location[1]+result1.shape[1],:]
output[output_location[0]:output_location[0]+result1.shape[0], output_location[1]:output_location[1]+result1.shape[1],:] = result1*mask12 + tmp_out*np.logical_not(mask12)

(result1, vis1,H,mask11,mask12,coord_shift) = stitch.stitch([img4, img2], showMatches=True)
output_location = [output_center[0] - coord_shift[0],output_center[1] - coord_shift[1]]
if result1 is 0:
    print "Error: No match in stitch 2"
    sys.exit(0)
    
tmp_out = output[output_location[0]:output_location[0]+result1.shape[0], output_location[1]:output_location[1]+result1.shape[1],:]
output[output_location[0]:output_location[0]+result1.shape[0], output_location[1]:output_location[1]+result1.shape[1],:] = result1*mask12 +tmp_out*np.logical_not(mask12)

(result1, vis1,H,mask11,mask12,coord_shift) = stitch.stitch([img4, img3], showMatches=True)
output_location = [output_center[0] - coord_shift[0],output_center[1] - coord_shift[1]]
if result1 is 0:
    print "Error: No match in stitch 3"
    sys.exit(0)
    
tmp_out = output[output_location[0]:output_location[0]+result1.shape[0], output_location[1]:output_location[1]+result1.shape[1],:]
output[output_location[0]:output_location[0]+result1.shape[0], output_location[1]:output_location[1]+result1.shape[1],:] = result1*mask12 + tmp_out*np.logical_not(mask12) 

cv2.imshow('Output',output.astype('uint8'))
cv2.waitKey(0)