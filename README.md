## Usage: 
In order to perform image stitching, run command: 

`python pistream_faststitch.py`

During streaming, the stitcher follows the following keyboard inputs:

* `q` - quit the current script. 
* `r` - re-calibrate the image stitcher
* `0,1,2,3,4` - Change top image to camera of corresponding number
* `,` - Rotate the panorama clockwise
* `.` - Rotate the panorama counter clockwise



## Setup and Miscellaneous Notes:
Code is NOT compatible with Python 3.

* Requires [Numpy(Should be compatible with most recent version)](https://www.scipy.org/scipylib/download.html)
* Requires [OpenCV(Version 3.x)](https://opencv.org/releases.html)
* Requires [OpenCV contrib-folder](https://github.com/opencv/opencv_contrib)

It may be possible to use OpenCV Version 2.x without the OpenCV contrib_folder (this is NOT recommended as it is untested)

Currently the program does NOT seem compatible with Windows.


## File Overview:
#### (Main programs)
The **Stitcher** folder contains all code used for running the image stitching pipeline.  
`pistream_faststitch.py` is used stitch a stream of multiple videos together into a panoramic video.  
`lazy_stitcher.py` is used to stitch a multiple images together into a panoramic.  
`stitcher2.py` is used to stitch two images together.

The files listed above are dependent on each other from top to bottom.

#### (Data)
**Sample** folder within the stitcher folder contains 4 videos: m.avi, the main or central view, and four side views that can be used to test the code.  
The **Data** folder contains other miscellaneous videos and images to test different aspects of the program. Including still frames to test only image stitching and calibration videos to test only calibration.  
The **Data** folder should also include several output videos and images to show what the project is currently capable of.  

#### (Non-Main Programs)
`stab_pistream_faststitch.py` is meant to stabalize and stitch video at the same time; however, it is currently not in a working state and will not compile.  
The **tools** folder contains test programs that were used in the creation of the project. It can still be used to test if a computer can open the and stream from a camera or file.  



## Useful OpenCV Readings:
**Homography:** https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html  
**Brute Force Feature Matching:** https://docs.opencv.org/3.4.4/dc/dc3/tutorial_py_matcher.html  
**Image Erosion and Dilation:** https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html  
**Rotation and Translation Matrices:** https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html  
