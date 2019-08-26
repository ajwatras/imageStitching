EasyVis Software Tutorial

Dependencies:
Python 2.7
OpenCv 4.0.0-pre
OpenCV Contrib nonfree repositories (for SURF)
urllib
numpy
ulti
matplotlib

Installation Guide:
Use your package manager to install python 2.7
Install OpenCV with the opencv_contrib nonfree modules
*Fill in the rest of this after performing clean install. 

Running Instructions: 
1. Ensure that pistream_faststitch3.py has been updated with your camera locations. Each camera should be given a cv2.VideoCapture("camera location") object, which will be used
   to read the video frames. "camera location" should be set based on the type of input desired. For webcams, "camera location" should be an int denoting the index of the camera. 
   for video files, "camera location" should be a string giving the filepath to the video file, and for streaming video, it should be the web address of the video stream.  
2. Place calibration pattern in front of video cameras. Calibration pattern should be any feature dense pattern which can be seen from all cameras. An example pattern can be found in 
   tools/calibration_pattern.png
3. Run the command "python pistream_faststitch3.py" to perform image stitching. 
4. Remove Calibration Pattern from scene, and set scene up as desired.
5. Press "p" to begin background modeling
6. Press "q" to close program once you are done. 

Control Options:
No features currently require user input. 



