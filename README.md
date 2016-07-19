# imageStitching

This repository holds code to stitch together four video feeds. 


DATA/ - contains all test data sets. This includes examples for stitching, camera calibration, as well as challenge data sets that will be used for testing the quality of stitching algorithms.

STITCHER/ - This contains all code used to stitch together images. 
4stitch - Basic stitching of four images. Obsolete with development of videoStitch.
9Stitch - script to stitch together 9 images. 
depthMap - Under development. Method for getting a sparse depth map from a pair of images. 
stitch - Basic stitching of two images.
stitcher - Image stitching libraries. More documentation soon.
stitcher2 - Testing platform for new features to be added to stitcher. Once features are complete, they will be moved over to stitcher.
videoStitch - Framework for real time four image stitching. 
videoStitch2 - Testing platform for four image stitching using stitcher2.
videoStitch3 - Development platform for grab cut image re-stitching. 


TOOLS/ - Contains tools useful for gathering or processing data, as well as interfacing with cameras but not directly related to image stitching.
4to1 - Takes four video frames and combines them into a single output video feed. Inputs are given when each of the cap variables is initialized. the output will be written to ./output.avi
4vidStream - Plays the video from each of four inputs. Feeds are concatenated together so that they can all be seen simultaneously. The input video feeds are taken from the location specified in the vidcap variable initialization. 
chessboard_calibration - Calculates the radial distortion parameters and 
fgSeparation - Sample script that generates bounding boxes around foreground objects in a video stream. Will be removed when grab cut restitching is implemented in the main stitcher.
test - A platform for developing new scripts or testing ideas for existing scripts. 
vid2frame - Converts .avi files to sequence of .jpg files. Input video is set by the vid1 variable, the output location is set by saveDest, and FrameLimit is used to set the output filenames. It should be higher than the number of frames in the video. 
vidwrite - Captures video from a 4 camera array and saves it as .avi files. The video files are stored as output1-output4.avi in the path specified by VIDEOWRITER_OUTPUT_PATH.

