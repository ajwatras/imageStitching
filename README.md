## imageStitcher

This software was designed to create a mosaicked visualization of live camera feeds. The results of our EasyVis camera system used on our prototype camera array can be seen [here](./sample.mp4) This software has been tested on Ubuntu 16.04. If you have any issues with the software or installation procedures, please reach  out to watras@wisc.edu

## Installation

#### Ubuntu
1. First install python 2.7 by running `sudo apt install python python-pip`
2. Install the necessary python packages by using pip. Run `sudo pip install numpy imutils matplotlib `
3. Install opencv with the opencv_contrib repository by following the instructions found [here](https://elbauldelprogramador.com/en/how-to-compile-opencv3-nonfree-part-from-source/)

#### Windows

TBD

#### OS X

TBD

## Running the program

1. In _**&lt;pathToDirectory>**/imageStitching/pistream_\__faststitch.py_ set your input video feeds by modifying the `caps` variable. Check the [openCV Documentation](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html) for details on how to use the VideoCapture object. **Note:** In older versions of the software, this value was set by the `addr1` through `addr5` variables. 

2. Run `python pistream_faststitch.py`
3. While the video is streaming, use the following commands to control the software:
    * `q` will terminate the program.
    * `r` will re-calibrate the stitching software
    * `0,1,2,3,4` will set the chosen camera to be placed on top of the others during blending
    * `,` will rotate the panorama counter clockwise
    * `.` will rotate the panorama clockwise