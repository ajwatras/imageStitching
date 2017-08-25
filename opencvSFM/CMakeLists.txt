cmake_minimum_required(VERSION 2.8)
project( opencvSFM )
find_package( OpenCV REQUIRED )
find_package(Eigen3 REQUIRED) 
include_directories(${EIGEN3_INCLUDE_DIR})
add_executable( opencvSFM main.cpp )
message([STATUS] ${OpenCV_LIBS})
target_link_libraries( opencvSFM ${OpenCV_LIBS} )

