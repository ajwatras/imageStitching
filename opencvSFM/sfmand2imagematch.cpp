#define CERES_FOUND true

#include <opencv2/sfm.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

using namespace std;
using namespace cv;
using namespace cv::sfm;
using namespace xfeatures2d;

static void help() {
  cout
      << "\n------------------------------------------------------------------------------------\n"
      << " This program shows the multiview reconstruction capabilities in the \n"
      << " OpenCV Structure From Motion (SFM) module.\n"
      << " It reconstruct a scene from a set of 2D images \n"
      << " Usage:\n"
      << "        example_sfm_scene_reconstruction <path_to_file> <f> <cx> <cy>\n"
      << " where: path_to_file is the file absolute path into your system which contains\n"
      << "        the list of images to use for reconstruction. \n"
      << "        f  is the focal lenght in pixels. \n"
      << "        cx is the image principal point x coordinates in pixels. \n"
      << "        cy is the image principal point y coordinates in pixels. \n"
      << "------------------------------------------------------------------------------------\n\n"
      << endl;
}

int main(int argc, char* argv[])
{
  	vector<string> images_paths;
	images_paths.push_back("/home/jianwei/Documents/opencvSFM/images/p1.png");
  	images_paths.push_back("/home/jianwei/Documents/opencvSFM/images/p2.png");
  	images_paths.push_back("/home/jianwei/Documents/opencvSFM/images/p3.png");
  	images_paths.push_back("/home/jianwei/Documents/opencvSFM/images/p4.png");
  	cout<<images_paths.at(0)<<endl;
	cout<<images_paths.at(1)<<endl;
	cout<<images_paths.at(2)<<endl;
	cout<<images_paths.at(3)<<endl;
  	// Build instrinsics
  	float f  = 800;
  	float cx = 400;
  	float cy = 225;
  	Matx33d K = Matx33d( f, 0, cx,
                       0, f, cy,
                       0, 0,  1);
  	bool is_projective = true;
  	vector<Mat> Rs_est, ts_est, points3d_estimated;
	
	// surf feature points
	Mat img_1 = imread( "/home/jianwei/Documents/opencvSFM/images/p1.png", IMREAD_GRAYSCALE );
	Mat img_2 = imread( "/home/jianwei/Documents/opencvSFM/images/p2.png", IMREAD_GRAYSCALE );	
	if( !img_1.data || !img_2.data )
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }
	//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
	int minHessian = 400;
	Ptr<SURF> detector = SURF::create();
	detector->setHessianThreshold(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	Mat descriptors_1, descriptors_2;
	detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
	detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );
	//-- Step 2: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
  	std::vector< DMatch > matches;
  	matcher.match( descriptors_1, descriptors_2, matches );
	
	double max_dist = 0; double min_dist = 100;
	
  	//-- Quick calculation of max and min distances between keypoints
  	for( int i = 0; i < descriptors_1.rows; i++ )
  	{ double dist = matches[i].distance;
    	if( dist < min_dist ) min_dist = dist;
    	if( dist > max_dist ) max_dist = dist;
  	}
	printf("-- descriptors_1 : %d \n", descriptors_1.rows );
	printf("-- descriptors_2 : %d \n", descriptors_2.rows );
	printf("-- matches : %d \n", matches.size() );
  	printf("-- Max dist : %f \n", max_dist );
  	printf("-- Min dist : %f \n", min_dist );

  	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  	//-- small)
  	//-- PS.- radiusMatch can also be used here.
  	std::vector< DMatch > good_matches;
  	for( int i = 0; i < descriptors_1.rows; i++ )
  	{ if( matches[i].distance <= 3*min_dist )
    	{ good_matches.push_back( matches[i]); }
  	}
	printf("-- good_matches : %d \n", good_matches.size() );
  	//-- Draw only "good" matches
  	Mat img_matches;
  	drawMatches( img_1, keypoints_1, img_2, keypoints_2,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );	
	
  	//-- Show detected matches
  	imshow( "Good Matches", img_matches );
	moveWindow("Good Matches", 50,50);
  	for( int i = 0; i < (int)good_matches.size(); i++ )
  	{ printf( "-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx ); }	
	
	waitKey(0);
	
	// Performing SFM and record the time it takes
	clock_t start;
	double duration;
	start = clock();
	
  	reconstruct(images_paths, Rs_est, ts_est, K, points3d_estimated, is_projective);
	
	duration = ( clock() - start ) / (double) CLOCKS_PER_SEC;
    cout<<"reconstruction time "<< duration <<'\n';
	
  	// Print output
	cout << "Rs_est"<<Rs_est.at(0)<<endl;
	cout << "ts_est"<<ts_est.at(0)<<endl;
	cout << "3D points"<<points3d_estimated.at(0)<<endl;
	
  	cout << "\n----------------------------\n" << endl;
  	cout << "Reconstruction: " << endl;
  	cout << "============================" << endl;
  	cout << "Estimated 3D points: " << points3d_estimated.size() << endl;
  	cout << "Estimated cameras: " << Rs_est.size() << endl;
  	cout << "Refined intrinsics: " << endl << K << endl << endl;
  	cout << "3D Visualization: " << endl;
  	cout << "============================" << endl;
  	viz::Viz3d window("Coordinate Frame");
             window.setWindowSize(Size(500,500));
             window.setWindowPosition(Point(150,150));
             window.setBackgroundColor(cv::viz::Color::gray()); // black by default
  	// Create the pointcloud
  	cout << "Recovering points  ... ";
  	// recover estimated points3d
  	vector<Vec3f> point_cloud_est;
  	for (int i = 0; i < points3d_estimated.size(); ++i)
    	point_cloud_est.push_back(Vec3f(points3d_estimated[i]));
  	cout << "[DONE]" << endl;
  	cout << "Recovering cameras ... ";
  	vector<Affine3d> path;
  	for (size_t i = 0; i < Rs_est.size(); ++i)
    	path.push_back(Affine3d(Rs_est[i],ts_est[i]));
  	cout << "[DONE]" << endl;
  	if ( point_cloud_est.size() > 0 )
 	{
    	cout << "Rendering points   ... ";
    	viz::WCloud cloud_widget(point_cloud_est, viz::Color::green());
    	window.showWidget("point_cloud", cloud_widget);
    	cout << "[DONE]" << endl;
  	}
  	else
  	{
    	cout << "Cannot render points: Empty pointcloud" << endl;
  	}
  	if ( path.size() > 0 )
  	{
    	cout << "Rendering Cameras  ... ";
    	window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
    	window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K, 0.1, viz::Color::yellow()));
    	window.setViewerPose(path[0]);
    	cout << "[DONE]" << endl;
  	}
  	else
  	{
    	cout << "Cannot render the cameras: Empty path" << endl;
  	}
  	cout << endl << "Press 'q' to close each windows ... " << endl;
  	window.spin();	
	
  	return 0;
}

