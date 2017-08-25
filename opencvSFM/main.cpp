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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <map>
#include <set>
#include <cmath>        


using namespace std;
using namespace cv;
using namespace cv::sfm;
using namespace xfeatures2d;

//for output file 
void writeMeshToPLYFile( vector<Vec3f>& pointVec, 
							 vector<int>& triInd1, 
							 vector<int>& triInd2, 
							 vector<int>& triInd3, 
							 vector<int>& triInd4, 
							 string& outFilename)
{
    ofstream outFile( outFilename.c_str() );

    if ( !outFile )
    {
        cerr << "Error opening output file: " << outFilename << "!" << endl;
        exit( 1 );
    }

    ////
    // Header
    ////

    int pointNum    = ( int ) pointVec.size();
    int triangleNum = ( int ) (triInd1.size() + triInd2.size() + triInd3.size() + triInd4.size())/3;

    outFile << "ply" << endl;
    outFile << "format ascii 1.0" << endl;
    outFile << "element vertex " << pointNum << endl;
    outFile << "property float x" << endl;
    outFile << "property float y" << endl;
    outFile << "property float z" << endl;
    outFile << "element face " << triangleNum << endl;
    outFile << "property list uchar int vertex_index" << endl;
    outFile << "end_header" << endl;

    ////
    // Points
    ////

    for ( int pi = 0; pi < pointNum; ++pi )
    {
        Vec3f point = pointVec[ pi ];

        outFile << point.val[0] << " ";
		outFile << point.val[1] << " ";
		outFile << point.val[2] << " ";
        outFile << endl;
    }

    ////
    // Triangles
    ////
	
    for ( int ti = 0; ti < triInd1.size(); ti=ti+3 )
    {
        outFile << "3 ";

        outFile << triInd1[ ti ] << " ";
		outFile << triInd1[ ti+1 ] << " ";
		outFile << triInd1[ ti+2 ]<< " ";
        outFile << endl;
    }
	
	for ( int ti = 0; ti < triInd2.size(); ti=ti+3 )
    {
        outFile << "3 ";

        outFile << triInd2[ ti ]<< " ";
		outFile << triInd2[ ti+1 ] << " ";
		outFile << triInd2[ ti+2 ] << " ";
        outFile << endl;
    }

	for ( int ti = 0; ti < triInd3.size(); ti=ti+3 )
    {
        outFile << "3 ";

        outFile << triInd3[ ti ] << " ";
		outFile << triInd3[ ti+1 ] << " ";
		outFile << triInd3[ ti+2 ] << " ";
        outFile << endl;
    }
	
	for ( int ti = 0; ti < triInd4.size(); ti=ti+3 )
    {
        outFile << "3 ";

        outFile << triInd4[ ti ] << " ";
		outFile << triInd4[ ti+1 ] << " ";
		outFile << triInd4[ ti+2 ] << " ";
        outFile << endl;
    }
return;
	
}


void writeMeshToPLYFile1( vector<Vec3f>& pointVec, 
							 vector<int>& triInd, 
							 string& outFilename)
{
    ofstream outFile( outFilename.c_str() );

    if ( !outFile )
    {
        cerr << "Error opening output file: " << outFilename << "!" << endl;
        exit( 1 );
    }

    ////
    // Header
    ////

    int pointNum    = ( int ) pointVec.size();
    int triangleNum = ( int ) (triInd.size())/3;

    outFile << "ply" << endl;
    outFile << "format ascii 1.0" << endl;
    outFile << "element vertex " << pointNum << endl;
    outFile << "property float x" << endl;
    outFile << "property float y" << endl;
    outFile << "property float z" << endl;
    outFile << "element face " << triangleNum << endl;
    outFile << "property list uchar int vertex_index" << endl;
    outFile << "end_header" << endl;

    ////
    // Points
    ////

    for ( int pi = 0; pi < pointNum; ++pi )
    {
        Vec3f point = pointVec[ pi ];

        outFile << point.val[0] << " ";
		outFile << point.val[1] << " ";
		outFile << point.val[2] << " ";
        outFile << endl;
    }

    ////
    // Triangles
    ////
	
    for ( int ti = 0; ti < triInd.size(); ti=ti+3 )
    {
        outFile << "3 ";

        outFile << triInd[ ti ] << " ";
		outFile << triInd[ ti+1 ] << " ";
		outFile << triInd[ ti+2 ]<< " ";
        outFile << endl;
    }
return;
	
}



//


struct trc { 
	int ind1; 
	int ind2; 
	int ind3; 
	int ind4; 
	int ind5;
	trc(){
		ind1 = -1;
		ind2 = -1;
		ind3 = -1;
		ind4 = -1;
		ind5 = -1;
	}
	trc(int v1, int v2, int v3, int v4, int v5):
		ind1(v1),
		ind2(v2),
		ind3(v3),
		ind4(v4),
		ind5(v5){}
};

class MyDelaunay : public Subdiv2D
{
	public:
		MyDelaunay(Rect rect):Subdiv2D::Subdiv2D(rect){
			validGeometry = false;
			freeQEdge = 0;
			freePoint = 0;
			recentEdge = 0;

			initDelaunay(rect);			
		};
		
    void indices(vector<int> &ind) const
    {   // skips "outer" triangles.
        int i, total = (int)(qedges.size()*4);
        vector<bool> edgemask(total, false);
        for( i = 4; i < total; i += 2 )
        {
            if( edgemask[i] )
                continue;
            Point2f a, b, c;
            int edge = i;
            int A = edgeOrg(edge, &a);
            if ( A < 4 ) continue;
            edgemask[edge] = true;
            edge = getEdge(edge, NEXT_AROUND_LEFT);
            int B = edgeOrg(edge, &b);
            if ( B < 4 ) continue;
            edgemask[edge] = true;
            edge = getEdge(edge, NEXT_AROUND_LEFT);
            int C = edgeOrg(edge, &c);
            if ( C < 4 ) continue;
            edgemask[edge] = true;

            ind.push_back(A-4);
            ind.push_back(B-4);
            ind.push_back(C-4);
        }
    }
	
   void getVtx(vector<float>& x, vector<float>& y){
		for(int i = 0 ; i < vtx.size();i++){
			x.push_back(vtx[i].pt.x);
			y.push_back(vtx[i].pt.y);
		}
	}
};


vector< DMatch > returnGoodMatches(vector< DMatch >& maches, Mat& query_descriptor, Mat& train_descriptor, Mat& img_1, Mat& img_2, double min_dist);
void trackBuilder(vector< DMatch >& matches_12, 
				  vector< DMatch >& matches_13, 
				  vector< DMatch >& matches_14, 
				  vector< DMatch >& matches_22, 
				  vector< DMatch >& matches_23, 
				  vector< DMatch >& matches_34,
				  vector <trc>& tracks);
void matchToMap(vector< DMatch >& matches, map<int,int>& myMap);
void refineMatches(vector< DMatch >& matches);
void refineTracks(vector<trc>& tracks, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<KeyPoint> keypoints_3, vector<KeyPoint> keypoints_4);
void constrctPoints2d(vector<KeyPoint> keypoints_1, 
					  vector<KeyPoint> keypoints_2, 
					  vector<KeyPoint> keypoints_3, 
					  vector<KeyPoint> keypoints_4, 
					  vector<trc> tracks, 
					  vector<Mat>& points2d,
					  Mat_<double>& frame1,
					  Mat_<double>& frame2,
					  Mat_<double>& frame3,
					  Mat_<double>& frame4,
					  vector<KeyPoint>& re_keypoints_1, 
					  vector<KeyPoint>& re_keypoints_2, 
					  vector<KeyPoint>& re_keypoints_3, 
					  vector<KeyPoint>& re_keypoints_4);
static void draw_delaunay( Mat& img, MyDelaunay& subdiv, Scalar delaunay_color )
{
 
    vector<Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);
    vector<Point> pt(3);
    Size size = img.size();
    Rect rect(0,0, size.width, size.height);
 
    for( size_t i = 0; i < triangleList.size(); i++ )
    {
        Vec6f t = triangleList[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
         
        // Draw rectangles completely inside the image.
        if ( rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
        {
            line(img, pt[0], pt[1], delaunay_color, 1, CV_AA, 0);
            line(img, pt[1], pt[2], delaunay_color, 1, CV_AA, 0);
            line(img, pt[2], pt[0], delaunay_color, 1, CV_AA, 0);
        }
    }
}
void Denaulay_index(MyDelaunay& subdiv1, MyDelaunay& subdiv2, MyDelaunay& subdiv3, MyDelaunay& subdiv4,
				   Mat_<double>& frame1, Mat_<double>& frame2, Mat_<double>& frame3, Mat_<double>& frame4,vector<trc> tracks,
				   vector<int>& triInd1, vector<int>& triInd2, vector<int>& triInd3, vector<int>& triInd4);	


int main(int argc, char* argv[])
{
  	vector<string> images_paths;
	images_paths.push_back("/home/jianwei/Documents/opencvSFM/images/p1.png");
  	images_paths.push_back("/home/jianwei/Documents/opencvSFM/images/p2.png");
  	images_paths.push_back("/home/jianwei/Documents/opencvSFM/images/p3.png");
  	images_paths.push_back("/home/jianwei/Documents/opencvSFM/images/p4.png");
	
	// Read Images
	Mat img_1 = imread( "/home/jianwei/Documents/opencvSFM/images/p1.png", IMREAD_GRAYSCALE );
	Mat img_2 = imread( "/home/jianwei/Documents/opencvSFM/images/p2.png", IMREAD_GRAYSCALE );	
	Mat img_3 = imread( "/home/jianwei/Documents/opencvSFM/images/p3.png", IMREAD_GRAYSCALE );	
	Mat img_4 = imread( "/home/jianwei/Documents/opencvSFM/images/p4.png", IMREAD_GRAYSCALE );	
	
	if( !img_1.data || !img_2.data || !img_3.data ||!img_4.data)
	{ std::cout<< " --(!) Error reading images " << std::endl; return -1; }
	
	//-- Step 1: Detect the keypoints using SURF/SIFT Detector, compute the descriptors
	int minHessian = 400;
	Ptr<SIFT> detector = SIFT::create();
	//detector->setHessianThreshold(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_3, keypoints_4;
	Mat descriptors_1, descriptors_2, descriptors_3, descriptors_4;
	
	detector->detectAndCompute( img_1, Mat(), keypoints_1, descriptors_1 );
	detector->detectAndCompute( img_2, Mat(), keypoints_2, descriptors_2 );
	detector->detectAndCompute( img_3, Mat(), keypoints_3, descriptors_3 );
	detector->detectAndCompute( img_4, Mat(), keypoints_4, descriptors_4 );
		
	//-- Step 2: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
  	std::vector< DMatch > matches_12, matches_13, matches_14, matches_23, matches_24, matches_34;
	
	matcher.match( descriptors_1, descriptors_2, matches_12 );
	matcher.match( descriptors_1, descriptors_3, matches_13 );
	matcher.match( descriptors_1, descriptors_4, matches_14 );
	matcher.match( descriptors_2, descriptors_3, matches_23 );
	matcher.match( descriptors_2, descriptors_4, matches_24 );
	matcher.match( descriptors_3, descriptors_4, matches_34 );
	
	double max_dist_12 = 0; double min_dist_12 = 100;
	double max_dist_13 = 0; double min_dist_13 = 100;
	double max_dist_14 = 0; double min_dist_14 = 100;
	double max_dist_23 = 0; double min_dist_23 = 100;
	double max_dist_24 = 0; double min_dist_24 = 100;
	double max_dist_34 = 0; double min_dist_34 = 100;
	
  	//-- Quick calculation of max and min distances between keypoints
  	for( int i = 0; i < descriptors_1.rows; i++ )
  	{ double dist = matches_12[i].distance;
    	if( dist < min_dist_12 ) min_dist_12 = dist;
    	if( dist > max_dist_12 ) max_dist_12 = dist;
  	}

	for( int i = 0; i < descriptors_1.rows; i++ )
  	{ double dist = matches_13[i].distance;
    	if( dist < min_dist_13 ) min_dist_13 = dist;
    	if( dist > max_dist_13 ) max_dist_13 = dist;
  	}
	
	for( int i = 0; i < descriptors_1.rows; i++ )
  	{ double dist = matches_14[i].distance;
    	if( dist < min_dist_14 ) min_dist_14 = dist;
    	if( dist > max_dist_14 ) max_dist_14 = dist;
  	}
	
	for( int i = 0; i < descriptors_2.rows; i++ )
  	{ double dist = matches_23[i].distance;
    	if( dist < min_dist_23 ) min_dist_23 = dist;
    	if( dist > max_dist_23 ) max_dist_23 = dist;
  	}
	
	for( int i = 0; i < descriptors_2.rows; i++ )
  	{ double dist = matches_24[i].distance;
    	if( dist < min_dist_24 ) min_dist_24 = dist;
    	if( dist > max_dist_24 ) max_dist_24 = dist;
  	}
	
	for( int i = 0; i < descriptors_3.rows; i++ )
  	{ double dist = matches_34[i].distance;
    	if( dist < min_dist_34 ) min_dist_34 = dist;
    	if( dist > max_dist_34 ) max_dist_34 = dist;
  	}
	
	printf("-- descriptors_1 : %d \n", descriptors_1.rows );
	printf("-- descriptors_2 : %d \n", descriptors_2.rows );
	printf("-- descriptors_3 : %d \n", descriptors_3.rows );
	printf("-- descriptors_4 : %d \n", descriptors_4.rows );
	
	printf("-- matches_12 : %d \n", int(matches_12.size()) );
	printf("-- matches_13 : %d \n", int(matches_13.size()) );
	printf("-- matches_14 : %d \n", int(matches_14.size()) );
	printf("-- matches_23 : %d \n", int(matches_23.size()) );
	printf("-- matches_24 : %d \n", int(matches_24.size()) );
	printf("-- matches_34 : %d \n", int(matches_34.size()) );
	
	vector< DMatch > good_matches_12 = returnGoodMatches(matches_12, descriptors_1, descriptors_2, img_1, img_2, min_dist_12);
	vector< DMatch > good_matches_13 = returnGoodMatches(matches_13, descriptors_1, descriptors_3, img_1, img_3, min_dist_13);
	vector< DMatch > good_matches_14 = returnGoodMatches(matches_14, descriptors_1, descriptors_4, img_1, img_4, min_dist_14);
	vector< DMatch > good_matches_23 = returnGoodMatches(matches_23, descriptors_2, descriptors_3, img_2, img_3, min_dist_23);
	vector< DMatch > good_matches_24 = returnGoodMatches(matches_24, descriptors_2, descriptors_4, img_2, img_4, min_dist_24);
	vector< DMatch > good_matches_34 = returnGoodMatches(matches_34, descriptors_3, descriptors_4, img_3, img_4, min_dist_34);
	
	refineMatches(good_matches_12);
	refineMatches(good_matches_13);
	refineMatches(good_matches_14);
	refineMatches(good_matches_23);
	refineMatches(good_matches_24);
	refineMatches(good_matches_34);
		
  	cout<<"Good matches_12: "<< int(good_matches_12.size()) <<endl;
	cout<<"Good matches_13: "<< int(good_matches_13.size()) <<endl;
	cout<<"Good matches_14: "<< int(good_matches_14.size()) <<endl;
	cout<<"Good matches_23: "<< int(good_matches_23.size()) <<endl;
	cout<<"Good matches_24: "<< int(good_matches_24.size()) <<endl;
	cout<<"Good matches_34: "<< int(good_matches_34.size()) <<endl;
	
	//Build track
	vector <trc> tracks;
	trackBuilder(good_matches_12, good_matches_13, good_matches_14, good_matches_23, good_matches_24, good_matches_34, tracks);
	refineTracks(tracks, keypoints_1, keypoints_2, keypoints_3, keypoints_4);
	
	vector<Mat> points2d;
	vector<KeyPoint> re_keypoints_1, re_keypoints_2, re_keypoints_3, re_keypoints_4;
	Mat_<double> frame1(2, tracks.size());
	Mat_<double> frame2(2, tracks.size());
	Mat_<double> frame3(2, tracks.size()); 
	Mat_<double> frame4(2, tracks.size());
	constrctPoints2d(keypoints_1, keypoints_2, keypoints_3, keypoints_4, 
					 tracks, points2d, frame1, frame2, frame3, frame4,
					 re_keypoints_1, re_keypoints_2, re_keypoints_3, re_keypoints_4);
	
	//perform SfM
	//Build instrinsics
  	float f  = 480;
  	float cx = 325;
  	float cy = 236;
  	Matx33d K = Matx33d( f, 0, cx,
                       0, f, cy,
                       0, 0,  1);
  	bool is_projective = true;
  	vector<Mat> Rs_est, ts_est, points3d_estimated;
	reconstruct(points2d, Rs_est, ts_est, K, points3d_estimated, is_projective);
	
	//Denaulay tiangulation
	
	Size size1 = img_1.size();
	Size size2 = img_2.size();
	Size size3 = img_3.size();
	Size size4 = img_4.size();
	
	Rect rect1(0, 0, size1.width, size1.height);
	Rect rect2(0, 0, size2.width, size2.height);
	Rect rect3(0, 0, size3.width, size3.height);
	Rect rect4(0, 0, size4.width, size4.height);
			
	MyDelaunay subdiv1(rect1);
	MyDelaunay subdiv2(rect2);
	MyDelaunay subdiv3(rect3);
	MyDelaunay subdiv4(rect4);
	
	vector<int> triInd1;
	vector<int> triInd2;
	vector<int> triInd3;
	vector<int> triInd4;
	
	
	Denaulay_index( subdiv1, subdiv2, subdiv3, subdiv4,
				   frame1, frame2, frame3, frame4, tracks,
				   triInd1, triInd2, triInd3, triInd4);	
		
	vector<Vec3f> point_cloud_est;
	for (int i = 0; i < points3d_estimated.size(); ++i)
    	point_cloud_est.push_back(Vec3f(points3d_estimated[i]));
	
	string outFilename = "mymesh.ply";
	writeMeshToPLYFile( point_cloud_est, 
							 triInd1, 
							 triInd2, 
							 triInd3, 
							 triInd4, 
							 outFilename);
	string outFilename1 = "mymesh1.ply";
	writeMeshToPLYFile1(point_cloud_est, 
					triInd1, 
					outFilename1);	
	
	string outFilename2 = "mymesh2.ply";
	writeMeshToPLYFile1(point_cloud_est, 
					triInd2, 
					outFilename2);	
	
	string outFilename3 = "mymesh3.ply";
	writeMeshToPLYFile1(point_cloud_est, 
					triInd3, 
					outFilename3);	
	
	string outFilename4 = "mymesh4.ply";
	writeMeshToPLYFile1(point_cloud_est, 
					triInd4, 
					outFilename4);	
	
	// Define colors for drawing.
    Scalar delaunay_color(255,255,0);
	
	Mat img1_copy = img_1.clone();
	Mat img2_copy = img_2.clone();
	Mat img3_copy = img_3.clone();
	Mat img4_copy = img_4.clone();
	
	draw_delaunay( img1_copy, subdiv1, delaunay_color );
	draw_delaunay( img2_copy, subdiv2, delaunay_color );
	draw_delaunay( img3_copy, subdiv3, delaunay_color );
	draw_delaunay( img4_copy, subdiv4, delaunay_color );
	
	Mat img_deln_fea_1;
	Mat img_deln_fea_2;
	Mat img_deln_fea_3;
	Mat img_deln_fea_4;
	
	drawKeypoints( img1_copy, re_keypoints_1, img_deln_fea_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	drawKeypoints( img2_copy, re_keypoints_2, img_deln_fea_2, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	drawKeypoints( img3_copy, re_keypoints_3, img_deln_fea_3, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	drawKeypoints( img4_copy, re_keypoints_4, img_deln_fea_4, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
	
	imshow("img1_delaunay",img_deln_fea_1);
	imshow("img2_delaunay",img_deln_fea_2);
	imshow("img3_delaunay",img_deln_fea_3);
	imshow("img4_delaunay",img_deln_fea_4);
	moveWindow("img1_delaunay", 20,20);
	moveWindow("img2_delaunay", 30,30);
	moveWindow("img3_delaunay", 40,40);
	moveWindow("img4_delaunay", 50,50);
	waitKey(0);
	
	
	// Print output
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
             window.setBackgroundColor(); // black by default
    // Create the pointcloud
    cout << "Recovering points  ... ";
    // recover estimated points3d
	//vector<Vec3f> point_cloud_est;
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
	
    	waitKey(0);
		
		return 0;
}

vector< DMatch > returnGoodMatches(vector< DMatch >& matches, Mat& query_descriptor, Mat& train_descriptor, Mat& img_1, Mat& img_2, double min_dist){
  	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
  	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
  	//-- small)
  	//-- PS.- radiusMatch can also be used here.	
	vector< DMatch > good_matches;
  	for( int i = 0; i < query_descriptor.rows; i++ )
  	{ if( matches[i].distance <= max(2*min_dist,0.02) )
    	{ good_matches.push_back( matches[i]); }
  	}
	return good_matches;
}


void trackBuilder(vector< DMatch >& matches_12, 
				  vector< DMatch >& matches_13, 
				  vector< DMatch >& matches_14, 
				  vector< DMatch >& matches_23, 
				  vector< DMatch >& matches_24, 
				  vector< DMatch >& matches_34,
				  vector <trc>& tracks){
	
	map<int,int> map12, map13, map14, map23, map24, map34;
	matchToMap(matches_12, map12);
	matchToMap(matches_13, map13);
	matchToMap(matches_14, map14);
	matchToMap(matches_23, map23);
	matchToMap(matches_24, map24);
	matchToMap(matches_34, map34);
	
	vector<int> ind1;
	vector<int> ind2;
	vector<int> ind3;
	vector<int> ind4;
	vector<int> ind5;
	
	//add pair (1,2) to existing tracks
	for( vector< DMatch >::iterator it = matches_12.begin(); it != matches_12.end(); ++it ){
		
		trc* curTrc = new trc();
		curTrc->ind1 = it->queryIdx;
		curTrc->ind2 = it->trainIdx;
		//track_1_.push_back(*new trc(it->queryIdx,it->trainIdx,-1,-1,-1));
		tracks.push_back(*curTrc);
		ind1.push_back(curTrc->ind1);
		ind2.push_back(curTrc->ind2);
		ind3.push_back(curTrc->ind3);
		ind4.push_back(curTrc->ind4);
		ind5.push_back(curTrc->ind5);
	}
	//add pair (1,3) to existing tracks
	for( vector< DMatch >::iterator it = matches_13.begin(); it != matches_13.end(); ++it ){
		
		//find if match (1,3) is linked to (1,2)
		vector< int >::iterator intIt = find(ind1.begin(), ind1.end(),it->queryIdx);
		if(intIt != ind1.end()){ // Element found
			int index = distance(ind1.begin(),intIt);
			ind3[index] = it->trainIdx;
			tracks[index].ind3 = it->trainIdx;
		}
		else{									//Element not found
			trc* curTrc = new trc();
			curTrc->ind1 = it->queryIdx;
			curTrc->ind3 = it->trainIdx;
			tracks.push_back(*curTrc);
			ind1.push_back(curTrc->ind1);
			ind2.push_back(curTrc->ind2);
			ind3.push_back(curTrc->ind3);
			ind4.push_back(curTrc->ind4);
			ind5.push_back(curTrc->ind5);			
		}	
	}	
	//add pair (1,4) to existing tracks
	for( vector< DMatch >::iterator it = matches_14.begin(); it != matches_14.end(); ++it ){
		
		//find if match (1,4) is linked to existing tracks
		vector< int >::iterator intIt = find(ind1.begin(), ind1.end(),it->queryIdx);
		if(intIt != ind1.end()){ // Element found
			int index = distance(ind1.begin(),intIt);
			ind4[index] = it->trainIdx;
			tracks[index].ind4 = it->trainIdx;
		}
		else{									//Element not found
			trc* curTrc = new trc();
			curTrc->ind1 = it->queryIdx;
			curTrc->ind4 = it->trainIdx;
			tracks.push_back(*curTrc);
			ind1.push_back(curTrc->ind1);
			ind2.push_back(curTrc->ind2);
			ind3.push_back(curTrc->ind3);
			ind4.push_back(curTrc->ind4);
			ind5.push_back(curTrc->ind5);			
		}	
	}
	
	//add pair (2,3) to existing tracks
	//cout<<"------------Adding pair(2,3)----------"<<endl;
	for( vector< DMatch >::iterator it = matches_23.begin(); it != matches_23.end(); ++it ){
		//find if match (2,3) is linked to existing tracks
		vector< int >::iterator intIt2 = find(ind2.begin(), ind2.end(),it->queryIdx);
		if(intIt2 != ind2.end()){ // Element found
			int index = distance(ind2.begin(),intIt2);
			if(ind3[index] == it->trainIdx){ //Element found. Already exist in the track
				// do nothing
				//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"Already exist"<<endl;
			}else{//Element not found
				if(ind3[index] == -1){//extend the existing track.
					ind3[index] = it->trainIdx;
					tracks[index].ind3 = it->trainIdx;
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"2 found,3 not. Appended."<<endl;
				}else{ //conflict,delete this track
					tracks.erase(tracks.begin()+index);
					ind1.erase(ind1.begin()+index);
					ind2.erase(ind2.begin()+index);
					ind3.erase(ind3.begin()+index);
					ind4.erase(ind4.begin()+index);
					ind5.erase(ind5.begin()+index);
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"conflict. Delete"<<endl;
				} 
			}
		}
		else{//Element not found
			vector< int >::iterator intIt3 = find(ind3.begin(), ind3.end(),it->trainIdx);
			if(intIt3 != ind3.end()){ // Element found
				int index = distance(ind3.begin(),intIt3);
				if(ind2[index] == -1){ //Element found. Extend the existing track
					ind2[index] = it->queryIdx;
					tracks[index].ind2 = it->queryIdx;
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"2 not,3 found. Appended."<<endl;
				}else{  //conflict, delete this track
					tracks.erase(tracks.begin()+index);
					ind1.erase(ind1.begin()+index);
					ind2.erase(ind2.begin()+index);
					ind3.erase(ind3.begin()+index);
					ind4.erase(ind4.begin()+index);
					ind5.erase(ind5.begin()+index);
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"conflict. Delete."<<endl;
				}
			}else{ //Elementn not found. This is a new track.
				trc* curTrc = new trc();
				curTrc->ind2 = it->queryIdx;
				curTrc->ind3 = it->trainIdx;
				tracks.push_back(*curTrc);
				ind1.push_back(curTrc->ind1);
				ind2.push_back(curTrc->ind2);
				ind3.push_back(curTrc->ind3);
				ind4.push_back(curTrc->ind4);
				ind5.push_back(curTrc->ind5);	
				//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"new track"<<endl;
			}		
		}	
	}	
	
	//add pair (2,4)
	//cout<<"------------Adding pair(2,4)----------"<<endl;
	for( vector< DMatch >::iterator it = matches_24.begin(); it != matches_24.end(); ++it ){
		//find if match (2,4) is linked to existing tracks
		vector< int >::iterator intIt2 = find(ind2.begin(), ind2.end(),it->queryIdx);
		if(intIt2 != ind2.end()){ // Element found
			int index = distance(ind2.begin(),intIt2);
			if(ind4[index] == it->trainIdx){ //Element found. Already exist in the track
				// do nothing
				//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"Already exist"<<endl;
			}else{//Element not found
				if(ind4[index] == -1){//extend the existing track.
					ind4[index] = it->trainIdx;
					tracks[index].ind4 = it->trainIdx;
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"2 found,4 not. Appended."<<endl;
				}else{ //conflict,delete this track
					tracks.erase(tracks.begin()+index);
					ind1.erase(ind1.begin()+index);
					ind2.erase(ind2.begin()+index);
					ind3.erase(ind3.begin()+index);
					ind4.erase(ind4.begin()+index);
					ind5.erase(ind5.begin()+index);
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"conflict. Delete"<<endl;
				} 
			}
		}
		else{//Element not found
			vector< int >::iterator intIt4 = find(ind4.begin(), ind4.end(), it->trainIdx);
			if(intIt4 != ind4.end()){ // Element found
				int index = distance(ind4.begin(),intIt4);
				if(ind2[index] == -1){ //Element found. Extend the existing track
					ind2[index] = it->queryIdx;
					tracks[index].ind2 = it->queryIdx;
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"2 not,4 found. Appended."<<endl;
				}else{  //conflict, delete this track
					tracks.erase(tracks.begin()+index);
					ind1.erase(ind1.begin()+index);
					ind2.erase(ind2.begin()+index);
					ind3.erase(ind3.begin()+index);
					ind4.erase(ind4.begin()+index);
					ind5.erase(ind5.begin()+index);
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"conflict. Delete."<<endl;
				}
			}else{ //Elementn not found. This is a new track.
				trc* curTrc = new trc();
				curTrc->ind2 = it->queryIdx;
				curTrc->ind4 = it->trainIdx;
				tracks.push_back(*curTrc);
				ind1.push_back(curTrc->ind1);
				ind2.push_back(curTrc->ind2);
				ind3.push_back(curTrc->ind3);
				ind4.push_back(curTrc->ind4);
				ind5.push_back(curTrc->ind5);	
				//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"new track"<<endl;
			}		
		}	
	}

	//add pair (3,4)	
	//cout<<"------------Adding pair(3,4)----------"<<endl;
	for( vector< DMatch >::iterator it = matches_34.begin(); it != matches_34.end(); ++it ){
		//find if match (3,4) is linked to existing tracks
		vector< int >::iterator intIt3 = find(ind3.begin(), ind3.end(),it->queryIdx);
		if(intIt3 != ind3.end()){ // Element in view 3 found
			int index = distance(ind3.begin(),intIt3);
			if(ind4[index] == it->trainIdx){ //Element in view 4 found. Already exist in the track
				// do nothing
				//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"Already exist"<<endl;
			}else{//Element in view 4 not found
				if(ind4[index] == -1){//extend the existing track.
					ind4[index] = it->trainIdx;
					tracks[index].ind4 = it->trainIdx;
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"3 found,4 not. Appended."<<endl;
				}else{ //conflict,delete this track
					tracks.erase(tracks.begin()+index);
					ind1.erase(ind1.begin()+index);
					ind2.erase(ind2.begin()+index);
					ind3.erase(ind3.begin()+index);
					ind4.erase(ind4.begin()+index);
					ind5.erase(ind5.begin()+index);
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"conflict. Delete"<<endl;
				} 
			}
		}
		else{//Element not found
			vector< int >::iterator intIt4 = find(ind4.begin(), ind4.end(), it->trainIdx);
			if(intIt4 != ind4.end()){ // Element found
				int index = distance(ind4.begin(),intIt4);
				if(ind3[index] == -1){ //Element found. Extend the existing track
					ind3[index] = it->queryIdx;
					tracks[index].ind3 = it->queryIdx;
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"3 not,4 found. Appended."<<endl;
				}else{  //conflict, delete this track
					tracks.erase(tracks.begin()+index);
					ind1.erase(ind1.begin()+index);
					ind2.erase(ind2.begin()+index);
					ind3.erase(ind3.begin()+index);
					ind4.erase(ind4.begin()+index);
					ind5.erase(ind5.begin()+index);
					//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"conflict. Delete."<<endl;
				}
			}else{ //Elementn not found. This is a new track.
				trc* curTrc = new trc();
				curTrc->ind3 = it->queryIdx;
				curTrc->ind4 = it->trainIdx;
				tracks.push_back(*curTrc);
				ind1.push_back(curTrc->ind1);
				ind2.push_back(curTrc->ind2);
				ind3.push_back(curTrc->ind3);
				ind4.push_back(curTrc->ind4);
				ind5.push_back(curTrc->ind5);	
				//cout<<"("<<it->queryIdx<<","<<it->trainIdx<<")"<<"new track"<<endl;
			}		
		}	
	}
	
	//cout<<"------------current track----------------"<<endl;
	//for( vector< trc >::iterator it = tracks.begin(); it != tracks.end(); ++it ){
	//	cout<<"("<< it->ind1<<","
	//		<<it->ind2<<","
	//		<<it->ind3<<","
	//		<<it->ind4<<","
	//		<<it->ind5<<")"
	//		<<endl;
	//}
}

void refineMatches(vector< DMatch >& matches){ 
	
	//(1)get rid of duplicated tracks
	vector<int> left;
	vector<int> right;
	for( vector< DMatch >::iterator it = matches.begin(); it != matches.end(); ++it ){
		left.push_back(it->queryIdx);
		right.push_back(it->trainIdx);
	}
	int i = 0;
	int length = right.size();
	std::set<int> duplicate;
	while (i < length){
		vector<int>::iterator it = find(right.begin()+i+1, right.end(), right[i]);
		if(it != right.end()){ // element found
			int index = distance(right.begin(),it);
			right.erase(right.begin()+index);
			matches.erase(matches.begin()+index);
			left.erase(right.begin()+index);
			length = right.size();
			duplicate.insert(right[i]);
			continue;
		}
		i++;
	}
	set<int>:: iterator setIt;
	for( setIt = duplicate.begin();setIt!=duplicate.end();++setIt){
		int cur = *setIt;
		vector<int>::iterator it = find(right.begin(), right.end(), cur);
		if(it != right.end()){
			int index = distance(right.begin(),it);	
			right.erase(right.begin()+index);
			left.erase(right.begin()+index);
			matches.erase(matches.begin()+index);
		}
	}	
}

void refineTracks(vector<trc>& tracks, vector<KeyPoint> keypoints_1, vector<KeyPoint> keypoints_2, vector<KeyPoint> keypoints_3, vector<KeyPoint> keypoints_4){

	vector<int> ind1, ind2, ind3, ind4;
	
	for( vector< trc >::iterator it = tracks.begin(); it != tracks.end(); ++it ){
		ind1.push_back(it->ind1);
		ind2.push_back(it->ind2);
		ind3.push_back(it->ind3);
		ind4.push_back(it->ind4);
	}
	
	//(2)get rid of duplicate feature points
	//check query image
	//view 1 
	int i = 0;
	int length = ind1.size();
	while(i < length){
		if(ind1[i]!=-1){
			float curX = keypoints_1[ ind1[i] ].pt.x;
			float curY = keypoints_1[ ind1[i] ].pt.y;
			bool duplicated = false;
			for(int j = i+1; j < ind1.size(); j++){
				if(ind1[j]!=-1){
					float temX = keypoints_1[ ind1[j] ].pt.x;
					float temY = keypoints_1[ ind1[j] ].pt.y;
					if( abs(curX-temX)< 0.001 && abs(curY-temY)< 0.001 ){
						duplicated = true;
						ind1.erase(ind1.begin()+j);
						ind2.erase(ind2.begin()+j);
						ind3.erase(ind3.begin()+j);
						ind4.erase(ind4.begin()+j);
						tracks.erase(tracks.begin()+j);
						length = ind1.size();
						break;
					}
				}
			}
			if(!duplicated){
				i++;
			}
		}else{
			i++;
		}
	}
	//view 2 
	i = 0;
	length = ind2.size();
	while(i < length){
		if(ind2[i]!=-1){
			float curX = keypoints_2[ ind2[i] ].pt.x;
			float curY = keypoints_2[ ind2[i] ].pt.y;
			bool duplicated = false;
			for(int j = i+1; j < ind2.size(); j++){
				if(ind2[j]!=-1){
					float temX = keypoints_2[ ind2[j] ].pt.x;
					float temY = keypoints_2[ ind2[j] ].pt.y;
					if( abs(curX-temX)< 0.001 && abs(curY-temY)< 0.001 ){
						duplicated = true;
						ind1.erase(ind1.begin()+j);
						ind2.erase(ind2.begin()+j);
						ind3.erase(ind3.begin()+j);
						ind4.erase(ind4.begin()+j);
						tracks.erase(tracks.begin()+j);
						length = ind2.size();
						break;
					}
				}
			}
			if(!duplicated){
				i++;
			}
		}else{
			i++;
		}
	}
	
	//view 3
	i = 0;
	length = ind3.size();
	while(i < length){
		if(ind3[i]!=-1){
			float curX = keypoints_3[ ind3[i] ].pt.x;
			float curY = keypoints_3[ ind3[i] ].pt.y;
			bool duplicated = false;
			for(int j = i+1; j < ind3.size(); j++){
				if(ind3[j]!=-1){
					float temX = keypoints_3[ ind3[j] ].pt.x;
					float temY = keypoints_3[ ind3[j] ].pt.y;
					if( abs(curX-temX)< 0.001 && abs(curY-temY)< 0.001 ){
						duplicated = true;
						ind1.erase(ind1.begin()+j);
						ind2.erase(ind2.begin()+j);
						ind3.erase(ind3.begin()+j);
						ind4.erase(ind4.begin()+j);
						tracks.erase(tracks.begin()+j);
						length = ind3.size();
						break;
					}
				}
			}
			if(!duplicated){
				i++;
			}
		}else{
			i++;
		}
	}
	
	//view 4
	i = 0;
	length = ind4.size();
	while(i < length){
		if(ind4[i]!=-1){
			float curX = keypoints_4[ ind4[i] ].pt.x;
			float curY = keypoints_4[ ind4[i] ].pt.y;
			bool duplicated = false;
			for(int j = i+1; j < ind4.size(); j++){
				if(ind4[j]!=-1){
					float temX = keypoints_4[ ind4[j] ].pt.x;
					float temY = keypoints_4[ ind4[j] ].pt.y;
					if( abs(curX-temX)< 0.001 && abs(curY-temY)< 0.001 ){
						duplicated = true;
						ind1.erase(ind1.begin()+j);
						ind2.erase(ind2.begin()+j);
						ind3.erase(ind3.begin()+j);
						ind4.erase(ind4.begin()+j);
						tracks.erase(tracks.begin()+j);
						length = ind4.size();
						break;
					}
				}
			}
			if(!duplicated){
				i++;
			}
		}else{
			i++;
		}
	}		
}

void matchToMap(vector< DMatch >& matches, map<int,int>& myMap){
	for( vector< DMatch >::iterator it = matches.begin(); it != matches.end(); ++it ){
		myMap[it->queryIdx] = it->trainIdx;
	}	
	//for( vector< DMatch >::iterator it = matches.begin(); it != matches.end(); ++it ){
	//	cout<<"("<<it->queryIdx
	//		<<","<<myMap[it->queryIdx]<<")"<<endl;
	//}
}

void constrctPoints2d(vector<KeyPoint> keypoints_1, 
					  vector<KeyPoint> keypoints_2, 
					  vector<KeyPoint> keypoints_3, 
					  vector<KeyPoint> keypoints_4, 
					  vector<trc> tracks, 
					  vector<Mat>& points2d,
					  Mat_<double>& frame1,
					  Mat_<double>& frame2,
					  Mat_<double>& frame3,
					  Mat_<double>& frame4,
					  vector<KeyPoint>& re_keypoints_1, 
					  vector<KeyPoint>& re_keypoints_2, 
					  vector<KeyPoint>& re_keypoints_3, 
					  vector<KeyPoint>& re_keypoints_4)
{
	int n_tracks = tracks.size();
	for( int i = 0; i < n_tracks; ++i ){
		
		int index1 = tracks[i].ind1;
		int index2 = tracks[i].ind2;
		int index3 = tracks[i].ind3;
		int index4 = tracks[i].ind4;
		if(index1 != -1){
			frame1(0,i) = keypoints_1[index1].pt.x;
			frame1(1,i) = keypoints_1[index1].pt.y;
			re_keypoints_1.push_back(keypoints_1[index1]);
		}else{
			frame1(0,i) = -1;
			frame1(1,i) = 0;
		}
		
		if(index2 != -1){
			frame2(0,i) = keypoints_2[index2].pt.x;
			frame2(1,i) = keypoints_2[index2].pt.y;
			re_keypoints_2.push_back(keypoints_2[index2]);
		}else{
			frame2(0,i) = -1;
			frame2(1,i) = 0;
		}		
		
		if(index3 != -1){
			frame3(0,i) = keypoints_3[index3].pt.x;
			frame3(1,i) = keypoints_3[index3].pt.y;
			re_keypoints_3.push_back(keypoints_3[index3]);
		}else{
			frame3(0,i) = -1;
			frame3(1,i) = 0;
		}
		
		if(index4 != -1){
			frame4(0,i) = keypoints_4[index4].pt.x;
			frame4(1,i) = keypoints_4[index4].pt.y;
			re_keypoints_4.push_back(keypoints_4[index4]);
		}else{
			frame4(0,i) = -1;
			frame4(1,i) = 0;
		}
	}
	points2d.push_back(Mat(frame1));
	points2d.push_back(Mat(frame2));
	points2d.push_back(Mat(frame3));
	points2d.push_back(Mat(frame4));
}

void Denaulay_index(MyDelaunay& subdiv1, MyDelaunay& subdiv2, MyDelaunay& subdiv3, MyDelaunay& subdiv4,
				   Mat_<double>& frame1, Mat_<double>& frame2, Mat_<double>& frame3, Mat_<double>& frame4, vector<trc> tracks,
				   vector<int>& triInd1, vector<int>& triInd2, vector<int>& triInd3, vector<int>& triInd4){

	vector<Point2f> points1;
	vector<Point2f> points2;
	vector<Point2f> points3;
	vector<Point2f> points4;
	vector<int> indfor1, indfor2, indfor3, indfor4;

	cout<<"------------current track----------------"<<endl;
	int q = 0;
	for( vector< trc >::iterator it = tracks.begin(); it != tracks.end(); ++it ){
		cout<<q<<" :("<< it->ind1<<","
			<<it->ind2<<","
			<<it->ind3<<","
			<<it->ind4<<","
			<<it->ind5<<")"
			<<endl;
		q++;
	}		
	
	for(int i = 0; i < tracks.size();++i){
		if(frame1(0,i)!=-1){
			subdiv1.insert(Point2f(frame1(0,i),frame1(1,i)));
			points1.push_back(Point2f(frame1(0,i),frame1(1,i)));
			indfor1.push_back(i);
		}
		if(frame2(0,i)!=-1){
			subdiv2.insert(Point2f(frame2(0,i),frame2(1,i)));
			points2.push_back(Point2f(frame2(0,i),frame2(1,i)));
			indfor2.push_back(i);
		}
		
		if(frame3(0,i)!=-1){
			subdiv3.insert(Point2f(frame3(0,i),frame3(1,i)));
			points3.push_back(Point2f(frame3(0,i),frame3(1,i)));
			indfor3.push_back(i);
		}
		
		if(frame4(0,i)!=-1){
			subdiv4.insert(Point2f(frame4(0,i),frame4(1,i)));
			points4.push_back(Point2f(frame4(0,i),frame4(1,i)));
			indfor4.push_back(i);
		}
	}
	
	vector<int> myind1;
	vector<int> myind2;
	vector<int> myind3;
	vector<int> myind4;
	subdiv1.indices(myind1);
	subdiv2.indices(myind2);
	subdiv3.indices(myind3);
	subdiv4.indices(myind4);
	
	//cout<<"points1 X"<<endl;
	//for(int i = 0; i < points1.size(); i++){
	//	cout<<points1[i].x<<endl;
	//}
	
	//cout<<"points1 Y"<<endl;
	//for(int i = 0; i < points1.size(); i++){
	//	cout<<points1[i].y<<endl;
	//}	
	
	//cout<<"cTri:"<<endl;
	//for(int i = 0; i < myind1.size();i = i+3){
	//	cout<<myind1[i]<<","<<myind1[i+1]<<","<<myind1[i+2]<<";"<<endl;
	//}	
	
	for(int i = 0; i < myind1.size();i++){
		triInd1.push_back(indfor1[myind1[i]]);
	}
	for(int i = 0; i < myind2.size();i++){
		triInd2.push_back(indfor2[myind2[i]]);
	}
	for(int i = 0; i < myind3.size();i++){
		triInd3.push_back(indfor3[myind3[i]]);
	}
	for(int i = 0; i < myind4.size();i++){
		triInd4.push_back(indfor4[myind4[i]]);
	}	
}














