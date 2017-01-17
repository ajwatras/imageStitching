#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/video/background_segm.hpp"
#include <iostream>
#include <cmath>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::cuda;
using namespace std;

struct GPUResult
{
	Mat pano;
	Mat H;
	GpuMat kptsA;
	GpuMat kptsB;
	vector<DMatch> goodMatches;
	Mat drawMatches;
	GpuMat enlargedImageA;
	GpuMat enlargedImageB;
	Point shift;
	bool status = false;
};

void PullKpPosition(GpuMat kps, GpuMat Xpos, GpuMat Ypos, vector<Point2f> pts){
	Xpos = kps.row(0);
	Ypos = kps.row(1);

	Mat X,Y;

	X = Mat(Xpos);
	Y = Mat(Ypos);

	for(int i = 0; i < Xpos.cols; i++){
		//int x = X.at<int>[i,0];
		Point2f pt(X.at<int>(i),Y.at<int>(i));
		pts.push_back(pt);
	}

}

/*
void matchKeypoints(GpuMat featureA, GpuMat featureB, GPUResult& result)
	{
		// if there is enougn good matches, find H

		//Does not run on gpu
		if (result.goodMatches.size() > 10)
		{
			SURF_CUDA surf;
			vector<Point2f> ptsA, ptsB;
			vector<KeyPoint> kpsA,kpsB;

			surf.downloadKeypoints(result.kptsA,kpsA);
			surf.downloadKeypoints(result.kptsB,kpsB);

			for (vector<DMatch>::iterator itr = result.goodMatches.begin(); itr != result.goodMatches.end(); itr++)
			{
				ptsA.push_back(kpsA[itr->queryIdx].pt);
				ptsB.push_back(kpsB[itr->trainIdx].pt);
			}

			result.H = findHomography(ptsA, ptsB, CV_FM_RANSAC, this->reprojThresh);
			result.status = true;
			return;
		}
		else
		{
			result.status = false;
			cout << "No valid Homography" << endl;
		}

	}
*/

int main(int argc, char** argv){

}