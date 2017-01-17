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

struct StitchingResult
{
	Mat pano;
	Mat H;
	vector<KeyPoint> kptsA;
	vector<KeyPoint> kptsB;
	vector<DMatch> goodMatches;
	Mat drawMatches;
	Mat enlargedImageA;
	Mat enlargedImageB;
	Point shift;
	bool status = false;
};
struct GPUResult
{
	Mat pano;
	Mat H;
	GpuMat kptsA;
	GpuMat kptsB;
	vector<Point2f> ptsA;
	vector<Point2f> ptsB;
	vector<DMatch> goodMatches;
	Mat drawMatches;
	GpuMat enlargedImageA;
	GpuMat enlargedImageB;
	Point shift;
	bool status = false;
};

class ImageStitcher
{
private:
	// Variables used to change stitcher settings.ie.Tunable Parameters
	int BLEND_IMAGES = 0;
	int EDGE_CORRECTION = 0;
	int DEBUGGING = 0;
	int EROSION_LOOPS = 1;
	int DILATION_LOOPS = 6;
	int EDGE_WIN_SIZE = 40;
	int SEAM_PAD = 45;

	int minHessian = 400;
	double ratio = 0.75;
	double reprojThresh = 4;
	bool showMatches = true;
	bool reStitching = false;
	int seam = 0;

public: 
	// The following tic and toc functions were designed under a creative commons
	// license by Tim Zaman. They function like matlab's tic and toc. 
	double tt_tic=0; 
	SURF_CUDA surf;
	void tic(){
    	tt_tic = getTickCount();
	}
	void toc(){
    	double tt_toc = (getTickCount() - tt_tic)/(getTickFrequency());
    	printf ("toc: %f milliseconds \n", tt_toc*1000);

	}

	void stitch(vector<GpuMat> images, GPUResult& result, double ratio = 0.75, double reprojThresh = 4,
		bool showMatches = false, bool reStitching = false, int seam = 0)
	{
		this->ratio = ratio;
		this->reprojThresh = reprojThresh;
		this->showMatches = showMatches;
		this->reStitching = reStitching;
		this->seam = seam;

		double start_time = getTickCount();

		if (images.size() != 2)
		{
			cout << "ERRPR: the number of images being stitched should be 2" << endl;
			return;
		}

		GpuMat imageB = images.back();
		images.pop_back();
		GpuMat imageA = images.back();
		images.pop_back();

		GpuMat featureA, featureB;

		tic();
		detectAndDescribe(imageA, imageB, result, featureA, featureB);
		cout << "Detect and Describe: ";
		toc(); tic();
		cout << "Match Keypoints: ";
		matchKeypoints(featureA, featureB, result);
		
		if (result.status == false)
		{
			cout << "stitching failed" << endl;
			return;
		}
		tic();
		applyHomography(imageA, imageB, result);
		cout << "Apply Homography: ";toc();

		if (result.status == false)
		{
			cout << "stitching failed" << endl;
			return;
		}
		tic();
		stitchImage(result);
		cout << "Stitch Image: "; toc();
		if (showMatches)
		{
			drawMatches(imageA, imageB, result);
		}
		double total_time = (getTickCount() - start_time)/(getTickFrequency());
		cout << "Total Computation Time for stitch: " << total_time*1000 << " Milliseconds" << endl;
		cout << "Frames Per Second: " << 1/total_time << endl;
	}

	void stitchImage(GPUResult& result)
	{
		//tested
		if (BLEND_IMAGES == 1)
		{
			GpuMat maskAC3, maskBC3,maskC3,temp;
			cuda::threshold(result.enlargedImageA, maskAC3,0,1,THRESH_BINARY);
			cuda::threshold(result.enlargedImageB, maskBC3,0,1,THRESH_BINARY);
			cuda::add(maskAC3,maskBC3,maskC3);
			cuda::add(result.enlargedImageA, result.enlargedImageB,temp);
			cuda::divide(temp, maskC3, temp);
			result.pano = Mat(temp);
		}
		else
		{
			GpuMat maskC3;
			GpuMat output;
			cuda::threshold(result.enlargedImageB, maskC3, 0, 1, THRESH_BINARY_INV);
			cuda::multiply(maskC3,result.enlargedImageA,output);
			cuda::add(output,result.enlargedImageB,output);
			result.pano = Mat(output);
		}
	}
	void detectAndDescribe(GpuMat img1, GpuMat img2, GPUResult& result, GpuMat& featuresA, GpuMat& featuresB)
	{
   		//SURF_CUDA surf;

    	// detecting keypoints & computing descriptors
    	surf(img1, GpuMat(), result.kptsA, featuresA);
    	surf(img2, GpuMat(), result.kptsB, featuresB);

    	cout << "FOUND " << result.kptsA.cols << " keypoints on first image" << endl;
    	cout << "FOUND " << result.kptsB.cols << " keypoints on second image" << endl;

    	// matching descriptors
    	Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
    	matcher->match(featuresA, featuresB, result.goodMatches );
    }

	void matchKeypoints(GpuMat featureA, GpuMat featureB, GPUResult& result)
	{
		// if there are enougn good matches, find H
		//SURF_CUDA surf;
    	//Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(surf.defaultNorm());
    	//matcher->match(featureA, featureB, result.goodMatches );

		//Does not run on gpu
		if (result.goodMatches.size() > 10)
		{
			
			//SURF_CUDA surf;
			vector<Point2f> ptsA, ptsB;
			vector<KeyPoint> kpsA,kpsB;
			
			surf.downloadKeypoints(result.kptsA,kpsA);
			surf.downloadKeypoints(result.kptsB,kpsB);

			for (vector<DMatch>::iterator itr = result.goodMatches.begin(); itr != result.goodMatches.end(); itr++)
			{
				ptsA.push_back(kpsA[itr->queryIdx].pt);
				ptsB.push_back(kpsB[itr->trainIdx].pt);
			}
			toc();tic();
			
			result.H = findHomography(ptsA, ptsB, CV_FM_RANSAC, this->reprojThresh);
			result.status = true;
			cout << "Calculate homography: "; toc(); 


			return;
		}
		else
		{
			result.status = false;
			cout << "No valid Homography" << endl;
		}
	}

	void drawMatches(GpuMat imageA, GpuMat imageB, GPUResult& result)
	{
		SURF_CUDA surf;
		// tested Does not run on GPU.
		vector<KeyPoint> kps1,kps2;
		surf.downloadKeypoints(result.kptsA, kps1);
    	surf.downloadKeypoints(result.kptsB, kps2);

		cv::drawMatches(Mat(imageA), kps1, Mat(imageB), kps2, result.goodMatches, result.drawMatches); 
	}


	void applyHomography(GpuMat imageA, GpuMat imageB, GPUResult& result)
	{
		// fixed imageB

		int imageAWidth = imageA.size().width;
		int imageAHeight = imageA.size().height;
		int imageBWidth = imageB.size().width;
		int imageBHeight = imageB.size().height;

		Mat corner = (Mat_<double>(3, 4) << 0, 0, imageAWidth, imageAWidth,
			0, imageAHeight, 0, imageAHeight,
			1, 1, 1, 1);

		// find image bound and shift of imageB
		// Performed on CPU as there is little need for GPU parallelization

		Mat imgBound = result.H * corner;
		Mat xBound;
		cv::divide(imgBound.row(0), imgBound.row(2), xBound);
		Mat yBound;
		cv::divide(imgBound.row(1), imgBound.row(2), yBound);
		int xShift = 0;
		int yShift = 0;

		double min, max;
		minMaxLoc(xBound, &min, &max); 
		
		if (min < 0)
		{
			xShift = -(int)(min);
		}
		minMaxLoc(yBound, &min, &max);
		if (min < 0)
		{
			yShift = -(int)(min);
		}

		result.shift.y = yShift;
		result.shift.x = xShift;

		Mat shiftM = (Mat_<double>(3, 3) << 1, 0, xShift, 0, 1, yShift, 0, 0, 1);

		minMaxLoc(xBound, &min, &max);
		int x_Bound = findMax((int)(max), imageBWidth);
		
		minMaxLoc(yBound, &min, &max);
		int y_Bound = findMax((int)(max), imageBHeight);

		cout << "X Bound: " << x_Bound << "  Y Bound: " << y_Bound << "  X Shift: " << xShift << "  Y Shift: " << yShift << endl;

		int SIZE_BOUNDS[2] = { 1080, 1920 };
		if ((x_Bound + xShift > SIZE_BOUNDS[0]) || (y_Bound + yShift > SIZE_BOUNDS[1]))
		{
			result.status = false;
			cout << "X Bound: " << x_Bound << "  Y Bound: " << y_Bound << endl;
			cout << "ERROR: Image Too Large" << endl;
			
			return;
		}
		
		//Apply transformation and reshape images appropriately, this is now performed on the GPU
		GpuMat imageB2;
		cuda::copyMakeBorder(imageB, imageB2, yShift, 0, xShift, 0, BORDER_CONSTANT, Scalar(0, 0, 0));
		cuda::warpPerspective(imageA, result.enlargedImageA, shiftM * result.H, Size(x_Bound + xShift, y_Bound + yShift));
		cuda::copyMakeBorder(imageB2, result.enlargedImageB, 0, y_Bound + yShift - imageB2.size().height, 0, x_Bound + xShift - imageB2.size().width, BORDER_CONSTANT, Scalar(0, 0, 0));

	}

	/**
	* @param maskA the nonzero mask of StitchingResult.enlargedImageA
	* @param maskB the nonzero mask of StitchingResult.enlargedImageB
	* @param outMask output mask whose size is the same as maskA and maskB
	*/
	void locateSeam(Mat maskA, Mat maskB, Mat& outMask)
	{
		// tested
		if (maskA.type() != CV_8UC1)
		{
			cvtColor(maskA, maskA, CV_RGB2GRAY);
		}
		if (maskB.type() != CV_8UC1)
		{
			cvtColor(maskB, maskB, CV_RGB2GRAY);
		}
		outMask = Mat::zeros(maskA.size(), CV_8UC1);
		vector< vector<Point> > contours;
		findContours(maskA, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_NONE);
		drawContours(outMask, contours, -1, Scalar(255), 1);
		outMask = outMask.mul(maskB);
	}

	void calcMinDistance(vector<Point> ptsA, Mat maskB, vector<double> output)
	{
		// not tested yet
		vector<Point> ptsB;
		nonzeros(maskB, ptsB);

		for (vector<Point>::iterator itrA = ptsA.begin(); itrA != ptsA.end(); itrA++)
		{
			Point A = *itrA;
			double minDistance = 9999999999999;
			for (vector<Point>::iterator itrB = ptsB.begin(); itrB != ptsB.end(); itrB++)
			{
				Point B = *itrB;
				double dist = sqrt(pow(A.x - B.x, 2) + pow(A.y - B.y, 2));
				if (minDistance > dist)
					minDistance = dist;
			}
			output.push_back(minDistance);
		}
	}

	void downloadResults(GPUResult& result1, StitchingResult& result2){
		result2.pano = result1.pano;
		result2.H = Mat(result1.H);
		result2.goodMatches = result1.goodMatches;
		result2.drawMatches = result1.drawMatches;
		result2.enlargedImageA =  Mat(result1.enlargedImageA);
		result2.enlargedImageB =  Mat(result1.enlargedImageB);
		result2.shift = result1.shift;
	}

	void uploadResults(StitchingResult& result1, GPUResult& result2){
		result2.pano = result1.pano;
		result2.H = result1.H;
		result2.goodMatches = result1.goodMatches;
		result2.drawMatches = result1.drawMatches;
		result2.enlargedImageA.upload(result1.enlargedImageA);
		result2.enlargedImageB.upload(result1.enlargedImageB);
		result2.shift = result1.shift;
	}

	void displayResults(GPUResult result){
		imshow("display", Mat(result.pano));
		//imshow("imageA",Mat(result.enlargedImageA));
		//imshow("imageB",Mat(result.enlargedImageB));
	}

	/**
	* @param imageA StitchingResult.enlargedImageA
	* @param imageB StitchingResult.enlargedImageB
	* @param canvas StitchingResult.pano with larger boundary
	* @param fgbg Background Subtractor
	* @param seam mask computed by locateSeam function
	* @param outPos a point whose x is the width of the enlarged left boundary and y is the height of the enlarged top boundary
	*/



private:
	int abs(int a)
	{
		if (a > 0)
			return a;
		else
			return -a;
	}
	int findMax(int A, int B)
	{
		if (A > B)
		{
			return A;
		}
		return B;
	}
	int findMin(int A, int B)
	{
		if (A < B)
		{
			return A;
		}
		return B;
	}

	void nonzeros(Mat src, vector<Point>& pts)
	{
		if (src.type() != CV_8UC1)
		{
			cvtColor(src, src, CV_RGB2GRAY);
		}

		int rowNumber = src.size().height;
		int colNumber = src.size().width;
		for (int i = 0; i < rowNumber; i++)
		{
			uchar* data = src.ptr<uchar>(i);
			for (int j = 0; j < colNumber; j++)
			{
				if (data[j] != 0)
					pts.push_back(Point(j, i));
			}
		}

	}
};

/*
int main(int argc, char* argv[]){
		return 1;
}*/ 