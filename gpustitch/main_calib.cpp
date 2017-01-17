//#include "opencv2/core/core.hpp"
#include "stitcher.cpp"
#include "opencv2/imgproc.hpp"
Mat radial_dst_mat = (Mat_<double>(1,5) << -0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628 );
InputArray radial_dst(radial_dst_mat);
Mat mtx =  (Mat_<double>(3,3) << 444.64628787, 0., 309.40196271, 0., 501.63984347, 255.86111216, 0., 0., 1.);

int main(int argc, char* argv[]){
	VideoCapture vid1("output2.avi");
	VideoCapture vid2("output1.avi");
	bool status;
	Mat img1, img2, fimg1,fimg2;
	GpuMat gpuimg1,gpuimg2;
	vector<GpuMat> images;
	Mat H;
	double timer=0;

	ImageStitcher stitch;
	GPUResult result1;
	StitchingResult result;

	status = vid1.read(img1);
	vid2.read(img2);

	//Calibrate transformation
	if(status){
		undistort(img1,fimg1, mtx, radial_dst );
		undistort(img2,fimg2, mtx, radial_dst );

		cvtColor(fimg1,fimg1, CV_BGR2GRAY);
		cvtColor(fimg2, fimg2, CV_BGR2GRAY);

		gpuimg1.upload(fimg1);
		CV_Assert(!gpuimg1.empty());
		gpuimg2.upload(fimg2);
		CV_Assert(!gpuimg2.empty());

		images.push_back(gpuimg1);
		images.push_back(gpuimg2);

		stitch.stitch(images, result1);
		H = result1.H;

		images.pop_back();
		images.pop_back();

		vid1.read(img1);
		vid2.read(img2);

	}

	VideoWriter vid_out;
	vid_out.open("result.avi", vid1.get(CV_CAP_PROP_FOURCC),vid1.get(CV_CAP_PROP_FPS),result1.pano.size());
	cout << "SIZE: " << result1.pano.size() << endl;

	while(status){

		timer = getTickCount();
		undistort(img1,fimg1, mtx, radial_dst );
		undistort(img2,fimg2, mtx, radial_dst );

		cvtColor(fimg1,fimg1, CV_BGR2GRAY);
		cvtColor(fimg2, fimg2, CV_BGR2GRAY);

		gpuimg1.upload(fimg1);
		CV_Assert(!gpuimg1.empty());
		gpuimg2.upload(fimg2);
		CV_Assert(!gpuimg2.empty());


		stitch.applyHomography(gpuimg1,gpuimg2, result1);
		stitch.stitchImage(result1);	
		//if result1.enlargedImageB.size(){
		//imshow("imageA",.5*(result.enlargedImageA + result.enlargedImageB));
		//imshow("image1", Mat(gpuimg1));
		//imshow("image2", Mat(gpuimg2));
		stitch.displayResults(result1);
		if (waitKey(10) == 'q')
			break; 

		//}

		vid1.read(img1);
		vid2.read(img2);


		cout << "Frame computation time" << (getTickCount() - timer)/(getTickFrequency()) << endl;
		vid_out.write(Mat(result1.pano));
	}
	vid_out.release();
}