//#include "opencv2/core/core.hpp"
#include "stitcher.cpp"
#include "opencv2/imgproc.hpp"
Mat radial_dst_mat = (Mat_<double>(1,5) << -0.38368541,  0.17835109, -0.004914,    0.00220994, -0.04459628 );
InputArray radial_dst(radial_dst_mat);
Mat mtx =  (Mat_<double>(3,3) << 444.64628787, 0., 309.40196271, 0., 501.63984347, 255.86111216, 0., 0., 1.);

void fullstitch(Mat img1, Mat img2, GPUResult& result1, ImageStitcher& stitch){
		//Extra variables used in the function. Should maybe be declared as public outside.
		Mat fimg1, fimg2;
		GpuMat gpuimg1,gpuimg2;
		vector<GpuMat> images;

		//remove barrel distortion using pre calibrated values
                //cout << "Removing barrel distortion: ";
                //stitch.tic();
                undistort(img1,fimg1, mtx, radial_dst );
                undistort(img2,fimg2, mtx, radial_dst );
                //stitch.toc();
		
		cout << "Converting to grayscale: ";
		stitch.tic();
                cvtColor(fimg1,fimg1, CV_BGR2GRAY);
                cvtColor(fimg2, fimg2, CV_BGR2GRAY);
		stitch.toc();

		cout << "Uploading image to GPU: ";
		stitch.tic();
		gpuimg1.upload(fimg1);
		CV_Assert(!gpuimg1.empty());
		gpuimg2.upload(fimg2);
		CV_Assert(!gpuimg2.empty());
		stitch.toc();

		images.push_back(gpuimg1);
		images.push_back(gpuimg2);

		stitch.stitch(images, result1);	

}

int main(int argc, char* argv[]){

	//Declare needed variables.
	bool status;
	Mat img1, img2;
	ImageStitcher stitch;
	GPUResult result1;

	// Set up video feed.
        VideoCapture vid2("http://10.42.0.105:8050/?action=stream");
        VideoCapture vid1("http://10.42.0.124:8070/?action=stream");
        //VideoCapture vid1("output1.avi");
        //VideoCapture vid2("output2.avi");
	//Read incoming frame
	status = vid1.read(img1);
	vid2.read(img2);

	int cc = 0;
	while(status){
		cc = cc+1;
		cout << endl << endl;
		cout << "Stitching Frame " << cc << endl;
		//Perform stitching between images
		fullstitch(img1,img2, result1, stitch);

		//Display stitched image
		stitch.displayResults(result1);
		if (waitKey(10) == 'q')
			break; 

		//Read in new frames.
		vid1.read(img1);
		vid2.read(img2);

	}
}
