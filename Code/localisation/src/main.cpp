#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include "std_msgs/String.h"
#include "opencv2/opencv.hpp"
#include <sstream>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>

using namespace cv;
using namespace std;

typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, sensor_msgs::Image
  > MySyncPolicy;

    

	//initialise variables which are required for the algorithm
	Mat imgLeft,imgRight,imgLeftc,imgRightc, imgLeftNew,imgLeftOld,imgRightOld;
	TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
	vector<float> err;
	Size winSize(31,31);
	vector<uchar> statusLKT;
	vector<KeyPoint> keypointsLeft, keypointsRight,keypointsLeftNew, goodKeypointsStereoLeft, goodKeypointsStereoRight;
	Mat descriptorsLeft, descriptorsRight,descriptorsLeftNew,goodDescriptorsStereo;
    	OrbDescriptorExtractor extractor(2000);
	std::vector< DMatch > matches;
	double avrgTime,tick;
	std::vector<Point2f>  goodPointsNew, goodPointsTriLeft, goodPointsTriRight;
	Mat K1,D1,P1,P2,HandEye,RTnew,Rnew,Rtotal,Ttotal,intrinsics,distortion,rvec,tvec;
 	Mat RTtotal=(Mat_<double>(4,4) <<1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);
    	Mat tvecSaved=(Mat_<double>(3,1) <<0,0,0);
    	int initFeatures=0;
	int maxFeat=0;
	double bestAngle=0;

	//function for calculating corresponding feature points
	void calcFeatureSets (vector<Point2f>  &goodPointsNew, vector<Point2f> &goodPointsTriLeft, vector<Point2f> &goodPointsTriRight) {
		//clear all old datasets
		goodPointsNew.clear();
		goodPointsTriLeft.clear();
		goodPointsTriRight.clear();
		keypointsLeftNew.clear();
		goodKeypointsStereoLeft.clear();
		goodKeypointsStereoRight.clear();
		keypointsLeft.clear();
		keypointsRight.clear();

		//detect FAST features in left and right image
		extractor.detect(imgLeftOld, keypointsLeft);
		extractor.detect(imgRightOld, keypointsRight);
		//extract ORB descripors
		extractor.compute(imgLeftOld, keypointsLeft, descriptorsLeft);
		extractor.compute(imgRightOld, keypointsRight, descriptorsRight);
        	//cout<<"Features detected: "<<keypointsLeft.size()<<endl;

		//Match feature descriptors with brute force matcher and cross check (using hamming distance)
        	BFMatcher matcher=BFMatcher(NORM_HAMMING, true);
	  	matches.clear();
	  	matcher.match( descriptorsLeft, descriptorsRight, matches );

		/* 
		//Ratio Test
		vector<vector< DMatch> > twoMatches;
		matcher.knnMatch(descriptorsLeft,descriptorsRight,twoMatches, 2);
		for (int i=0;i<twoMatches.size();i++){
			if(twoMatches[i][0].distance < 0.9*twoMatches[i][1].distance){
				matches.push_back(twoMatches[i][0]);
				}
		}
        	cout<<"Matches Stereo: "<<matches.size()<<endl;
		*/
	

		//remove bad stereo matches that are not on the same y-coordinate
	  	std::vector< DMatch > good_matches_stereo, veryGood;
		for( int i = 0; i < matches.size(); i++ )
		  { 
			if (fabs(keypointsLeft[matches[i].queryIdx].pt.y-keypointsRight[matches[i].trainIdx].pt.y)<=2){
		      		good_matches_stereo.push_back( matches[i]);
		 	}
		}


		//remove matches with a distance higher than a threshol
		for( int i = 0; i < good_matches_stereo.size(); i++ ){ 
			if (good_matches_stereo[i].distance<40 ){ 
					veryGood.push_back(good_matches_stereo[i]);
			}
		}

		good_matches_stereo=veryGood;
        	//cout<<"Very Good Matches: "<<veryGood.size()<<endl;

		for( int i = 0; i < good_matches_stereo.size(); i++ ){  
		    goodKeypointsStereoLeft.push_back(keypointsLeft[good_matches_stereo[i].queryIdx]);
		    goodKeypointsStereoRight.push_back(keypointsRight[good_matches_stereo[i].trainIdx]);
		}

		vector<Point2f> goodPointsStereoLeft,goodPointsStereoRight,pointsNew;
		for (int i=0;i<goodKeypointsStereoLeft.size();i++){
			goodPointsStereoLeft.push_back(goodKeypointsStereoLeft[i].pt);
			goodPointsStereoRight.push_back(goodKeypointsStereoRight[i].pt);
			}


		//remove outliers with RANSAC
		Mat statusStereo;
        	findFundamentalMat(goodPointsStereoLeft, goodPointsStereoRight, CV_FM_RANSAC, 3, 0.999, statusStereo);

		int stereoInliers=0;
		vector <KeyPoint> left,right;
		for( int i = 0; i < statusStereo.rows; i++ )
		  {
			if(statusStereo.at<bool>(0,i)){
				left.push_back(goodKeypointsStereoLeft[i]);
		    	right.push_back(goodKeypointsStereoRight[i]);
				stereoInliers++;
			}
		  }
		//cout<<"Stereo Inliers: "<<stereoInliers<<endl;
		goodKeypointsStereoLeft=left;
		goodKeypointsStereoRight=right; 
        	goodPointsStereoLeft.clear(); goodPointsStereoRight.clear();
		for (int i=0;i<goodKeypointsStereoLeft.size();i++){
			goodPointsStereoLeft.push_back(goodKeypointsStereoLeft[i].pt);
			goodPointsStereoRight.push_back(goodKeypointsStereoRight[i].pt);
			}


		//track features in current frame based of features in old farme
		calcOpticalFlowPyrLK(imgLeftOld, imgLeft, goodPointsStereoLeft, pointsNew, statusLKT, err, winSize, 3, termcrit, 0, 0.001);

		//generate sets of corresponding keypoints of old stereo images (for triangulation) and new left image
		int count=0;
		for( int i = 0; i < statusLKT.size(); i++ )
		  {
			if(statusLKT[i]){
				goodPointsNew.push_back( pointsNew[i] );
				goodPointsTriLeft.push_back(goodKeypointsStereoLeft[i].pt);
				goodPointsTriRight.push_back(goodKeypointsStereoRight[i].pt);
				count++;
			}
		  }
		  cout<<"matched features: "<<count<<endl;
	}



	//triangulate 3D points with the linear method
	Mat Triangulation(Point2f u, Mat P, Point2f u1, Mat P1)
	{
	    Mat A=(Mat_<double>(4,3)<<u.x*P.at<double>(2,0)-P.at<double>(0,0),    u.x*P.at<double>(2,1)-P.at<double>(0,1),      u.x*P.at<double>(2,2)-P.at<double>(0,2),
		  u.y*P.at<double>(2,0)-P.at<double>(1,0),    u.y*P.at<double>(2,1)-P.at<double>(1,1),      u.y*P.at<double>(2,2)-P.at<double>(1,2),
		  u1.x*P1.at<double>(2,0)-P1.at<double>(0,0), u1.x*P1.at<double>(2,1)-P1.at<double>(0,1),   u1.x*P1.at<double>(2,2)-P1.at<double>(0,2),
		  u1.y*P1.at<double>(2,0)-P1.at<double>(1,0), u1.y*P1.at<double>(2,1)-P1.at<double>(1,1),   u1.y*P1.at<double>(2,2)-P1.at<double>(1,2)
		      );
	    Mat B = (Mat_<double>(4,1) <<    -(u.x*P.at<double>(2,3)    -P.at<double>(0,3)),
		              -(u.y*P.at<double>(2,3)  -P.at<double>(1,3)),
		              -(u1.x*P1.at<double>(2,3)    -P1.at<double>(0,3)),
		              -(u1.y*P1.at<double>(2,3)    -P1.at<double>(1,3)));
	 
	    Mat X;
	    solve(A,B,X,DECOMP_SVD);
	 
	    return X;
	}


	//ROS class which receives the rectified images and publishes the pose of the robot
	class ImageConverter
	{
		public:
			ImageConverter();
			void InitializePubSub();
		private:
			ros::NodeHandle nh_;
			image_transport::ImageTransport it_;
			//image subscriber
			image_transport::SubscriberFilter left_image_sub_;
			image_transport::SubscriberFilter right_image_sub_;
			message_filters::Synchronizer< MySyncPolicy > sync_;
			//pose Publisher
			ros::Publisher chatter_pub=nh_.advertise<geometry_msgs::Pose>("/cmd_pos", 100);  
			//Subscriber Callback method
			void stereoCallback(
					const sensor_msgs::ImageConstPtr& left_image_msg,
					const sensor_msgs::ImageConstPtr& right_image_msg);
	};

	void ImageConverter::InitializePubSub() {
		sync_.registerCallback( boost::bind( &ImageConverter::stereoCallback, this, _1, _2) );
	}

	//for implementation on the apollon notebook, subscribe the following: left_image_sub_(it_, "/apollon/vrmagic/left/image_rect_color", 1), right_image_sub_(it_, "/apollon/vrmagic/right/image_rect_color", 1)
	ImageConverter::ImageConverter() :
			it_(nh_), left_image_sub_(it_, "/vrmagic/left/image_rect_color", 1), right_image_sub_(it_, "/vrmagic/right/image_rect_color", 1),
					sync_(MySyncPolicy(10), left_image_sub_, right_image_sub_) {
		InitializePubSub();
	}

	//converting ROS image Message to OpenCV Mat with CVBridge
	void ImageConverter::stereoCallback(const sensor_msgs::ImageConstPtr& msg1,const sensor_msgs::ImageConstPtr& msg2)
	  {
		cv_bridge::CvImagePtr cv_ptr1,cv_ptr2;
		try
		{
		  cv_ptr1=cv_bridge::toCvCopy(msg1,"bgr8");
		  cv_ptr2=cv_bridge::toCvCopy(msg2,"bgr8");
		}
		catch (cv_bridge::Exception& e)
		{
		  ROS_ERROR("cv_bridge exception: %s", e.what());
		  return;
		}

	//VO ALGORITHM:
	//start clock and load images
    	tick = (double)getTickCount(); 
	imgLeftc=cv_ptr1->image;
	imgRightc=cv_ptr2->image;

	//convert images to grayscale
	cvtColor(imgLeftc, imgLeft, COLOR_BGR2GRAY);
	cvtColor(imgRightc, imgRight, COLOR_BGR2GRAY);

	if(imgLeftOld.empty()){
        imgLeft.copyTo(imgLeftOld);
		imgRight.copyTo(imgRightOld);
		}
	
	//detect and match features in the previous left and right image (for triangulation) and track those features in the new left image
	calcFeatureSets (goodPointsNew, goodPointsTriLeft, goodPointsTriRight);
	
	//triangulate goodPointsTriLeft and goodPointsTriRight to get a set of 3D worldPoints
	if(goodPointsTriLeft.size()>5){
	std::vector<Point3f>  worldPointsHart;
	for (int i=0;i<goodPointsTriLeft.size();i++){
		Mat world=Triangulation(goodPointsTriLeft.at(i),P1,goodPointsTriRight.at(i),P2);
		worldPointsHart.push_back(Point3f(world.at<double>(0,0), world.at<double>(1,0), world.at<double>(2,0)));
	}
	
	//Solve PnP with RANSAC
	Mat inliers;
    	solvePnPRansac(worldPointsHart, goodPointsNew, K1, D1, rvec, tvec, false,1000,2.0,-1,inliers,CV_P3P);
	tvec=-tvec;
    	rvec=-rvec;
	//print out current motion
	cout<<"R: "<<rvec*180/3.14<<endl;
	cout<<"T: "<<tvec<<endl;

	/*
	//calculate the inliers of the PnP algorithm
	int inlier=0;
	for( int i = 0; i < inliers.rows; i++ )
	  {
		if(inliers.at<bool>(0,i)){
			inlier++;
		}
	  }
	cout<<"PnP Inliers: "<<inlier<<endl;
	*/

	//remove very small or large rotations and translations
	rvec.at<double>(0)=0;
    	rvec.at<double>(2)=0;
    	tvec.at<double>(1)=0;
	for (int i=0;i<3;i++){
        if((fabs(rvec.at<double>(i))<0.2/180*3.14)||(fabs(rvec.at<double>(i))>1.5)){
			rvec.at<double>(i)=0;
		}
	}
    	for (int i=0;i<3;i++){
		if((fabs(tvec.at<double>(i))<1)||(fabs(tvec.at<double>(i))>300)){
			tvec.at<double>(i)=0;
		}
	}


	//Calculate total pose
	Rodrigues(rvec,Rnew);

	RTnew=(Mat_<double>(4,4) <<Rnew.at<double>(0,0),Rnew.at<double>(0,1),Rnew.at<double>(0,2),tvec.at<double>(0),Rnew.at<double>(1,0),Rnew.at<double>(1,1),Rnew.at<double>(1,2),tvec.at<double>(1),Rnew.at<double>(2,0),Rnew.at<double>(2,1),Rnew.at<double>(2,2),tvec.at<double>(2),0,0,0,1);

	//Calculating the robots pose from the cameras pose (Hand-Eye-Relation)
    	RTnew=HandEye*RTnew*HandEye.inv();
	RTtotal=RTtotal*RTnew;

	//print out total pose
	Rtotal=(Mat_<double>(3,3) <<RTtotal.at<double>(0,0),RTtotal.at<double>(0,1),RTtotal.at<double>(0,2),RTtotal.at<double>(1,0),RTtotal.at<double>(1,1),RTtotal.at<double>(1,2),RTtotal.at<double>(2,0),RTtotal.at<double>(2,1),RTtotal.at<double>(2,2));
	Ttotal=(Mat_<double>(3,1) <<RTtotal.at<double>(0,3),RTtotal.at<double>(1,3),RTtotal.at<double>(2,3));
	Rodrigues(Rtotal, rvec);

	cout<<"R Total: "<<rvec*180/3.14<<endl;
	cout<<"T Total: "<<Ttotal<<endl;	


	//VISUALISATION:
	//draw lines for moving Keypoints that moved more than a threshold
	for( int i=0; i < goodPointsNew.size(); i++ ){
		Point p0( ceil( goodPointsNew[i].x ), ceil( goodPointsNew[i].y ) );
		Point p1( ceil( goodPointsTriLeft[i].x ), ceil( goodPointsTriLeft[i].y ) );

		double res = cv::norm(p1-p0);
		if (res>5){
				line( imgLeftc, p0, p1, CV_RGB(0,255,0), 2 );
		}
	}

	//calculate colour according to depth of the 3D point
	vector<float> depth;
	float maxDepth=0;	
	float minDepth=10000;
	for (int i=0;i<worldPointsHart.size();i++){
		if(worldPointsHart.at(i).z>0){
			if(worldPointsHart.at(i).z>maxDepth){
				maxDepth=worldPointsHart.at(i).z;
			}
			if(worldPointsHart.at(i).z<minDepth){
				minDepth=worldPointsHart.at(i).z;
			}
		}
	}
	for (int i=0;i<goodPointsTriLeft.size();i++){
		depth.push_back((worldPointsHart.at(i).z-minDepth)/(maxDepth-minDepth)*255);
	}

	//draw feature points and depth information
	for (int i=0;i<goodPointsTriLeft.size();i++){
		Point p0( ceil( goodPointsTriLeft[i].x ), ceil( goodPointsTriLeft[i].y ) );
		Point p1( ceil( goodPointsTriRight[i].x ), ceil( goodPointsTriRight[i].y ) );
		Scalar colorDepth = Scalar(0,depth[i],255);
		circle(imgLeftc,p0,2,Scalar(0,255,0),2);
		circle(imgRightc,p1,2,colorDepth,2);
		}
	}


	//Output Images and copy current images to old images
	imshow( "Input1", imgLeftc );
    	imshow( "Input2", imgRightc );

	//save the current image frame for the next calculation step
	imgLeft.copyTo(imgLeftOld);
	imgRight.copyTo(imgRightOld);

	//publish current pose
	geometry_msgs::Pose cmd_pos_msg;
	cmd_pos_msg.position.x=Ttotal.at<double>(0,0);
	cmd_pos_msg.position.y=0;
	cmd_pos_msg.position.z=Ttotal.at<double>(2,0);
	cmd_pos_msg.orientation.x=rvec.at<double>(0,0);
	cmd_pos_msg.orientation.y=rvec.at<double>(1,0);
	cmd_pos_msg.orientation.z=rvec.at<double>(2,0);	


    //STATIC FEATURE SEARCH
	//initiate feature search if there are too less features
    if ((initFeatures==3)&&(goodPointsNew.size()<30)){
        tvecSaved=Ttotal;
        initFeatures=0;
        maxFeat=0;
    }

	//turn to the left, searching for most features
    if (initFeatures==0){
        cmd_pos_msg.position.y=2;
		if(maxFeat<goodPointsNew.size()){
			maxFeat=goodPointsNew.size();
            bestAngle=rvec.at<double>(1);
		}	
        if ((rvec.at<double>(1)<-1.5)||(goodPointsNew.size()<60)){
            initFeatures=1;
		}
	}

	//turn to the right, searching for most features
    if (initFeatures==1){
        cmd_pos_msg.position.y=1;
        if(maxFeat<goodPointsNew.size()){
            maxFeat=goodPointsNew.size();
            bestAngle=rvec.at<double>(1);
        }
        if ((rvec.at<double>(1)>1.5)||((goodPointsNew.size()<60)&&(rvec.at<double>(1)>0))){
            initFeatures=2;
            cout<<"Feature search done! Best angle: "<<bestAngle*180/3.14<<" with "<<maxFeat<<" features."<<endl;
            waitKey(2000);
        }
    }

	//turn to the direction with the most features
    if (initFeatures==2){
        cmd_pos_msg.position.y=2;
        if (rvec.at<double>(1)<=bestAngle){
            initFeatures=3;
            cmd_pos_msg.position.y=0;
            RTtotal.at<double>(0,3)=tvecSaved.at<double>(0);
            RTtotal.at<double>(1,3)=tvecSaved.at<double>(1);
            RTtotal.at<double>(2,3)=tvecSaved.at<double>(2);
            cout<<"Best angle reached, starting to move!"<<endl;
            waitKey(2000);
        }
    }


/*
    //DYNAMIC FEATURE SEARCH
    if(goodPointsNew.size()<100){
		cmd_pos_msg.position.y=3;
	}
	else {
		cmd_pos_msg.position.y=0;
	}
*/

    //wait 1ms
    char k=waitKey(1);
    if(k == 'a'){k=waitKey(20000); } 

    chatter_pub.publish(cmd_pos_msg);

    //calculate time
    avrgTime = ((double)getTickCount() - tick) / getTickFrequency();
    cout<<"Time: "<<avrgTime*1000<<"ms"<<endl;
    cout<<"__________________________________"<<endl;
}






int main(int argc, char** argv)
{
	//OpenCV Initialisation
	namedWindow( "Input1", 1 );
    namedWindow( "Input2", 1 );
	//load calibration data
	FileStorage fs("calib.xml", FileStorage::READ);
	fs["K1"] >> K1;
	fs["D1"] >> D1;
	fs["P1"] >> P1;
	fs["P2"] >> P2;
	fs["HandEye"] >> HandEye;
	fs.release();

	//ROS
	ros::init(argc, argv, "localisation");
	ImageConverter ic;
	ros::spin();
	return 0;
}

