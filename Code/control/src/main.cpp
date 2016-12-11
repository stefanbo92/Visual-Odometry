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

//destination trajectory in mm
double destinationsX [4] ={0,1000,1000,0};
double destinationsY [4] ={1000,1000,0,0};
//maximum speed of the robotino
double SPEED=0.2,speed;

double xr,yr,angle,xw,yw;
int k=0;
	

class Controller
{
	public:
	Controller()
	{
		//initialise pose subscriber and velocity publisher (for robotino_node)
		 chatter_pub=n.advertise<geometry_msgs::Twist>("/cmd_vel", 100);  
		 sub = n.subscribe("/cmd_pos", 1000, &Controller::chatterCallback,this);
	}

	void chatterCallback(const geometry_msgs::Pose::ConstPtr& msg)
	{
		//update current destination
		xw=destinationsX[k];
		yw=destinationsY[k];

		//rotate to the right if desired
        if(msg->position.y==1){
			geometry_msgs::Twist cmd_vel_msg;
			cmd_vel_msg.angular.z=-0.2;
			chatter_pub.publish(cmd_vel_msg);
		}
		//rotate to the left if desired
        else if (msg->position.y==2){
            geometry_msgs::Twist cmd_vel_msg;
            cmd_vel_msg.angular.z=0.2;
            chatter_pub.publish(cmd_vel_msg);
        }
		//otherwise move to the destination
		else {
			//load current pose of the robot
			xr=msg->position.x;
			yr=msg->position.z;
            angle=-msg->orientation.y;

			//calculate distance from robot to destination
			double dist=fabs(xw-xr)+fabs(yw-yr);

			//introduce some variables
            double xwn,ywn,xvel,yvel,xvelout,yvelout;
			geometry_msgs::Twist cmd_vel_msg;

			//rotate the world coorinate system to the robots coordiante system
			xwn=cos(angle)*(xw-xr)+sin(angle)*(yw-yr);
			ywn=-sin(angle)*(xw-xr)+cos(angle)*(yw-yr);

			//adapt coordinate systems
			xvel=ywn;
			yvel=-xwn;


            cmd_vel_msg.angular.z=0;
			//check if the destination is close or already reached
            if ((dist<=50)&&(k==3)) {
				speed=0;
			}
			//move to the next point of the given trajectory 
            else if ((dist>50)&&(dist<150)) {
                speed=0.10;
				if(k!=3){
                    k++;
				}
			}
			else {
				speed=SPEED;
			}

			//scale the velocity to maximum
			if (fabs(xvel)>=fabs(yvel)){
			xvelout=xvel*speed/fabs(xvel);
			yvelout=yvel*speed/fabs(xvel);
			}
			else {
			xvelout=xvel*speed/fabs(yvel);
			yvelout=yvel*speed/fabs(yvel);
			}

			//rotate while moving for dynamic feature search
			if(msg->position.y==3){
			cmd_vel_msg.angular.z=-0.2;
			}

			//output information
			cout<<"x velocity: "<<xvelout<<endl;
			cout<<"y velocity: "<<yvelout<<endl;
			cout<<"x POS: "<<xr<<endl;
			cout<<"y POS: "<<yr<<endl;
			cout<<"Theta: "<<angle*180/M_PI<<endl;
			cout<<"Distance: "<<dist<<endl;
			cout<<"___________________________________"<<endl;

			//publish velocity commands
			cmd_vel_msg.linear.x=xvelout;
			cmd_vel_msg.linear.y=yvelout;

			chatter_pub.publish(cmd_vel_msg);
		}
	}

	private:
	  ros::NodeHandle n; 
	  ros::Publisher chatter_pub;
	  ros::Subscriber sub;
	};



int main(int argc, char** argv)
{
	ros::init(argc, argv, "control");
	Controller controller;
	ros::spin();
	return 0;
}

