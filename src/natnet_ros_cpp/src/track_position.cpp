#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Pose.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "ros/ros.h"
#include <Eigen/Dense>

class PositionInterface
{
    private:
        ros::NodeHandle n;
        ros::Timer timer;
        ros::Publisher des_pos_pub;

        std::string control_interface;

        Eigen::Matrix<double, 3, 3> matA;
        Eigen::Matrix<double, 3, 1> matB;

        Eigen::Vector3d curr_pos;

        std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};
        std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
        
    
    public:
        PositionInterface(ros::NodeHandle nh, std::string control_interface):
        timer(nh.createTimer(ros::Duration(0.01), &PositionInterface::main_loop, this)),
        control_interface(control_interface)
        {

            des_pos_pub = nh.advertise<geometry_msgs::Pose>("obs_pos", 10);
            
            tf_buffer_ = std::make_unique<tf2_ros::Buffer>();
            transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);


        }

        void publishDesiredOptitrackPose(){
            geometry_msgs::TransformStamped transformStamped;
            try{
                transformStamped = tf_buffer_->lookupTransform("franka", "object",
                                        ros::Time(0));
            }
            catch (tf2::TransformException &ex) {
                ROS_WARN("%s",ex.what());
                ros::Duration(1.0).sleep();
            }
            // std::cout << " " << std::endl;
            // std::cout << "x: " << -transformStamped.transform.translation.y << std::endl;
            // std::cout << "y: " << transformStamped.transform.translation.x << std::endl;
            // std::cout << "z: " << transformStamped.transform.translation.z << std::endl;

            geometry_msgs::Pose pose;
            pose.position.x = -transformStamped.transform.translation.y;
            pose.position.y = transformStamped.transform.translation.x;
            pose.position.z = transformStamped.transform.translation.z;

            pose.orientation.x = 0.70710678118;
            pose.orientation.y = 0.0;
            pose.orientation.z = 0.70710678118;
            pose.orientation.w = 0.0;
            des_pos_pub.publish(pose);


            //curr_pos << (double)pose.position.x, (double)pose.position.y, (double)pose.position.z;
        }


        void main_loop(const ros::TimerEvent &){


            publishDesiredOptitrackPose();
            
        }
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "PositionInterface");
    ros::NodeHandle n;

    std::string control_interface = "position";
    n.getParam("control_interface", control_interface);

    PositionInterface ds = PositionInterface(n, control_interface);
    ros::spin();
    return 0;
}