#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/transforms.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Empty.h>

using namespace std;

string filepath;
bool stop_triggle = false;

void triggerCallback(const std_msgs::Empty& msg) {
  stop_triggle = true;
  cout << "Triggered!" << endl;
}

int main(int argc, char** argv) {
  // Initialize ROS
  ros::init(argc, argv, "map_publisher");
  ros::NodeHandle nh("~");

  // Create a publisher for the PointCloud2 message
  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("point_cloud", 1);
  // ros::Subscriber trigger_sub_ = nh.subscribe("/waypoint_generator/waypoints", 1, triggerCallback);
  ros::Subscriber trigger_sub_ = nh.subscribe("/planning/triggle_map", 1, triggerCallback);
  nh.param<string>("map_publisher/filepath", filepath, "");

  // Read the PCD file
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  pcl::PLYReader reader;
  if (reader.read<pcl::PointXYZ>(filepath, *cloud) < 0) {
    ROS_ERROR("Failed to read PLY file");
    return -1;
  }
  Eigen::Vector3d mmin(0, 0, 0), mmax(0, 0, 0);
  for (const auto& pt : *cloud) {
    mmin[0] = min(mmin[0], double(pt.x));
    mmin[1] = min(mmin[1], double(pt.y));
    mmin[2] = min(mmin[2], double(pt.z));
    mmax[0] = max(mmax[0], double(pt.x));
    mmax[1] = max(mmax[1], double(pt.y));
    mmax[2] = max(mmax[2], double(pt.z));
  }
  ROS_WARN("[map_publisher] load map: %s \nmap size: min(%.4f %.4f %.4f) max(%.4f %.4f %.4f)", filepath.c_str(), mmin[0], mmin[1],
      mmin[2], mmax[0], mmax[1], mmax[2]);

  // Convert the PCL point cloud to a ROS message
  sensor_msgs::PointCloud2 msg;
  pcl::toROSMsg(*cloud, msg);
  msg.header.frame_id = "world";  // Set the frame ID
  // Publish the ROS message
  ros::Rate rate(5);
  while (ros::ok()) {
    pub.publish(msg);
    ros::spinOnce();
    if (stop_triggle) break;
    rate.sleep();
  }
  ROS_WARN("[map_publisher] Map publisher finish!!!!!!!");
  return 0;
}