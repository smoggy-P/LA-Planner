#ifndef FEATURE_MAP_H
#define FEATURE_MAP_H

#include "plan_env/raycast.h"

#include "utils/utils.h"

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#include <Eigen/Eigen>

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/centroid.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <pcl_ros/transforms.h>
#include <tf2_eigen/tf2_eigen.h>

using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::vector;

using namespace Eigen;

namespace perception_aware_planner {

class SDFMap;
class FeatureMap {
public:
  using Ptr = shared_ptr<FeatureMap>;
  using ConstPtr = shared_ptr<const FeatureMap>;

  shared_ptr<SDFMap> sdf_map_ = nullptr;
  shared_ptr<SDFMap> global_sdf_map_ = nullptr;
  CameraParam::Ptr feature_cam_ = nullptr;

  void setMap(std::shared_ptr<SDFMap>& global_map, std::shared_ptr<SDFMap>& map);
  void initMap(ros::NodeHandle& nh);
  void loadMap(const std::string& filename);

  void getFeatureCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
  Vector3d getFeatureByID(const int id);

  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
  void sensorPoseCallback(const geometry_msgs::PoseStampedConstPtr& pose);
  void visFeatureMap();

  void pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg);
  void updateFeatureMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr& new_features);
  void reset();

  int getFeatureUsingCamPosOrient(const Vector3d& pos, const Quaterniond& orient, vector<pair<int, Vector3d>>& res);
  int getFeatureUsingCamPosOrient(const Vector3d& pos, const Quaterniond& orient, set<int>& res) {
    res.clear();
    vector<pair<int, Vector3d>> res_pair;
    int f_num = getFeatureUsingCamPosOrient(pos, orient, res_pair);
    for (const auto& pair : res_pair) res.insert(pair.first);

    return f_num;
  }
  int getFeatureUsingCamPosOrient(const Vector3d& pos, const Quaterniond& orient, vector<Vector3d>& res) {
    res.clear();
    vector<pair<int, Vector3d>> res_pair;
    int f_num = getFeatureUsingCamPosOrient(pos, orient, res_pair);
    for (const auto& pair : res_pair) res.push_back(pair.second);

    return f_num;
  }
  int getFeatureUsingCamPosOrient(const Vector3d& pos, const Quaterniond& orient) {
    vector<pair<int, Vector3d>> res;
    return getFeatureUsingCamPosOrient(pos, orient, res);
  }

  template <typename... Args>
  int getFeatureUsingPos(const Vector3d& pos, Args&&... args) {
    Vector3d pos_transformed;
    feature_cam_->fromOdom2Cam(pos, pos_transformed);

    Quaterniond ori;
    ori.w() = 0.0;
    ori.x() = 0.0;
    ori.y() = 0.0;
    ori.z() = 0.0;

    return getFeatureUsingCamPosOrient(pos_transformed, ori, args...);
  }

  template <typename... Args>
  int getFeatureUsingOdom(const Vector3d& pos, const Quaterniond& orient, Args&&... args) {
    Vector3d pos_transformed;
    Quaterniond orient_transformed;
    feature_cam_->fromOdom2Cam(pos, orient, pos_transformed, orient_transformed);

    return getFeatureUsingCamPosOrient(pos_transformed, orient_transformed, args...);
  }

  template <typename... Args>
  int getFeatureUsingOdom(const nav_msgs::OdometryConstPtr& msg, Args&&... args) {
    Vector3d pos;
    pos.x() = msg->pose.pose.position.x;
    pos.y() = msg->pose.pose.position.y;
    pos.z() = msg->pose.pose.position.z;
    Quaterniond orient;
    orient.w() = msg->pose.pose.orientation.w;
    orient.x() = msg->pose.pose.orientation.x;
    orient.y() = msg->pose.pose.orientation.y;
    orient.z() = msg->pose.pose.orientation.z;

    return getFeatureUsingOdom(pos, orient, args...);
  }

  // Called during determination of range of each intermidiate waypoint of yaw traj
  // Method1(Deprecated)
  Vector2d genLocalizableCorridor(const vector<Vector3d>& targets, const Vector3d& pos, const Vector3d& acc, const double& yaw);
  void genLocalizableCorridor(
      const vector<Vector3d>& p_vec, const vector<Vector3d>& acc_vec, const MatrixXd& yaw, VectorXd& lb, VectorXd& ub);

  // Method2(Normal)
  Vector2d genLocalizableCorridor(const set<int>& targets, const Vector3d& pos, const Vector3d& acc, const double& yaw);
  void genLocalizableCorridor(const YawOptData::Ptr& data, const MatrixXd& yaw, VectorXd& lb, VectorXd& ub);
  //------------------------------------------------

  // Called by frontier_finder.cpp
  // TODO: Remove the external raycast pointer
  void getYawRangeUsingPos(
      const Vector3d& pos, const vector<double>& sample_yaw, vector<vector<int>>& features_ids_per_yaw, RayCaster* raycaster);
  void getFeatureIDUsingPosYaw(const Vector3d& pos, double yaw, vector<int>& feature_id, RayCaster* raycaster);
  void clusterFeatures(const Vector3d& pos_now, const float& cluster_tolerance, const int& min_cluster_size,
      const int& max_cluster_size, std::vector<std::pair<Vector3d, pcl::PointCloud<pcl::PointXYZ>::Ptr>>& clustered_results);

  // Old API
  void getFeatures(const Eigen::Vector3d& pos, vector<Eigen::Vector3d>& res) {
    getFeatureUsingPos(pos, res);
  }

private:
  // Publishers && Subscribers
  ros::Subscriber odom_sub_, sensorpos_sub, pointcloud_sub_;
  ros::Publisher feature_map_pub_, visual_feature_cloud_pub_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  pcl::PointCloud<pcl::PointXYZ> features_cloud_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr known_features_cloud_;
  pcl::KdTreeFLANN<pcl::PointXYZ> features_kdtree_;
  vector<bool> known_flag_;
  double merge_radius_ = 0.5;  // m
};
}  // namespace perception_aware_planner

#endif  // FEATURE_MAP_H