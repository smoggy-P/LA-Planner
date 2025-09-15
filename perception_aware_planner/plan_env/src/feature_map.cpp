#include "plan_env/feature_map.h"
#include "plan_env/sdf_map.h"

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/centroid.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <mutex>
#include <cmath>
#include <limits>

using namespace Eigen;

namespace perception_aware_planner {

static inline bool isFiniteXYZ(const pcl::PointXYZ& p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}

void FeatureMap::setMap(shared_ptr<SDFMap>& global_map, shared_ptr<SDFMap>& map) {
  global_sdf_map_ = global_map;
  sdf_map_ = map;
}

void FeatureMap::initMap(ros::NodeHandle& nh) {

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>();
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  {
    std::lock_guard<std::mutex> lock(features_mutex_);
    features_cloud_.clear();
    known_flag_.clear();
    known_features_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
    // 确保 KD-Tree 初始为空输入
    features_kdtree_.setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>));
  }

  feature_cam_ = Utils::getGlobalParam().feature_cam_;
  if (!feature_cam_) {
    ROS_ERROR("[FeatureMap] Failed to initialize feature_cam_");
    return;
  }

  pointcloud_sub_ = nh.subscribe("/r2d2/point_cloud", 5, &FeatureMap::pointCloudCallback, this);
  odom_sub_       = nh.subscribe("/drone/odom", 50, &FeatureMap::odometryCallback, this);
  sensorpos_sub   = nh.subscribe("/drone/gt_pose", 50, &FeatureMap::sensorPoseCallback, this);

  feature_map_pub_          = nh.advertise<sensor_msgs::PointCloud2>("/feature/feature_map", 10);
  visual_feature_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/feature/visual_feature_cloud", 10);
}

void FeatureMap::pointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
  if (!tf_buffer_->canTransform("world", msg->header.frame_id, msg->header.stamp, ros::Duration(0.0))) {
    ROS_WARN_THROTTLE(2.0, "[FeatureMap] TF %s->world unavailable, skip frame.", msg->header.frame_id.c_str());
    return;
  }
  geometry_msgs::TransformStamped tf =
      tf_buffer_->lookupTransform("world", msg->header.frame_id, msg->header.stamp, ros::Duration(0.0));

  pcl::PointCloud<pcl::PointXYZ>::Ptr original_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(*msg, *original_cloud);

  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (!pcl_ros::transformPointCloud("world", *original_cloud, *transformed_cloud, *tf_buffer_)) {
    ROS_WARN_THROTTLE(2.0, "[FeatureMap] pcl_ros::transformPointCloud failed, skip frame.");
    return;
  }

  transformed_cloud->points.erase(
      std::remove_if(transformed_cloud->points.begin(), transformed_cloud->points.end(),
                     [](const pcl::PointXYZ& p) { return !isFiniteXYZ(p); }),
      transformed_cloud->points.end());

  updateFeatureMap(transformed_cloud);
}

void FeatureMap::updateFeatureMap(const pcl::PointCloud<pcl::PointXYZ>::Ptr& new_features) {
  if (!new_features || new_features->empty()) return;

  std::lock_guard<std::mutex> lock(features_mutex_);

  if (features_cloud_.empty()) {
    features_cloud_ = *new_features;
    *known_features_cloud_ = features_cloud_;
    known_flag_.assign(features_cloud_.size(), true);

    if (!features_cloud_.empty()) {
      features_kdtree_.setInputCloud(features_cloud_.makeShared());
    }
    visFeatureMap(); 
    return;
  }

  if (!features_kdtree_.getInputCloud() || features_kdtree_.getInputCloud()->empty()) {
    features_kdtree_.setInputCloud(features_cloud_.makeShared());
  }

  for (const auto& pt : new_features->points) {
    if (!isFiniteXYZ(pt)) continue;

    std::vector<int> indices;
    std::vector<float> sqr_distances;

    try {
      if (features_kdtree_.radiusSearch(pt, merge_radius_, indices, sqr_distances) == 0) {
        features_cloud_.points.push_back(pt);
        known_flag_.push_back(true);
      }
    } catch (const std::exception& e) {
      ROS_ERROR("[FeatureMap] Exception in updateFeatureMap radiusSearch: %s", e.what());
      features_cloud_.points.push_back(pt);
      known_flag_.push_back(true);
    }
  }

  if (features_cloud_.size() != known_flag_.size()) {
    ROS_WARN("[FeatureMap] known_flag_ size mismatch, fix it: cloud=%zu flag=%zu",
             features_cloud_.size(), known_flag_.size());
    if (features_cloud_.size() > known_flag_.size())
      known_flag_.resize(features_cloud_.size(), true);
    else
      known_flag_.erase(known_flag_.begin() + features_cloud_.size(), known_flag_.end());
  }

  *known_features_cloud_ = features_cloud_;
  if (!features_cloud_.empty()) {
    features_kdtree_.setInputCloud(features_cloud_.makeShared());
  }

  visFeatureMap();
}

void FeatureMap::loadMap(const string& filename) {
  std::lock_guard<std::mutex> lock(features_mutex_);
  features_cloud_.clear();
  known_flag_.clear();
  known_features_cloud_->clear();
  features_kdtree_.setInputCloud(features_cloud_.makeShared());
  ROS_WARN("[FeatureMap] loadMap called but using sensor-based mode, ignoring file: %s", filename.c_str());
}

void FeatureMap::getFeatureCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
  std::lock_guard<std::mutex> lock(features_mutex_);
  cloud = features_cloud_.makeShared();
}

void FeatureMap::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
  vector<Vector3d> visual_points_vec;
  getFeatureUsingOdom(msg, visual_points_vec);

  pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZ>);
  pointcloud->width = visual_points_vec.size();
  pointcloud->height = 1;
  pointcloud->is_dense = true;
  pointcloud->header.frame_id = "world";
  pointcloud->points.resize(pointcloud->width);
  for (size_t i = 0; i < visual_points_vec.size(); ++i) {
    pointcloud->points[i].x = visual_points_vec[i].x();
    pointcloud->points[i].y = visual_points_vec[i].y();
    pointcloud->points[i].z = visual_points_vec[i].z();
  }

  sensor_msgs::PointCloud2 pointcloud_msg;
  pcl::toROSMsg(*pointcloud, pointcloud_msg);
  visual_feature_cloud_pub_.publish(pointcloud_msg);
}

void FeatureMap::sensorPoseCallback(const geometry_msgs::PoseStampedConstPtr& /*pose*/) {
  visFeatureMap();
}

void FeatureMap::visFeatureMap() {
  if (!known_features_cloud_ || known_features_cloud_->empty()) return;
  known_features_cloud_->header.frame_id = "world";
  known_features_cloud_->width = known_features_cloud_->points.size();
  known_features_cloud_->height = 1;
  known_features_cloud_->is_dense = true;
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*known_features_cloud_, cloud_msg);
  feature_map_pub_.publish(cloud_msg);
}

int FeatureMap::getFeatureUsingCamPosOrient(const Vector3d& pos, const Quaterniond& orient,
                                            vector<pair<int, Vector3d>>& res) {
  std::lock_guard<std::mutex> lock(features_mutex_);
  if (features_cloud_.empty()) return 0;
  res.clear();

  if (!features_kdtree_.getInputCloud() || features_kdtree_.getInputCloud()->empty()) {
    features_kdtree_.setInputCloud(features_cloud_.makeShared());
  }

  vector<int> idx_vec;
  vector<float> dis_vec;
  pcl::PointXYZ searchPoint;
  searchPoint.x = pos.x();
  searchPoint.y = pos.y();
  searchPoint.z = pos.z();

  try {
    features_kdtree_.radiusSearch(searchPoint, feature_cam_->visual_max, idx_vec, dis_vec);
  } catch (const std::exception& e) {
    ROS_ERROR("[FeatureMap] radiusSearch exception: %s", e.what());
    return 0;
  }

  for (const auto& index : idx_vec) {
    if (index < 0 || index >= static_cast<int>(features_cloud_.size())) continue;
    if (index >= static_cast<int>(known_flag_.size())) continue;
    if (!known_flag_[index]) continue;

    Vector3d f(features_cloud_[index].x, features_cloud_[index].y, features_cloud_[index].z);

    const bool in_fov = (orient.norm() > 0.1) ? feature_cam_->inFOV(pos, f, orient) : feature_cam_->inFOV(pos, f);
    if (in_fov && (!global_sdf_map_ || !global_sdf_map_->checkObstacleBetweenPoints(pos, f))) {
      res.emplace_back(index, f);
    }
  }
  return static_cast<int>(res.size());
}

Vector3d FeatureMap::getFeatureByID(const int id) {
  std::lock_guard<std::mutex> lock(features_mutex_);
  if (id >= 0 && id < static_cast<int>(features_cloud_.size())) {
    Vector3d f(features_cloud_[id].x, features_cloud_[id].y, features_cloud_[id].z);
    return f;
  }
  return Vector3d::Zero();
}

Vector2d FeatureMap::genLocalizableCorridor(
    const vector<Vector3d>& targets, const Vector3d& pos, const Vector3d& acc, const double& yaw) {
  Vector2d ret;

  const double step = 0.05;

  double cur_yaw = yaw;
  bool quit = false;

  while (!quit) {
    cur_yaw += step;
    Quaterniond q = Utils::calcOrientation(cur_yaw, acc);

    Vector3d pc;
    Quaterniond qc;
    feature_cam_->fromOdom2Cam(pos, q, pc, qc);

    quit = false;
    for (const auto& target : targets) {
      if (!feature_cam_->inFOV(pc, target, qc)) {
        quit = true;
        break;
      }
    }
  }
  ret(0) = cur_yaw - step;
  if (std::abs(ret(0) - yaw) < 1e-4) ret(0) += 1e-3;

  cur_yaw = yaw;
  quit = false;
  while (!quit) {
    cur_yaw -= step;
    Quaterniond q = Utils::calcOrientation(cur_yaw, acc);

    Vector3d pc;
    Quaterniond qc;
    feature_cam_->fromOdom2Cam(pos, q, pc, qc);

    quit = false;
    for (const auto& target : targets) {
      if (!feature_cam_->inFOV(pc, target, qc)) {
        quit = true;
        break;
      }
    }
  }
  ret(1) = cur_yaw + step;
  if (std::abs(ret(1) - yaw) < 1e-4) ret(1) -= 1e-3;

  return ret;
}

void FeatureMap::genLocalizableCorridor(
    const vector<Vector3d>& p_vec, const vector<Vector3d>& acc_vec, const MatrixXd& yaw, VectorXd& lb, VectorXd& ub) {

  for (size_t i = 0; i < p_vec.size(); i++) {
    auto p = p_vec[i];
    auto ori = Utils::calcOrientation(yaw(0, i), acc_vec[i]);
    vector<Vector3d> features;

    int feature_num = getFeatureUsingOdom(p, ori, features);
    ROS_ASSERT(feature_num > Utils::getGlobalParam().min_feature_num_plan_);

    Vector2d ret = genLocalizableCorridor(features, p, acc_vec[i], yaw(0, i));
    lb(i) = ret(1);
    ub(i) = ret(0);
  }
}

Vector2d FeatureMap::genLocalizableCorridor(
    const set<int>& targets, const Vector3d& pos, const Vector3d& acc, const double& yaw) {
  Vector2d ret;

  const vector<double> resolutions = { 0.05, 0.01 };

  auto searchBoundray = [&](const bool& dir, double& bound) {
    double start_yaw = yaw;

    for (size_t i = 0; i < resolutions.size(); i++) {
      double res = dir ? resolutions[i] : -resolutions[i];

      double cur_yaw = start_yaw;

      while (true) {
        cur_yaw += res;
        Quaterniond ori = Utils::calcOrientation(cur_yaw, acc);
        set<int> id;
        int feature_num = getFeatureUsingOdom(pos, ori, id);
        int covis_num = Utils::getSameCount(id, targets);

        if (covis_num < static_cast<int>(targets.size()) ||
            feature_num <= Utils::getGlobalParam().min_feature_num_plan_) break;
      }

      start_yaw = cur_yaw - res;
    }

    bound = start_yaw;
  };

  searchBoundray(true, ret(0));
  searchBoundray(false, ret(1));

  if (std::abs(ret(0) - yaw) < 1e-4) ret(0) += 1e-3;
  if (std::abs(ret(1) - yaw) < 1e-4) ret(1) -= 1e-3;

  return ret;
}

void FeatureMap::genLocalizableCorridor(const YawOptData::Ptr& data, const MatrixXd& yaw, VectorXd& lb, VectorXd& ub) {
  for (size_t i = 0; i < data->pos_vec_.size(); i++) {
    const auto& pos = data->pos_vec_[i];
    const auto& acc = data->acc_vec_[i];

    Vector2d ret = genLocalizableCorridor(data->target_covis_features_[i], pos, acc, yaw(0, i));
    lb(i) = ret(1);
    ub(i) = ret(0);
  }
}

void FeatureMap::getYawRangeUsingPos(
    const Vector3d& pos, const vector<double>& sample_yaw,
    vector<vector<int>>& features_ids_per_yaw, RayCaster* raycaster) {

  features_ids_per_yaw.clear();
  features_ids_per_yaw.resize(sample_yaw.size());

  std::lock_guard<std::mutex> lock(features_mutex_);

  if (features_cloud_.empty()) {
    ROS_WARN("[FeatureMap] Features cloud is empty in getYawRangeUsingPos");
    return;
  }

  if (!features_kdtree_.getInputCloud() || features_kdtree_.getInputCloud()->empty()) {
    features_kdtree_.setInputCloud(features_cloud_.makeShared());
  }

  Vector3d pos_cam;
  feature_cam_->fromOdom2Cam(pos, pos_cam);  // 仍保留相机系用于 FOV/深度判断

  pcl::PointXYZ searchPoint;
  searchPoint.x = pos.x();   // —— KD-Tree 查询使用 world 坐标 —— //
  searchPoint.y = pos.y();
  searchPoint.z = pos.z();

  vector<int> pointIdxRadiusSearch;
  vector<float> pointRadiusSquaredDistance;

  try {
    int found_points = features_kdtree_.radiusSearch(
        searchPoint, feature_cam_->visual_max,
        pointIdxRadiusSearch, pointRadiusSquaredDistance);
    if (found_points <= 0) {
      ROS_DEBUG("[FeatureMap] No points found in radius search");
      return;
    }
  } catch (const std::exception& e) {
    ROS_ERROR("[FeatureMap] Exception in radiusSearch: %s", e.what());
    return;
  }

  for (auto& feature_ids : features_ids_per_yaw)
    feature_ids.reserve(pointIdxRadiusSearch.size());

  for (const auto& index : pointIdxRadiusSearch) {
    if (index < 0 || index >= static_cast<int>(features_cloud_.size())) continue;

    Eigen::Vector3d f(features_cloud_[index].x,
                      features_cloud_[index].y,
                      features_cloud_[index].z);

    if (feature_cam_->inVisbleDepthAtLevel(pos_cam, f) &&
        (!global_sdf_map_ ||
         !global_sdf_map_->checkObstacleBetweenPoints(pos, f, raycaster))) {

      Eigen::Vector2d yaw_range = feature_cam_->calculateYawRange(pos_cam, f);

      for (size_t i = 0; i < sample_yaw.size(); ++i) {
        double yaw = sample_yaw[i];
        if (yaw_range(0) < yaw_range(1)) {
          if (yaw >= yaw_range(0) && yaw <= yaw_range(1))
            features_ids_per_yaw[i].push_back(index);
        } else {
          if (yaw >= yaw_range(0) || yaw <= yaw_range(1))
            features_ids_per_yaw[i].push_back(index);
        }
      }
    }
  }
}

void FeatureMap::getFeatureIDUsingPosYaw(const Vector3d& pos, double yaw, vector<int>& feature_id, RayCaster* raycaster) {
  feature_id.clear();

  Eigen::AngleAxisd angle_axis(yaw, Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond odom_orient(angle_axis);

  Vector3d pos_cam;
  Eigen::Quaterniond odom_cam;
  feature_cam_->fromOdom2Cam(pos, odom_orient, pos_cam, odom_cam);

  std::lock_guard<std::mutex> lock(features_mutex_);
  if (features_cloud_.empty()) return;

  if (!features_kdtree_.getInputCloud() || features_kdtree_.getInputCloud()->empty()) {
    features_kdtree_.setInputCloud(features_cloud_.makeShared());
  }

  pcl::PointXYZ searchPoint;
  searchPoint.x = pos.x();  
  searchPoint.y = pos.y();
  searchPoint.z = pos.z();

  vector<int> pointIdxRadiusSearch;
  vector<float> pointRadiusSquaredDistance;

  try {
    features_kdtree_.radiusSearch(searchPoint, feature_cam_->visual_max,
                                  pointIdxRadiusSearch, pointRadiusSquaredDistance);
  } catch (const std::exception& e) {
    ROS_ERROR("[FeatureMap] radiusSearch exception: %s", e.what());
    return;
  }

  for (const auto& index : pointIdxRadiusSearch) {
    if (index < 0 || index >= static_cast<int>(features_cloud_.size())) continue;

    Eigen::Vector3d f(features_cloud_[index].x, features_cloud_[index].y, features_cloud_[index].z);

    if (feature_cam_->inFOV(pos_cam, f, odom_cam) &&
        (!global_sdf_map_ || !global_sdf_map_->checkObstacleBetweenPoints(pos, f, raycaster))) {
      feature_id.push_back(index);
    }
  }
}

void FeatureMap::clusterFeatures(const Vector3d& pos_now, const float& cluster_tolerance, const int& min_cluster_size,
    const int& max_cluster_size, std::vector<std::pair<Vector3d, pcl::PointCloud<pcl::PointXYZ>::Ptr>>& clustered_results) {

  clustered_results.clear();

  std::lock_guard<std::mutex> lock(features_mutex_);

  if (features_cloud_.empty()) {
    ROS_WARN("Features cloud is empty in clusterFeatures");
    return;
  }
  if (cluster_tolerance <= 0 || min_cluster_size <= 0 || max_cluster_size <= 0) {
    ROS_ERROR("Invalid clustering parameters: tolerance=%f, min=%d, max=%d",
              cluster_tolerance, min_cluster_size, max_cluster_size);
    return;
  }
  if (!feature_cam_) {
    ROS_ERROR("feature_cam_ is null in clusterFeatures");
    return;
  }
  if (!features_kdtree_.getInputCloud()) {
    ROS_WARN("KD tree not initialized in clusterFeatures, rebuilding...");
    try {
      features_kdtree_.setInputCloud(features_cloud_.makeShared());
    } catch (const std::exception& e) {
      ROS_ERROR("Failed to initialize KD tree: %s", e.what());
      return;
    }
  }
  if (features_kdtree_.getInputCloud()->empty()) {
    ROS_ERROR("KD tree input cloud is empty");
    return;
  }
  if (features_kdtree_.getInputCloud()->size() != features_cloud_.size()) {
    ROS_WARN("KD tree size mismatch, rebuilding... (tree: %zu, features: %zu)",
             features_kdtree_.getInputCloud()->size(), features_cloud_.size());
    try {
      features_kdtree_.setInputCloud(features_cloud_.makeShared());
    } catch (const std::exception& e) {
      ROS_ERROR("Failed to rebuild KD tree: %s", e.what());
      return;
    }
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointXYZ searchPoint;
  searchPoint.x = pos_now(0);
  searchPoint.y = pos_now(1);
  searchPoint.z = pos_now(2);

  vector<int> pointIdxRadiusSearch;
  vector<float> pointRadiusSquaredDistance;
  try {
    features_kdtree_.radiusSearch(searchPoint, feature_cam_->visual_max,
                                  pointIdxRadiusSearch, pointRadiusSquaredDistance);
  } catch (const std::exception& e) {
    ROS_ERROR("radiusSearch failed in clusterFeatures: %s", e.what());
    return;
  }

  for (const auto& index : pointIdxRadiusSearch) {
    if (index < 0 || index >= static_cast<int>(features_cloud_.size())) continue;
    filtered_cloud->points.push_back(features_cloud_[index]);
  }

  if (filtered_cloud->points.empty()) return;

  boost::shared_ptr<pcl::search::KdTree<pcl::PointXYZ>> tree(new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud(filtered_cloud);

  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance);
  ec.setMinClusterSize(min_cluster_size);
  ec.setMaxClusterSize(max_cluster_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(filtered_cloud);

  std::vector<pcl::PointIndices> cluster_indices;
  ec.extract(cluster_indices);

  for (const auto& indices : cluster_indices) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cluster_cloud->points.reserve(indices.indices.size());
    for (const auto& idx : indices.indices) {
      if (idx >= 0 && idx < static_cast<int>(filtered_cloud->points.size()))
        cluster_cloud->points.push_back(filtered_cloud->points[idx]);
    }

    if (cluster_cloud->points.empty()) continue;

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cluster_cloud, centroid);

    Vector3d cluster_center(centroid[0], centroid[1], centroid[2]);
    clustered_results.push_back({ cluster_center, cluster_cloud });
  }
}

void FeatureMap::reset() {
  std::lock_guard<std::mutex> lock(features_mutex_);
  features_cloud_.clear();
  known_flag_.clear();
  known_features_cloud_->clear();
  features_kdtree_.setInputCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>));
}

}  // namespace perception_aware_planner