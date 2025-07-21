#include "plan_env/feature_map.h"

#include "plan_env/sdf_map.h"

#include <execution>

using namespace Eigen;

namespace perception_aware_planner {

void FeatureMap::setMap(shared_ptr<SDFMap>& global_map, shared_ptr<SDFMap>& map) {
  global_sdf_map_ = global_map;
  sdf_map_ = map;
}

void FeatureMap::initMap(ros::NodeHandle& nh) {

  std::string filename;
  nh.param<std::string>("feature/filename", filename, "");
  loadMap(filename);
  known_flag_.assign(features_cloud_.size(), false);
  known_features_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);

  feature_cam_ = Utils::getGlobalParam().feature_cam_;

  odom_sub_ = nh.subscribe("/odom_world", 1, &FeatureMap::odometryCallback, this);
  sensorpos_sub = nh.subscribe("/map_ros/pose", 1, &FeatureMap::sensorPoseCallback, this);
  feature_map_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/feature/feature_map", 10);
  visual_feature_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/feature/visual_feature_cloud", 10);
}

void FeatureMap::loadMap(const string& filename) {
  features_cloud_.clear();
  bool use_simple_features = (filename == "");
  if (use_simple_features) {
    // Features at the central of the wall
    for (double y = 3.0; y < 17.0; y += 0.2) {
      for (double z = 0.0; z < 5.0; z += 0.2) {
        pcl::PointXYZ p;
        p.x = 5.0;
        p.y = y;
        p.z = z;
        features_cloud_.push_back(p);
      }
    }
  }

  else {
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(filename, features_cloud_) == -1) {
      ROS_ERROR("[FeatureMap] Failed to load PLY file: %s", filename.c_str());
      return;
    }
  }

  features_kdtree_.setInputCloud(features_cloud_.makeShared());
  ROS_WARN("[FeatureMap] Load Success!!! filename: %s features num:%zu", filename.c_str(), features_cloud_.size());
}

void FeatureMap::getFeatureCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
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

void FeatureMap::sensorPoseCallback(const geometry_msgs::PoseStampedConstPtr& pose) {
  Vector3d camera_p;
  camera_p(0) = pose->pose.position.x;
  camera_p(1) = pose->pose.position.y;
  camera_p(2) = pose->pose.position.z;
  Quaterniond camera_q;
  camera_q.w() = pose->pose.orientation.w;
  camera_q.x() = pose->pose.orientation.x;
  camera_q.y() = pose->pose.orientation.y;
  camera_q.z() = pose->pose.orientation.z;

  if (!features_cloud_.empty() && global_sdf_map_->hasInitialized_) {
    pcl::PointXYZ searchPoint;
    searchPoint.x = camera_p(0);
    searchPoint.y = camera_p(1);
    searchPoint.z = camera_p(2);

    vector<int> idx_vec;
    vector<float> dis_vec;
    features_kdtree_.radiusSearch(searchPoint, feature_cam_->visual_max, idx_vec, dis_vec);

    for (const auto& index : idx_vec) {
      if (known_flag_[index]) continue;

      Vector3d f(features_cloud_[index].x, features_cloud_[index].y, features_cloud_[index].z);
      if (!feature_cam_->inFOV(camera_p, f, camera_q)) continue;

      if (global_sdf_map_->checkObstacleBetweenPoints(camera_p, f)) continue;

      known_flag_[index] = true;
      known_features_cloud_->push_back(features_cloud_[index]);
    }
  }
  visFeatureMap();
}

void FeatureMap::visFeatureMap() {

  if (known_features_cloud_->empty()) return;

  // ROS_INFO("[FeatureMap] Known Feature Map Size: %d", known_features_cloud_->points.size());
  // ROS_INFO("[FeatureMap] Feature Map Size: %d", features_cloud_.points.size());

  known_features_cloud_->header.frame_id = "world";

  known_features_cloud_->width = known_features_cloud_->points.size();
  known_features_cloud_->height = 1;
  known_features_cloud_->is_dense = true;

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*known_features_cloud_, cloud_msg);
  feature_map_pub_.publish(cloud_msg);
}

int FeatureMap::getFeatureUsingCamPosOrient(const Vector3d& pos, const Quaterniond& orient, vector<pair<int, Vector3d>>& res) {
  if (features_cloud_.empty()) return 0;
  res.clear();

  vector<int> idx_vec;
  vector<float> dis_vec;
  pcl::PointXYZ searchPoint;
  searchPoint.x = pos.x();
  searchPoint.y = pos.y();
  searchPoint.z = pos.z();
  features_kdtree_.radiusSearch(searchPoint, feature_cam_->visual_max, idx_vec, dis_vec);

  for (const auto& index : idx_vec) {
    if (!known_flag_[index]) continue;

    Vector3d f(features_cloud_[index].x, features_cloud_[index].y, features_cloud_[index].z);

    if ((orient.norm() > 0.1 ? feature_cam_->inFOV(pos, f, orient) : feature_cam_->inFOV(pos, f)) &&
        !global_sdf_map_->checkObstacleBetweenPoints(pos, f)) {
      res.emplace_back(index, f);
    }
  }
  return res.size();
}

Vector3d FeatureMap::getFeatureByID(const int id) {
  Vector3d f(features_cloud_[id].x, features_cloud_[id].y, features_cloud_[id].z);
  return f;
}

// Method1
Vector2d FeatureMap::genLocalizableCorridor(
    const vector<Vector3d>& targets, const Vector3d& pos, const Vector3d& acc, const double& yaw) {
  Vector2d ret;

  double step = 0.05;

  double cur_yaw = yaw;
  bool quit = false;
  while (!quit) {
    cur_yaw += step;

    Quaterniond q = Utils::calcOrientation(cur_yaw, acc);

    Vector3d pc;
    Quaterniond qc;
    feature_cam_->fromOdom2Cam(pos, q, pc, qc);

    std::for_each(std::execution::par, targets.begin(), targets.end(), [&](auto& target) {
      if (!feature_cam_->inFOV(pc, target, qc)) {
        quit = true;
      }
    });
  }
  ret(0) = cur_yaw - step;

  if (abs(ret(0) - yaw) < 1e-4) ret(0) += 1e-3;

  cur_yaw = yaw;
  quit = false;
  while (!quit) {
    cur_yaw -= step;

    Quaterniond q = Utils::calcOrientation(cur_yaw, acc);

    Vector3d pc;
    Quaterniond qc;
    feature_cam_->fromOdom2Cam(pos, q, pc, qc);

    std::for_each(std::execution::par, targets.begin(), targets.end(), [&](auto& target) {
      if (!feature_cam_->inFOV(pc, target, qc)) {
        quit = true;
      }
    });
  }

  ret(1) = cur_yaw + step;
  if (abs(ret(1) - yaw) < 1e-4) ret(1) -= 1e-3;

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

// Method2
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

        // All targets must be visible and have sufficient feature points
        if (covis_num < targets.size() || feature_num <= Utils::getGlobalParam().min_feature_num_plan_) break;
      }

      start_yaw = cur_yaw - res;
    }

    bound = start_yaw;
  };

  searchBoundray(true, ret(0));
  searchBoundray(false, ret(1));

  if (abs(ret(0) - yaw) < 1e-4) ret(0) += 1e-3;  // ub
  if (abs(ret(1) - yaw) < 1e-4) ret(1) -= 1e-3;  // lb

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
    const Vector3d& pos, const vector<double>& sample_yaw, vector<vector<int>>& features_ids_per_yaw, RayCaster* raycaster) {
  features_ids_per_yaw.clear();
  features_ids_per_yaw.resize(sample_yaw.size());

  Vector3d pos_transformed;
  feature_cam_->fromOdom2Cam(pos, pos_transformed);

  if (features_cloud_.empty()) return;

  pcl::PointXYZ searchPoint;
  searchPoint.x = pos_transformed(0);
  searchPoint.y = pos_transformed(1);
  searchPoint.z = pos_transformed(2);

  vector<int> pointIdxRadiusSearch;
  vector<float> pointRadiusSquaredDistance;
  features_kdtree_.radiusSearch(searchPoint, feature_cam_->visual_max, pointIdxRadiusSearch, pointRadiusSquaredDistance);

  for (auto& feature_ids : features_ids_per_yaw) feature_ids.reserve(pointIdxRadiusSearch.size());

  for (const auto& index : pointIdxRadiusSearch) {
    Eigen::Vector3d f(features_cloud_[index].x, features_cloud_[index].y, features_cloud_[index].z);
    if (!known_flag_[index]) continue;

    if (feature_cam_->inVisbleDepthAtLevel(pos_transformed, f) &&
        !global_sdf_map_->checkObstacleBetweenPoints(pos_transformed, f, raycaster)) {

      Eigen::Vector2d yaw_range = feature_cam_->calculateYawRange(pos_transformed, f);

      for (size_t i = 0; i < sample_yaw.size(); ++i) {
        double yaw = sample_yaw[i];

        if (yaw_range(0) < yaw_range(1)) {
          if (yaw >= yaw_range(0) && yaw <= yaw_range(1)) {
            features_ids_per_yaw[i].push_back(index);
          }
        }

        else {
          if (yaw >= yaw_range(0) || yaw <= yaw_range(1)) {
            features_ids_per_yaw[i].push_back(index);
          }
        }
      }
    }
  }
}

void FeatureMap::getFeatureIDUsingPosYaw(const Vector3d& pos, double yaw, vector<int>& feature_id, RayCaster* raycaster) {
  feature_id.clear();

  Eigen::AngleAxisd angle_axis(yaw, Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond odom_orient(angle_axis);

  Vector3d pos_transformed;
  Eigen::Quaterniond odom_transformed;
  feature_cam_->fromOdom2Cam(pos, odom_orient, pos_transformed, odom_transformed);

  if (features_cloud_.empty()) return;

  pcl::PointXYZ searchPoint;
  searchPoint.x = pos_transformed(0);
  searchPoint.y = pos_transformed(1);
  searchPoint.z = pos_transformed(2);

  vector<int> pointIdxRadiusSearch;
  vector<float> pointRadiusSquaredDistance;
  features_kdtree_.radiusSearch(searchPoint, feature_cam_->visual_max, pointIdxRadiusSearch, pointRadiusSquaredDistance);

  for (const auto& index : pointIdxRadiusSearch) {
    Eigen::Vector3d f(features_cloud_[index].x, features_cloud_[index].y, features_cloud_[index].z);
    if (!known_flag_[index]) continue;

    if (feature_cam_->inFOV(pos_transformed, f, odom_transformed) &&
        !global_sdf_map_->checkObstacleBetweenPoints(pos_transformed, f, raycaster)) {
      feature_id.push_back(index);
    }
  }
}

void FeatureMap::clusterFeatures(const Vector3d& pos_now, const float& cluster_tolerance, const int& min_cluster_size,
    const int& max_cluster_size, std::vector<std::pair<Vector3d, pcl::PointCloud<pcl::PointXYZ>::Ptr>>& clustered_results) {

  clustered_results.clear();
  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
  if (features_cloud_.empty()) return;

  // Step1: Collect pointcloud arround the position
  pcl::PointXYZ searchPoint;
  searchPoint.x = pos_now(0);
  searchPoint.y = pos_now(1);
  searchPoint.z = pos_now(2);

  vector<int> pointIdxRadiusSearch;
  vector<float> pointRadiusSquaredDistance;
  features_kdtree_.radiusSearch(searchPoint, feature_cam_->visual_max, pointIdxRadiusSearch, pointRadiusSquaredDistance);

  for (const auto& index : pointIdxRadiusSearch) {
    if (!known_flag_[index]) continue;
    filtered_cloud->points.push_back(features_cloud_[index]);
  }

  // Step2: Setup KdTree for the filtered point cloud
  boost::shared_ptr<pcl::search::KdTree<pcl::PointXYZ>> tree(new pcl::search::KdTree<pcl::PointXYZ>);
  if (filtered_cloud->points.empty()) return;
  tree->setInputCloud(filtered_cloud);

  // Step3: Apply Euclidean Cluster Extraction
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance(cluster_tolerance);  // Set the cluster tolerance (distance)
  ec.setMinClusterSize(min_cluster_size);     // Set the minimum cluster size
  ec.setMaxClusterSize(max_cluster_size);     // Set the maximum cluster size
  ec.setSearchMethod(tree);                   // Use KdTree for search
  ec.setInputCloud(filtered_cloud);           // Set the input cloud

  std::vector<pcl::PointIndices> cluster_indices;
  ec.extract(cluster_indices);  // Perform the clustering

  // Step4: Iterate through the clusters and compute their centers
  for (const auto& indices : cluster_indices) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZ>());

    // Collect points in the cluster
    for (const auto& index : indices.indices) cluster_cloud->points.push_back(filtered_cloud->points[index]);

    // Compute the centroid of the cluster
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cluster_cloud, centroid);

    // Convert to Eigen::Vector3d (centroid position)
    Vector3d cluster_center(centroid[0], centroid[1], centroid[2]);

    // Store the cluster center and point cloud in the result vector
    clustered_results.push_back({ cluster_center, cluster_cloud });
  }
}

}  // namespace perception_aware_planner