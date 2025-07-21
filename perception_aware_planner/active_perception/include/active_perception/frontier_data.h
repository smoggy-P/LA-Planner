#ifndef PERCEPTION_AWARE_FRONTIER_DATA_H_
#define PERCEPTION_AWARE_FRONTIER_DATA_H_

#include <ros/ros.h>
#include <Eigen/Eigen>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <mutex>

using Eigen::Vector3d;
using std::list;
using std::pair;
using std::set;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

class RayCaster;

namespace perception_aware_planner {

struct Viewpoint {
  // Position
  Vector3d pos_;
  double score_pos_;
  // Yaw
  double yaw_;
  double score_yaw_;
  // others
  double final_score_;
  vector<Vector3d> filtered_cells_;
  vector<int> visual_features_ids_;
  // path
  vector<Eigen::Vector3d> search_path;
  vector<double> search_yaw;
  vector<double> search_cost;
};

// A frontier cluster, the viewpoints to cover it
struct Frontier {
  // Complete voxels belonging to the cluster
  vector<Vector3d> cells_;
  // down-sampled voxels filtered by voxel grid filter
  vector<Vector3d> filtered_cells_;
  // Average position of all voxels
  Vector3d average_;
  // Idx of cluster
  int id_;
  // Viewpoints that can cover the cluster
  vector<Viewpoint> viewpoints_;
  // Bounding box of cluster, center & 1/2 side length
  Vector3d box_min_, box_max_;
  // Path and cost from this cluster to final
  vector<Vector3d> paths2goal;
  vector<double> path_cost;
};

class SortRefer {
public:
  // data
  Vector3d pos_now_;
  Vector3d vel_now_;
  double yaw_now_;

  Vector3d final_goal;
  bool get_final_goal;

  // param
  double we, wg, wf, wc;

  void update(const Vector3d& cur_pos, const Vector3d& cur_vel, const double cur_yaw) {
    pos_now_ = cur_pos;
    vel_now_ = cur_vel;
    yaw_now_ = cur_yaw;
  }
};

class ViewpointManager {
public:
  std::vector<Viewpoint> unavailable_viewpoint;
  pcl::KdTreeFLANN<pcl::PointXYZ> kd_tree;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
  std::vector<double> yaws;
  double pos_thr;
  double yaw_thr;

  ViewpointManager() : cloud(new pcl::PointCloud<pcl::PointXYZ>) {
  }

  // Add Viewpoint and Update the kd tree
  void addViewpoint(const Viewpoint& vp) {
    unavailable_viewpoint.push_back(vp);
    pcl::PointXYZ point;
    point.x = vp.pos_.x();
    point.y = vp.pos_.y();
    point.z = vp.pos_.z();
    cloud->points.push_back(point);
    yaws.push_back(vp.yaw_);
    kd_tree.setInputCloud(cloud);
  }

  // Retrieve the nearest Viewpoint for a given point and return whether it exists
  bool isViewpointExist(const Viewpoint& vp) {
    if (cloud->empty()) return false;

    pcl::PointXYZ search_point;
    search_point.x = vp.pos_.x();
    search_point.y = vp.pos_.y();
    search_point.z = vp.pos_.z();

    std::vector<int> indices;
    std::vector<float> distances;

    if (kd_tree.radiusSearch(search_point, pos_thr, indices, distances) > 0) {
      for (size_t i = 0; i < indices.size(); ++i) {
        double yaw_diff = std::abs(yaws[indices[i]] - vp.yaw_);
        yaw_diff = std::min(yaw_diff, 2 * M_PI - yaw_diff);
        if (yaw_diff <= yaw_thr) return true;
      }
    }

    return false;
  }

  void clear() {
    unavailable_viewpoint.clear();
    cloud->clear();
    yaws.clear();
  }
};

struct ShareFrontierParam {
  // param_output
  vector<vector<Vector3d>> active_frontiers;
  vector<vector<Vector3d>> dead_frontiers;

  vector<Vector3d> viewpoint_pos_vector;
  vector<double> viewpoint_yaw_vector;
  vector<Vector3d> viewpoint_frontier_cell;
  vector<double> score;

  Vector3d final_goal;
  bool get_final_goal;
  vector<Viewpoint> viewpoints;

  // param_input
  Vector3d cur_pos, cur_vel;
  double yaw_now;

  ViewpointManager vp_manager_;
};

struct CandidateParams {
  double candidate_rmax_, candidate_rmin_, candidate_dphi_, feature_sample_dphi_, z_sample_max_length_;
  int candidate_rnum_, z_sample_num_, cand_limit_per_cluster_;
};

}  // namespace perception_aware_planner

#endif