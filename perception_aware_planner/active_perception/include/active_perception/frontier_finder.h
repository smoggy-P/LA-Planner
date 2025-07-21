#ifndef PERCEPTION_AWARE_FRONTIER_FINDER_H_
#define PERCEPTION_AWARE_FRONTIER_FINDER_H_

#include "active_perception/localization_aware_graph_search.h"
#include "active_perception/frontier_data.h"

#include "path_searching/astar2.h"

#include "utils/utils.h"

#include <ros/ros.h>

#include <Eigen/Eigen>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>

#include <thread>
#include <mutex>

using Eigen::Vector3d;
using std::list;
using std::pair;
using std::set;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;

class RayCaster;

namespace perception_aware_planner {

class EDTEnvironment;
class FeatureMap;
class PlanningVisualization;
class LocalizationAwareGraphSearch;
class Astar;

class FrontierFinder {
public:
  FrontierFinder(const shared_ptr<EDTEnvironment>& edt, const shared_ptr<FeatureMap>& fea, ros::NodeHandle& nh);
  ~FrontierFinder();

  void setVisualization(const shared_ptr<PlanningVisualization>& visualization) {
    visualization_ = visualization;
  }

  // Interaction Function
  void getFrontiers(vector<vector<Vector3d>>& clusters);
  void getDormantFrontiers(vector<vector<Vector3d>>& clusters);

  void getFrontiersScore(vector<vector<Eigen::Vector3d>>& active_clusters, vector<vector<double>>& active_score,
      vector<Eigen::Vector3d>& active_pos, vector<vector<Eigen::Vector3d>>& dead_clusters, vector<vector<double>>& dead_score,
      vector<Eigen::Vector3d>& dead_pos);
  void getAstarPath(vector<vector<Vector3d>>& active_path, vector<vector<Vector3d>>& dead_path);

  void getBestViewpointData(const vector<Viewpoint>& available_viewpoints, vector<Vector3d>& points, vector<double>& yaws,
      vector<Vector3d>& frontier_cells, vector<double>& score);
  void getSortedViewpointVector(vector<Viewpoint>& viewpoints);

  bool isinterstFrontierCovered(vector<Vector3d>& frontier_cells);  // Get several viewpoints for a subset of frontiers

  // Independent Process
  void getShareFrontierParam(const Vector3d& cur_pos, const Vector3d& cur_vel, const double& yaw_now,  // input
      vector<vector<Eigen::Vector3d>>& active_frontiers, vector<vector<Eigen::Vector3d>>& dead_frontiers,
      vector<Vector3d>& points, vector<double>& yaws, vector<Vector3d>& frontier_cells, vector<double>& score);  // out_put
  void resetViewpointManager();
  bool chooseNextViewpoint(vector<Vector3d>& points, vector<double>& yaws, vector<Vector3d>& frontier_cells);
  void setFinalGoal(const Vector3d& final_goal);

private:
  // Params
  // Cluster Parameters
  int cluster_min_;
  double cluster_size_xy_;
  double min_candidate_clearance_;

  // Visualization Flags
  bool visual_scores;
  bool visual_all_frontier;
  bool visual_feature_cluster;
  bool visual_frontier_viewpoint;
  bool visual_final_viewpoint;
  bool visual_feature_viewpoint;
  bool visual_astar_path_;

  // Miscellaneous Parameters
  double min_view_finish_fraction_, resolution_, ceiling_dir_;
  int down_sample_, min_visib_num_;

  // Independent Process Params
  ShareFrontierParam shared_param_;  // Structure for storing shared data
  std::atomic<bool> running_;        // Atomic variable to control thread running state
  std::thread worker_thread_;        // Independent thread
  std::mutex data_mutex_share_;      // Mutex to protect shared data
  bool begin_ = false;

  // Utils
  shared_ptr<EDTEnvironment> edt_env_ = nullptr;
  unique_ptr<RayCaster> raycaster_ = nullptr;
  shared_ptr<FeatureMap> feature_map_ = nullptr;
  shared_ptr<PlanningVisualization> visualization_ = nullptr;
  shared_ptr<Astar> path_finder_ = nullptr;
  CameraParam::Ptr frontier_cam_ = nullptr;
  CameraParam::Ptr feature_cam_ = nullptr;

  SortRefer sort_refer_;
  CandidateParams frontier_candidate_params_, final_goal_candidate_params_, feature_candidate_params_;
  shared_ptr<LocalizationAwareGraphSearch> graph_search_ = nullptr;

  // Global Data
  vector<char> frontier_flag_;
  list<Frontier> frontiers_, dormant_frontiers_, tmp_frontiers_;
  list<Frontier>::iterator first_new_ftr_;
  vector<std::pair<Vector3d, pcl::PointCloud<pcl::PointXYZ>::Ptr>> feature_cluster;  // pair(center,features)
  vector<Viewpoint> final_viewpoint_;   // the sampling of the cylindrical area around the final goal
  vector<Viewpoint> goal_viewpoint_;    // the sampling of yaw on the final goal.
  vector<Viewpoint> feature_viewpoint;  // the sampling between two feature cluster.

  // Search Frontiers and Viewpoint
  void searchFrontiers();
  void computeConsistScore(Viewpoint& viewpoint);
  void computeFrontiersToVisit();
  void computeFinalViewpoint();
  void computeFeatureViewpoint();
  void computeViewpointinGoal();
  void computeFrontierViewpoint(Frontier& frontier);
  void sortFrontiers();
  void sampleCylindricalViewpoints(const CandidateParams& params, const Vector3d& center, vector<Viewpoint>& sampled_viewpoints);
  void sampleJunctionVPs(const CandidateParams& params,
      const std::vector<std::pair<Vector3d, pcl::PointCloud<pcl::PointXYZ>::Ptr>>& clustered_results,
      vector<Viewpoint>& sampled_viewpoints);
  double calExplorability(const Vector3d& pos, const double& yaw, const vector<Vector3d>& cluster);
  bool computeAstarPath(Frontier& frontier);

  // Search Graph
  void buildFeatureGraph();
  void isViewpointPathFeasible(const Viewpoint& vp);

  // Independent Process
  void frontierThread();  // Main function of the independent thread
  void updateShareFrontierParam();
  void visualFrontier();

  // Handle Frontier
  void splitLargeFrontiers(list<Frontier>& frontiers);
  bool splitHorizontally(const Frontier& frontier, list<Frontier>& splits);
  bool isFrontierChanged(const Frontier& ft);
  bool isFrontierCeiling(const Frontier& ft);
  bool haveOverlap(const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2);
  void computeFrontierInfo(Frontier& frontier);
  void downsample(const vector<Vector3d>& cluster_in, vector<Vector3d>& cluster_out);
  int countVisibleCells(const Vector3d& pos, const double yaw, const vector<Vector3d>& c_in, vector<Vector3d>& c_out);

  bool isNearUnknown(const Vector3d& pos);
  bool isNearKnown(const Vector3d& pos);
  vector<Eigen::Vector3i> sixNeighbors(const Eigen::Vector3i& voxel);
  vector<Eigen::Vector3i> tenNeighbors(const Eigen::Vector3i& voxel);
  vector<Eigen::Vector3i> allNeighbors(const Eigen::Vector3i& voxel);
  bool isNeighborUnknown(const Eigen::Vector3i& voxel);
  void expandFrontier(const Eigen::Vector3i& first /* , const int& depth, const int& parent_id */);

  // Wrapper of sdf map
  int toadr(const Eigen::Vector3i& idx);
  bool knownfree(const Eigen::Vector3i& idx);
  bool inmap(const Eigen::Vector3i& idx);
};

}  // namespace perception_aware_planner
#endif