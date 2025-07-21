#ifndef PERCEPTION_AWARE_YAW_INITIAL_PLANNER_H_
#define PERCEPTION_AWARE_YAW_INITIAL_PLANNER_H_

#include "plan_env/feature_map.h"
#include "plan_env/sdf_map.h"

#include "utils/utils.h"

#include <ros/ros.h>

#include <memory>
#include <vector>

namespace perception_aware_planner {

// vertex type for yaw initial planning
class YawVertex {
public:
  using Ptr = shared_ptr<YawVertex>;

  YawVertex(const double y, const size_t layer) {
    yaw_ = y;
    Utils::roundPi(yaw_);
    layer_ = layer;
  }

  double calEdgeDiff(const double yaw) const {
    double diff = fabs(yaw - yaw_);
    return std::min(diff, 2 * M_PI - diff);
  }

  double calEdgeDiff(const YawVertex::Ptr& v) const {
    return calEdgeDiff(v->yaw_);
  }

  vector<YawVertex::Ptr> edges_;
  YawVertex::Ptr parent_ = nullptr;

  int candiate_parent_num_ = 0;     // Number of candidate parent nodes
  int graph_id_;                    // ID in graph search
  size_t layer_;                    // Layer in the graph
  double g_value_;                  // g_score for graph search
  double yaw_;                      // Yaw angle
  int yaw_id_;                      // ID of the yaw angle
  vector<int> features_id_;         // IDs of features visible to this node
  set<int> frontiers_id_;           // IDs of frontiers visible to this node
  set<int> frontiers_id_path_;      // IDs of frontiers visible along the path to this node
  bool if_vis_final_goal_ = false;  // Whether the final goal is visible
};

class CovisibleCostComparator {
public:
  bool operator()(const pair<int, double>& a, const pair<int, double>& b) {
    return a.second < b.second;
  }
};

class YawInitialPlanner {
public:
  using Ptr = shared_ptr<YawInitialPlanner>;

  YawInitialPlanner(ros::NodeHandle& nh);

  void setFinalGoal(const Vector3d& final_goal) {
    final_goal_ = final_goal;
  }

  void setIfPlan2FinalGoal(const bool flag) {
    go2final_ = flag;
  }

  void setTargetFrontier(const vector<Vector3d>& target_frontier) {
    target_frontier_ = target_frontier;
    preprocessFrontier();
  }

  void setSDFmap(shared_ptr<SDFMap> sdf_map) {
    sdf_map_ = sdf_map;
  }

  void setFeatureMap(const shared_ptr<FeatureMap>& feature_map) {
    feature_map_ = feature_map;
  }

  void setPos(const vector<Vector3d>& pos) {
    pos_ = pos;
  }

  void setAcc(const vector<Vector3d>& acc) {
    acc_ = acc;
    // Since the drone will be forced to hover at the end of the trajectory, set it to zero
    acc_.back().setZero();
  }

  void preprocessFrontier();

  void reset();
  bool search(const double start_yaw, const vector<double>& end_yaw_vec, const double& dt, vector<double>& path);

  void prepareOptData(const YawOptData::Ptr& data);

private:
  struct Param {
    int piece_num_;
    vector<double> yaw_samples_;
    double ld_smoothness_;
    double ld_expl_;
    double ld_frontier_;
    double ld_final_goal_;
    double max_yaw_rate_;
    double basic_cost_;
    int max_diff_yaw_id_;
    double dt_;
    int min_feature_num_;
    int min_covisible_feature_num_;
  };

  Param param_;

  // Remember to reset before each search
  int graph_id_ = 0;
  vector<YawVertex::Ptr> vertice_;
  vector<YawVertex::Ptr> vert_path_;

  YawVertex::Ptr start_vert_ = nullptr;
  YawVertex::Ptr end_vert_ = nullptr;
  vector<YawVertex::Ptr> end_vert_vec_;

  CameraParam::Ptr frontier_cam_ = nullptr;
  CameraParam::Ptr feature_cam_ = nullptr;
  shared_ptr<FeatureMap> feature_map_ = nullptr;

  shared_ptr<SDFMap> sdf_map_ = nullptr;
  vector<Vector3d> target_frontier_;
  vector<vector<int>> target_frontier_aft_preprocess_;
  vector<Vector3d> pos_;
  vector<Vector3d> acc_;
  Vector3d final_goal_;
  bool go2final_;

  /*Function*/
  void yaw2id(const YawVertex::Ptr& node);
  void id2yaw(const YawVertex::Ptr& node);
  void addVertex(const YawVertex::Ptr& vertex);

  double calEdgeCost(const YawVertex::Ptr& from, const YawVertex::Ptr& to);

  bool checkFeasibility(const YawVertex::Ptr& v1, const YawVertex::Ptr& v2);

  double calcVisibility(const YawVertex::Ptr& v, const int id);
  double calcCoVisibility(const YawVertex::Ptr& v1, const YawVertex::Ptr& v2, const int id);
  void selectTargetCoVisibleID(const YawVertex::Ptr& v1, const YawVertex::Ptr& v2, const set<int>& s_in, set<int>& s_out);

  void setVisbleFrontiers(const YawVertex::Ptr& v);
  void setVisbleFeatures(const YawVertex::Ptr& v);
  int getCoVisibleNum(const YawVertex::Ptr& v1, const YawVertex::Ptr& v2);
  void getCoVisibleSet(const YawVertex::Ptr& v1, const YawVertex::Ptr& v2, set<int>& co_vis_id);

  void checkIfVisGoal(const YawVertex::Ptr& v);
  bool refineEndYaw(YawVertex::Ptr& v);
  bool handleEndLayer();
  bool graphSearch();
};

}  // namespace perception_aware_planner

#endif