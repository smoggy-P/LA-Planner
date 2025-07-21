#ifndef PERCEPTION_AWARE_FAILURE_DETECTOR_H
#define PERCEPTION_AWARE_FAILURE_DETECTOR_H

#include "local_plan_manager/perception_aware_planner_manager.h"

namespace perception_aware_planner {

class FeatureMap;

class FailureDetector {
public:
  FailureDetector(ros::NodeHandle& nh);
  ~FailureDetector();

  void setFeatureMap(const shared_ptr<FeatureMap>& feature_map) {
    feature_map_ = feature_map;
  }

  bool checkSingleFrameVisibility(const Eigen::Vector3d& pos, const Eigen::Quaterniond& orient);
  bool checkRealTime(const double timestamp, const Eigen::Vector3d& pos, const Eigen::Quaterniond& orient);
  bool checkTraj(const LocalTrajData& local_traj);

private:
  FeatureMap::Ptr feature_map_ = nullptr;

  std::string output_path_;
  std::ofstream f_realtime_odom_;
  std::ofstream f_realtime_vis_;
  std::ofstream f_traj_odom_;

  struct Param {
    double max_tol_time_;
    double min_check_freq_;
  };

  Param param_;

  deque<pair<double, set<int>>> slide_window_act_;

  void saveTraj_TUM(std::ofstream& fout, const double timestamp, const Vector3d& p, const Quaterniond& q);
  void saveTraj_Vis(std::ofstream& fout, const double timestamp, const int vis_num, const int covis_num);
};

}  // namespace perception_aware_planner

#endif