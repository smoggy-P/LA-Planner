#ifndef PERCEPTION_AWARE_EXPL_DATA_H_
#define PERCEPTION_AWARE_EXPL_DATA_H_

#include "traj_utils/Bspline.h"

#include <Eigen/Eigen>

#include <vector>

using Eigen::Vector3d;
using std::string;
using std::vector;

namespace perception_aware_planner {

struct FSMParam {
  bool auto_trigger_;
  double replan_thresh1_;
  double replan_thresh2_;
  double replan_thresh3_;
  double replan_thresh4_;
  double replan_time_;
  double replan_out_preset_;
};

enum NEXT_GOAL_TYPE { REACH_FINAL_GOAL, GOTO_FINAL_GOAL, TMP_VIEWPOINT, LOCAL_PLAN_FAIL };

struct NextGoalData {
  NEXT_GOAL_TYPE type_;
  Vector3d pos_;
  vector<double> yaw_vec_;
  double yaw_;
  vector<Vector3d> frontier_cell_;
};

struct ExplorationData {
  vector<vector<Vector3d>> frontiers_;
  vector<vector<Vector3d>> dead_frontiers_;

  vector<Vector3d> points_;
  vector<Vector3d> views_;
  vector<double> yaws_;
  vector<size_t> visb_num_;

  vector<int> frontier_ids_;

  vector<vector<Vector3d>> frontier_cells_;

  Vector3d next_goal_;
  vector<Vector3d> path_next_goal_;

  Vector3d point_now;
  vector<double> yaw_vector;
  vector<Vector3d> frontier_now;
};

}  // namespace perception_aware_planner

#endif