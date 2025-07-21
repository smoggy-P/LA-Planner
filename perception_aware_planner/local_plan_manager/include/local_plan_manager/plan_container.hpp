#ifndef PERCEPTION_AWARE_PLAN_CONTAINER_HPP_
#define PERCEPTION_AWARE_PLAN_CONTAINER_HPP_

#include "gcopter/gcopter.hpp"

#include "traj_utils/non_uniform_bspline.h"
#include "traj_utils/polynomial_traj.h"

#include <ros/ros.h>

#include <Eigen/Eigen>

namespace perception_aware_planner {

/* Parameters for Local Planning  */
struct PlanParameters {

  double max_vel_, max_acc_;

  double max_yawdot_;
  double ctrl_pt_dist;
  int bspline_degree_;

  double min_observed_ratio_;
  bool adjust_end_state_;
};

/* Info of generated local traj */
struct LocalTrajData {
  int traj_id_ = 0;
  double replan_begin_time_ = 0.0;
  double replan_stop_time_ = 0.0;
  double duration_;
  ros::Time start_time_;
  Eigen::Vector3d start_pos_, end_pos_;
  double end_yaw_;

  NonUniformBspline position_traj_, velocity_traj_, acceleration_traj_;
  Trajectory<5> yaw_traj_;
};

}  // namespace perception_aware_planner

#endif