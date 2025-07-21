#ifndef _PERCEPTION_AWARE_EXPLORATION_FSM_H_
#define _PERCEPTION_AWARE_EXPLORATION_FSM_H_

#include "exploration_manager/expl_data.h"
#include "exploration_manager/perception_aware_exploration_manager.h"

#include "local_plan_manager/plan_container.hpp"

#include "traj_utils/MixTraj.h"
#include "traj_utils/planning_visualization.h"

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/Marker.h>

using Eigen::Vector3d;
using std::pair;
using std::shared_ptr;
using std::string;
using std::unique_ptr;
using std::vector;

namespace perception_aware_planner {

class LocalPlanner;
class FailureDetector;
class PAExplorationManager;
class PlanningVisualization;
struct FSMParam;

enum FSM_EXEC_STATE { INIT, WAIT_TARGET, START_IN_STATIC, PUB_TRAJ, MOVE_TO_NEXT_GOAL, REPLAN, RETRY, TASK_FAIL };

enum TARGET_TYPE { MANUAL_TARGET = 1, PRESET_TARGET = 2 };

enum VISUAL_YAW_TYPE { FOV = 1, ARROW = 2 };

enum START_STATE_TYPE { LAST_TRAJ, ODOM };

enum REPLAN_REASON { REACH_TMP, CLUSTER_COVER, TIME_OUT, COLLISION_CHECK, NO_REPLAN };

enum ERROR_CODE { NOVIEWPOINT, LOCALIZATION, COLLISION };

class PAExplorationFSM : public std::enable_shared_from_this<PAExplorationFSM> {
public:
  shared_ptr<LocalPlanner> planner_manager_ = nullptr;
  shared_ptr<FailureDetector> failure_detector_ = nullptr;
  shared_ptr<PAExplorationManager> expl_manager_ = nullptr;
  shared_ptr<PlanningVisualization> visualization_ = nullptr;

  shared_ptr<FSMParam> fp_ = nullptr;

  bool task_start_ = false;

  // FSM data
  FSM_EXEC_STATE exec_state_ = FSM_EXEC_STATE::INIT;
  bool have_target_ = false;
  bool have_odom_ = false;
  vector<string> state_str_;

  Eigen::Vector3d odom_pos_, odom_vel_;  // odometry state
  Eigen::Quaterniond odom_orient_;
  double odom_yaw_;

  Eigen::Vector3d start_pos_, start_vel_, start_acc_, start_yaw_;  // start state
  Eigen::Vector3d end_pt_, end_vel_;                               // end state

  std::shared_ptr<LocalTrajData> last_traj_ = nullptr;
  traj_utils::MixTraj last_traj_msg_;
  Eigen::Vector3d last_vp_ = Eigen::Vector3d::Zero();

  Eigen::Vector3d final_goal_;

  int target_type_;  // 1 mannual select, 2 hard code
  double waypoints_[50][3];
  int waypoint_num_;
  int current_wp_ = 0;
  int stop_count_ = 0;
  int visual_yaw_type_;

  /* Debug utils */
  ERROR_CODE error_code_;
  vector<string> error_code_str_;

  /* ROS utils */
  ros::NodeHandle node_;
  ros::Timer exec_timer_, safety_timer_, frontier_timer_;
  ros::Publisher replan_pub_, traj_pub_, emergency_stop_pub_, triggle_map_pub_;
  ros::Subscriber odom_sub_, waypoint_sub_;

  std_msgs::Float64 replan_msg_;

  /* Reciprocating Operation */
  std::string end_state_file_path;
  vector<Eigen::Vector3d> tmp_vp_path;
  vector<double> tmp_vp_path_yaw;
  bool en_explore_check_;
  bool last_fail = false;

  /* helper functions */
  void callExplorationPlanner();
  void transitState(const FSM_EXEC_STATE new_state, const string& pos_call);

  /* ROS functions */
  void FSMCallback(const ros::TimerEvent& e);
  void safetyCallback(const ros::TimerEvent& e);
  void frontierCallback(const ros::TimerEvent& e);

  // Subscriber callbacks
  void odometryCallback(const nav_msgs::OdometryConstPtr& msg);
  void waypointCallback(const nav_msgs::PathConstPtr& msg);

  void visualize();

  void init(ros::NodeHandle& nh);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

public:
  // Variables for choose viewpoint
  VIEWPOINT_CHANGE_REASON last_fail_reason = VIEWPOINT_CHANGE_REASON::NONE;
  Eigen::Vector3d origin_pos_;

  // Function
  bool transitViewpoint();
  void transformViewpointFormat(
      const vector<Eigen::Vector3d>& temp_path, const vector<double>& temp_yaw, Eigen::Vector3d& point, vector<double>& yaw_vec_);
  void setStartState(START_STATE_TYPE replan_switch);

  void chooseBestViewpoint();
  double setStopTimeReplan();
  double setStopTimeCollision(const double& t_colli);
  void informReplan(const double& t_colli = -1.0);
};
}  // namespace perception_aware_planner

#endif