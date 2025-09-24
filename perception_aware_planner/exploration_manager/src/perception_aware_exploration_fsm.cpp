#include "exploration_manager/perception_aware_exploration_fsm.h"

#include "local_plan_manager/perception_aware_planner_manager.h"

#include "traj_utils/planning_visualization.h"

#include "utils/utils.h"

#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Int32MultiArray.h>

#include <thread>

#include <queue>

namespace perception_aware_planner {

void PAExplorationFSM::init(ros::NodeHandle& nh) {
  fp_.reset(new FSMParam);

  /*  Fsm param  */
  nh.param("fsm/flight_type", target_type_, -1);
  nh.param("fsm/auto_trigger", fp_->auto_trigger_, false);
  nh.param("fsm/thresh_replan1", fp_->replan_thresh1_, -1.0);
  nh.param("fsm/thresh_replan2", fp_->replan_thresh2_, -1.0);
  nh.param("fsm/thresh_replan3", fp_->replan_thresh3_, -1.0);
  nh.param("fsm/thresh_replan4", fp_->replan_thresh4_, -1.0);
  nh.param("fsm/replan_time", fp_->replan_time_, -1.0);
  nh.param("fsm/replan_out", fp_->replan_out_preset_, -1.0);

  nh.param<std::string>("fsm/end_state_file_path", end_state_file_path, "");
  nh.param("fsm/wp_num", waypoint_num_, -1);
  for (int i = 0; i < waypoint_num_; i++) {
    nh.param("fsm/wp" + to_string(i) + "_x", waypoints_[i][0], -1.0);
    nh.param("fsm/wp" + to_string(i) + "_y", waypoints_[i][1], -1.0);
    nh.param("fsm/wp" + to_string(i) + "_z", waypoints_[i][2], -1.0);
  }

  nh.param("fsm/visual_yaw_type", visual_yaw_type_, -1);

  /* Initialize main modules */
  Utils::initialize(nh);
  visualization_.reset(new PlanningVisualization(nh));

  expl_manager_ = make_shared<PAExplorationManager>(shared_from_this());
  expl_manager_->initialize(nh);
  expl_manager_->frontier_finder_->setVisualization(visualization_);

  planner_manager_ = expl_manager_->planner_manager_;
  failure_detector_ = expl_manager_->failure_detector_;

  state_str_ = { "INIT", "WAIT_TARGET", "START_IN_STATIC", "PUB_TRAJ", "MOVE_TO_NEXT_GOAL", "REPLAN", "RETRY", "TASK_FAIL" };
  error_code_str_ = { "NOVIEWPOINT", "LOCALIZATION", "COLLISION" };

  /* Ros sub, pub and timer */
  exec_timer_ = nh.createTimer(ros::Duration(0.05), &PAExplorationFSM::FSMCallback, this);
  safety_timer_ = nh.createTimer(ros::Duration(0.05), &PAExplorationFSM::safetyCallback, this);
  frontier_timer_ = nh.createTimer(ros::Duration(0.1), &PAExplorationFSM::frontierCallback, this);

  odom_sub_ = nh.subscribe("/odom_world", 10000, &PAExplorationFSM::odometryCallback, this);
  if (!fp_->auto_trigger_)
    waypoint_sub_ = nh.subscribe("/waypoint_generator/waypoints", 1, &PAExplorationFSM::waypointCallback, this);

  // to map_publisher
  triggle_map_pub_ = nh.advertise<std_msgs::Empty>("/planning/triggle_map", 10);

  // to traj_server
  emergency_stop_pub_ = nh.advertise<std_msgs::Empty>("/planning/emergency_stop", 10);
  replan_pub_ = nh.advertise<std_msgs::Float64>("/planning/replan", 10);
  traj_pub_ = nh.advertise<traj_utils::MixTraj>("/planning/trajectory", 10);
}

void PAExplorationFSM::waypointCallback(const nav_msgs::PathConstPtr& msg) {
  if (target_type_ == TARGET_TYPE::MANUAL_TARGET && msg->poses[0].pose.position.z < -0.1) return;

  // if (stop_count_ == waypoint_num_) return;

  // if (exec_state_ != WAIT_TARGET) return;

  ROS_WARN("Receive Goal!!!");

  if (target_type_ == TARGET_TYPE::MANUAL_TARGET) {
    final_goal_(0) = msg->poses[0].pose.position.x;
    final_goal_(1) = msg->poses[0].pose.position.y;
    final_goal_(2) = 1.0;
  }

  else {
    final_goal_(0) = waypoints_[current_wp_][0];
    final_goal_(1) = waypoints_[current_wp_][1];
    final_goal_(2) = waypoints_[current_wp_][2];
    current_wp_ = (current_wp_ + 1);
  }

  visualization_->drawGoal(final_goal_, 0.3, Eigen::Vector4d(1, 0, 0, 1.0));

  end_vel_.setZero();
  have_target_ = true;
  expl_manager_->frontier_finder_->setFinalGoal(final_goal_);

  transitState(START_IN_STATIC, "TRIG");

  last_fail = false;
  stop_count_++;

  triggle_map_pub_.publish(std_msgs::Empty());
}

void PAExplorationFSM::FSMCallback(const ros::TimerEvent& e) {

  ROS_INFO_STREAM_THROTTLE(1.0, "[FSM]: state: " << state_str_[int(exec_state_)]);

  // ---------------- [ADD] Helper: publish nearest frontier as next waypoint ----------------
  auto publish_nearest_frontier = [this]() -> bool {
    if (!expl_manager_ || !expl_manager_->frontier_finder_) {
      ROS_ERROR("Explorer manager or frontier finder is null");
      return false;
    }
    auto cmp = [](const std::pair<double, Eigen::Vector3d>& a,
                  const std::pair<double, Eigen::Vector3d>& b) { return a.first < b.first; };
    std::priority_queue<
        std::pair<double, Eigen::Vector3d>,
        std::vector<std::pair<double, Eigen::Vector3d>>,
        decltype(cmp)
    > pq(cmp);

    ROS_INFO("calculating next target");
    std::vector<std::vector<Eigen::Vector3d>> active_frontiers;
    expl_manager_->frontier_finder_->getFrontiers(active_frontiers);

    int i = 0;
    for (const auto& viewpoints : active_frontiers) {
      for (const auto& vp : viewpoints) {
        ROS_INFO("viewpoint %d: [%.3f, %.3f, %.3f]", i++, vp(0), vp(1), vp(2));
        pq.emplace((vp - odom_pos_).norm(), vp);
      }
    }

    if (pq.empty()) {
      ROS_WARN("No feature viewpoint, wait for target");
      return false;
    }

    nav_msgs::Path target_path;
    target_path.header.frame_id = "map";
    target_path.header.stamp = ros::Time::now();

    geometry_msgs::PoseStamped pose;
    pose.header = target_path.header;
    pose.pose.position.x = pq.top().second.x();
    pose.pose.position.y = pq.top().second.y();
    pose.pose.position.z = pq.top().second.z();
    pose.pose.orientation.w = 1.0;
    target_path.poses.push_back(pose);

    ROS_INFO("next target: [%.3f, %.3f, %.3f]",
             pq.top().second.x(), pq.top().second.y(), pq.top().second.z());
    waypointCallback(boost::make_shared<nav_msgs::Path>(target_path));
    return true;
  };

  // ---------------- [ADD] Watchdogs & throttles (local static state) ----------------
  static ros::Time last_vis_tick;
  static int poor_vis_cnt = 0;
  static int stuck_cnt = 0;
  static double last_t_cur = -1.0;
  static Eigen::Vector3d last_pos = Eigen::Vector3d::Zero();

  switch (exec_state_) {
    case INIT: {
      if (!have_odom_) {
        ROS_WARN_THROTTLE(1.0, "No Odom.");
        return;
      }
      if (expl_manager_->ed_->frontier_now.empty()) {
        ROS_WARN_THROTTLE(1.0, "No Frontier.");
        return;
      }
      transitState(WAIT_TARGET, "FSM");
      break;
    }

    case WAIT_TARGET: {

      if (stop_count_ == waypoint_num_) {
        ROS_WARN_THROTTLE(1.0, "Task Success!!!");
        if (!publish_nearest_frontier()) {
          // 保持等待，等待 frontiers 更新
        }
        break;
      }

      else if (fp_->auto_trigger_) {
        ROS_WARN_THROTTLE(1.0, "Wait For Target...");
        static int count = 0;
        if (odom_vel_.norm() < 1e-2) count++; else count = 0;
        if (count > 10) {
          nav_msgs::Path empty_path;
          waypointCallback(boost::make_shared<nav_msgs::Path>(empty_path));
          count = 0;
        }
        break;
      }

      else {
        ROS_WARN_THROTTLE(1.0, "Wait For Target...");
        (void)publish_nearest_frontier();
        break;
      }
    }

    case START_IN_STATIC: {
      task_start_ = true;
      static bool first_enter_start_ = true;
      if (first_enter_start_) {
        chooseBestViewpoint();
        origin_pos_ = expl_manager_->ngd_->pos_;
        expl_manager_->frontier_finder_->resetViewpointManager();
        first_enter_start_ = false;
        expl_manager_->setEnableExploreCheck(true);
      }

      setStartState(ODOM);
      callExplorationPlanner();

      switch (expl_manager_->ngd_->type_) {
        case REACH_FINAL_GOAL:
        case GOTO_FINAL_GOAL:
        case TMP_VIEWPOINT: {
          last_fail_reason = NONE;
          transitState(PUB_TRAJ, "FSM");
          first_enter_start_ = true;
          break;
        }
        case LOCAL_PLAN_FAIL: {
          ROS_ERROR("This viewpoint is not availabe-----------------------------");
          if (!transitViewpoint()) {
            ROS_ERROR("REPLAN_FAIL!!!!!!! EMERGENCY_STOP!!!!");
            transitState(RETRY, "FSM");
          }
          break;
        }
      }
      break;
    }

    case PUB_TRAJ: {
      // ---------------- [ADD] Null / invalid guard ----------------
      if (!last_traj_) {
        ROS_ERROR("No last_traj_ in PUB_TRAJ, force REPLAN");
        transitState(REPLAN, "FSM");
        break;
      }
      double dt = (ros::Time::now() - last_traj_->start_time_).toSec();
      if (!std::isfinite(dt) || dt < 0.0) {
        ROS_WARN_THROTTLE(1.0, "Invalid traj start time (dt=%.3f), force REPLAN", dt);
        transitState(REPLAN, "FSM");
        break;
      }

      if (dt > 0) {
        traj_pub_.publish(last_traj_msg_);
        visualization_->clearUnreachableMarker();
        // 重置看门狗基线
        last_t_cur = -1.0;
        stuck_cnt = 0;
        poor_vis_cnt = 0;
        last_pos = odom_pos_;
        transitState(MOVE_TO_NEXT_GOAL, "FSM");
      }
      break;
    }

    case MOVE_TO_NEXT_GOAL: {
      last_fail = false;
      const auto& type = expl_manager_->ngd_->type_;
      ROS_ASSERT(type != LOCAL_PLAN_FAIL);

      // ---------------- [MOD] 可视化线程节流 ----------------
      if ((ros::Time::now() - last_vis_tick).toSec() > 0.5) {
        thread vis_thread(&PAExplorationFSM::visualize, this);
        vis_thread.detach();
        last_vis_tick = ros::Time::now();
      }

      // ---------------- [MOD] 可见性看门狗 ----------------
      if (!failure_detector_->checkSingleFrameVisibility(odom_pos_, odom_orient_)) {
        poor_vis_cnt++;
        ROS_WARN_THROTTLE(1.0, "[Replan Gate] Poor visibility (%d)", poor_vis_cnt);
        if (poor_vis_cnt > 10) { // 约 ~1s（取决于回调频率）
          ROS_WARN("[Replan]: Visibility stuck -> REPLAN");
          poor_vis_cnt = 0;
          transitState(REPLAN, "FSM");
        }
        break; // 先退出本轮
      } else {
        poor_vis_cnt = 0;
      }

      visualization_->fail_reason = last_fail_reason;

      LocalTrajData* info = &planner_manager_->local_data_;
      double t_cur = (ros::Time::now() - info->start_time_).toSec();
      double duration = info->duration_;

      // ---------------- [ADD] 时间健壮性与卡死看门狗 ----------------
      if (!std::isfinite(t_cur) || !std::isfinite(duration) || duration <= 0.0) {
        ROS_WARN("[Replan]: Invalid traj timing (t_cur=%.3f, duration=%.3f) -> REPLAN",
                 t_cur, duration);
        transitState(REPLAN, "FSM");
        break;
      }
      if (t_cur < 0.0) { // 仿真时间回退
        ROS_WARN("[Replan]: Negative t_cur (%.3f), sim time glitch -> REPLAN", t_cur);
        transitState(REPLAN, "FSM");
        break;
      }

      // 进度看门狗：时间不前进 & 位置不前进
      const bool time_stuck = (last_t_cur >= 0.0 && (t_cur - last_t_cur) < 1e-3);
      const bool pos_stuck  = ((odom_pos_ - last_pos).norm() < 0.02 && odom_vel_.norm() < 1e-2);
      if (time_stuck && pos_stuck) {
        stuck_cnt++;
        ROS_WARN_THROTTLE(1.0, "[Replan]: Progress stuck cnt=%d (t_cur=%.3f)", stuck_cnt, t_cur);
        if (stuck_cnt > 50) { // ~5s
          ROS_WARN("[Replan]: Progress watchdog -> REPLAN");
          stuck_cnt = 0;
          transitState(REPLAN, "FSM");
          break;
        }
      } else {
        stuck_cnt = 0;
        last_pos = odom_pos_;
        last_t_cur = t_cur;
      }

      double time_to_end = duration - t_cur;

      // ---------------- [MOD] 原有 replan 条件保持，但已确保 t_cur/时间有效 ----------------
      if (time_to_end < fp_->replan_thresh1_) {
        if (type == REACH_FINAL_GOAL) {
          transitState(WAIT_TARGET, "FSM");
          ROS_WARN("[Replan]: Reach final goal=================================");
        } else {
          transitState(REPLAN, "FSM");
          ROS_WARN("[Replan]: Reach tmp viewpoint=================================");
        }
      }

      else if (type == REACH_FINAL_GOAL) {
        ROS_WARN_THROTTLE(1.0, "[Replan]: Final final goal,reject to replan====================");
      }

      else if (t_cur > fp_->replan_thresh2_ &&
               expl_manager_->frontier_finder_->isinterstFrontierCovered(expl_manager_->ngd_->frontier_cell_)) {
        transitState(REPLAN, "FSM");
        ROS_WARN("[Replan]: Cluster covered=====================================");
      }

      else if (t_cur > fp_->replan_thresh3_) {
        transitState(REPLAN, "FSM");
        ROS_WARN("[Replan]: Periodic call=================================");
      }

      break;
    }

    case REPLAN: {
      static bool first_enter_replan_ = true;
      if (first_enter_replan_ && last_fail_reason != COLLISION_CHECK_FAIL) {

        // Notify the traj server
        informReplan();

        // Select the latest viewpoint and corresponding frontier
        chooseBestViewpoint();

        expl_manager_->frontier_finder_->resetViewpointManager();

        first_enter_replan_ = false;
        origin_pos_ = expl_manager_->ngd_->pos_;
        expl_manager_->setEnableExploreCheck(true);
      }

      setStartState(LAST_TRAJ);
      callExplorationPlanner();

      switch (expl_manager_->ngd_->type_) {
        case REACH_FINAL_GOAL:
        case GOTO_FINAL_GOAL:
        case TMP_VIEWPOINT: {
          last_fail_reason = NONE;

          transitState(PUB_TRAJ, "FSM");
          first_enter_replan_ = true;

          break;
        }

        case LOCAL_PLAN_FAIL: {
          ROS_ERROR("This viewpoint is not availabe-----------------------------");
          if (!transitViewpoint()) {
            ROS_ERROR("Fail to transit viewpoint, turn to sample viewpoint");
            transitState(RETRY, "FSM");
          }
          break;
        }
      }
      break;
    }

    case RETRY: {
      if (odom_vel_.norm() > 0.01) {
        ROS_WARN_THROTTLE(1.0, "Wait for stop");
        return;
      }

      if (!last_fail) {
        expl_manager_->frontier_finder_->resetViewpointManager();
        transitState(REPLAN, "FSM");
        last_fail = true;
        break;
      }

      // Fail!!! Reason: No available viewpoint
      error_code_ = NOVIEWPOINT;
      transitState(TASK_FAIL, "FSM");
      break;
    }

    case TASK_FAIL: {
      ROS_WARN_THROTTLE(1.0, "Task failed!!!Error type: %s", error_code_str_[int(error_code_)].c_str());
      break;
    }
  }
}

void PAExplorationFSM::callExplorationPlanner() {
  ros::Time time_r = ros::Time::now() + ros::Duration(fp_->replan_time_);

  expl_manager_->setEnableExploreCheck(en_explore_check_);
  expl_manager_->selectNextGoal();

  auto& ngd = expl_manager_->ngd_;

  switch (ngd->type_) {
    case REACH_FINAL_GOAL:
      ROS_WARN("[callExplorationPlanner]: Reach final goal!!");
      break;

    case GOTO_FINAL_GOAL:
      ROS_WARN("[callExplorationPlanner]: Move to final goal!!");
      break;

    case TMP_VIEWPOINT:
      ROS_WARN("[callExplorationPlanner]: Move to tmp viewpoint!!");
      break;

    case LOCAL_PLAN_FAIL:
      ROS_ERROR("[callExplorationPlanner]: Fail to gen local traj!!");
      return;
  }

  auto info = &planner_manager_->local_data_;
  info->start_time_ = (ros::Time::now() - time_r).toSec() > 0 ? ros::Time::now() : time_r;
  info->replan_begin_time_ = -1.0;
  info->replan_stop_time_ = -1.0;

  last_traj_msg_ = planner_manager_->generateROSMsg();
  last_traj_.reset(new LocalTrajData(planner_manager_->local_data_));
}

void PAExplorationFSM::visualize() {
  auto& ngd = expl_manager_->ngd_;

  Vector4d traj_color(0.0, 1.0, 0.0, 0.5);  // green
  // Hybrid A* path
  // visualization_->drawGeometricPath(planner_manager_->kino_path_, 0.075, Eigen::Vector4d(1, 1, 0, 0.4));
  // pos traj
  visualization_->drawBspline(last_traj_->position_traj_, last_traj_->duration_, 0.05, traj_color, false, 0.15, traj_color);
  // yaw traj
  vector<Vector3d> pos_vec, acc_vec, yaw_vec;
  vector<double> boundary_vec;
  planner_manager_->getYawTrajForVis(pos_vec, acc_vec, yaw_vec, boundary_vec);
  if (visual_yaw_type_ == FOV)
    visualization_->drawYawFOVTraj(pos_vec, acc_vec, yaw_vec, traj_color);
  else if (visual_yaw_type_ == ARROW) {
    visualization_->drawYawArrow(pos_vec, acc_vec, yaw_vec);
    visualization_->drawYawCorridor(pos_vec, acc_vec, yaw_vec, boundary_vec, traj_color);
  }

  //  next viewpoint
  traj_color(3) = 1.0;
  visualization_->drawTargetViewpoint(ngd->pos_, ngd->yaw_, traj_color);
}

void PAExplorationFSM::frontierCallback(const ros::TimerEvent& e) {
  static int delay = 0;
  if (++delay < 5) {
    return;
  }

  auto ft = expl_manager_->frontier_finder_;
  auto ed = expl_manager_->ed_;

  vector<double> score;

  // The frequency of the frontier thread is 10Hz(at most)
  // Estimate the next position and velocity to prevent the start state of graph search is too far from the real state
  double yaw_next;
  Vector3d pos_next, vel_next;
  if (last_traj_ == nullptr) {
    pos_next = odom_pos_;
    vel_next = odom_vel_;
    yaw_next = odom_yaw_;
  }

  else {
    double t_next = (ros::Time::now() - last_traj_->start_time_).toSec() + 0.1;
    if (t_next > last_traj_->start_time_.toSec() + last_traj_->duration_ || exec_state_ == REPLAN) {
      pos_next = odom_pos_;
      vel_next = odom_vel_;
      yaw_next = odom_yaw_;
    }
    pos_next = last_traj_->position_traj_.evaluateDeBoorT(t_next);
    vel_next = last_traj_->velocity_traj_.evaluateDeBoorT(t_next);
    yaw_next = last_traj_->yaw_traj_.getPos(t_next)[0];
  }

  ft->getShareFrontierParam(
      pos_next, vel_next, yaw_next, ed->frontiers_, ed->dead_frontiers_, tmp_vp_path, tmp_vp_path_yaw, ed->frontier_now, score);
  transformViewpointFormat(tmp_vp_path, tmp_vp_path_yaw, ed->point_now, ed->yaw_vector);
}

void PAExplorationFSM::safetyCallback(const ros::TimerEvent& e) {
  if (!have_odom_ || !expl_manager_->global_sdf_map_->hasInitialized_ || !have_target_) {
    return;
  }

  if (exec_state_ == FSM_EXEC_STATE::MOVE_TO_NEXT_GOAL) {
    // Check safety and trigger replan if necessary
    double dist;
    double t_collision;
    if (!planner_manager_->checkTrajCollision(dist, t_collision)) {
      ROS_WARN("[Replan]: Collision detected==================================");
      last_fail_reason = COLLISION_CHECK_FAIL;
      informReplan(t_collision);
      transitViewpoint();
      transitState(REPLAN, "safetyCallback");
    }
  }
}

void PAExplorationFSM::odometryCallback(const nav_msgs::OdometryConstPtr& msg) {
  odom_pos_(0) = msg->pose.pose.position.x;
  odom_pos_(1) = msg->pose.pose.position.y;
  odom_pos_(2) = msg->pose.pose.position.z;

  odom_vel_(0) = msg->twist.twist.linear.x;
  odom_vel_(1) = msg->twist.twist.linear.y;
  odom_vel_(2) = msg->twist.twist.linear.z;

  odom_orient_.w() = msg->pose.pose.orientation.w;
  odom_orient_.x() = msg->pose.pose.orientation.x;
  odom_orient_.y() = msg->pose.pose.orientation.y;
  odom_orient_.z() = msg->pose.pose.orientation.z;

  Vector3d rot_x = odom_orient_.toRotationMatrix().block<3, 1>(0, 0);
  odom_yaw_ = atan2(rot_x(1), rot_x(0));

  have_odom_ = true;

  // visuzlize feature num
  int feature_num = expl_manager_->feature_map_->getFeatureUsingOdom(msg);
  visualization_->drawFeatureNum(feature_num, Utils::getGlobalParam().min_feature_num_act_, odom_pos_);

  if (exec_state_ == TASK_FAIL) return;

  // check the localization status
  // if (task_start_ && !failure_detector_->checkRealTime(msg->header.stamp.toSec(), odom_pos_, odom_orient_)) {
  //   emergency_stop_pub_.publish(std_msgs::Empty());
  //   // Fail!!! Reason: Fail to localize
  //   error_code_ = LOCALIZATION;
  //   transitState(TASK_FAIL, "odomCallback");
  //   return;
  // }

  // check the collision status
  // if (task_start_ && planner_manager_->isCollision(odom_pos_)) {
  //   ROS_ERROR("[odometryCallback]: Collision detected==================================");
  //   emergency_stop_pub_.publish(std_msgs::Empty());
  //   // Fail!!! Reason: Collision
  //   error_code_ = COLLISION;
  //   transitState(TASK_FAIL, "odomCallback");
  //   return;
  // }
}

void PAExplorationFSM::transitState(const FSM_EXEC_STATE new_state, const string& pos_call) {
  int pre_s = int(exec_state_);
  exec_state_ = new_state;
  cout << "[" + pos_call + "]: from " + state_str_[pre_s] + " to " + state_str_[int(new_state)] << endl;
}

bool PAExplorationFSM::transitViewpoint() {
  auto ed = expl_manager_->ed_;
  auto ft = expl_manager_->frontier_finder_;
  auto info = &planner_manager_->local_data_;

  visualization_->fail_reason = last_fail_reason;

  auto& ngd = expl_manager_->ngd_;
  visualization_->drawFrontiersUnreachable(
      ngd->frontier_cell_, ngd->pos_, ngd->yaw_, planner_manager_->kino_path_, info->position_traj_, info->duration_);

  double length2best = (ed->point_now - origin_pos_).norm();
  double diff_last_vp = (ed->point_now - last_vp_).norm();

  if (length2best > fp_->replan_thresh4_ && diff_last_vp > fp_->replan_thresh4_) {
    chooseBestViewpoint();
    last_vp_ = expl_manager_->ngd_->pos_;
  }

  else {

    if (!ft->chooseNextViewpoint(tmp_vp_path, tmp_vp_path_yaw, ngd->frontier_cell_)) {
      expl_manager_->frontier_finder_->resetViewpointManager();  // Reset the selection of viewpoint
      ROS_ERROR("[chooseNextViewpoint] ----------------------------------------------------------\n"
                "ERROR !!!NO AVAILABLE FRONTIER!!"
                "--------------------------------------------------------------------------------");
      return false;
    }

    transformViewpointFormat(tmp_vp_path, tmp_vp_path_yaw, ngd->pos_, ngd->yaw_vec_);
  }

  return true;
}

double PAExplorationFSM::setStopTimeReplan() {
  auto& local_data = planner_manager_->local_data_;

  double t_pass = local_data.replan_begin_time_ - last_traj_->start_time_.toSec();

  double t_stop = t_pass + fp_->replan_time_ + fp_->replan_out_preset_;
  t_stop = min(t_stop, last_traj_->duration_);

  const double time_step = 0.05;

  while (t_stop < last_traj_->duration_) {
    const Vector3d fut_pos = last_traj_->position_traj_.evaluateDeBoorT(t_stop);
    const Vector3d fut_acc = last_traj_->acceleration_traj_.evaluateDeBoorT(t_stop);
    const double fut_yaw = last_traj_->yaw_traj_.getPos(t_stop)[0];
    Quaterniond fut_ori = Utils::calcOrientation(fut_yaw, fut_acc);

    int feature_num = expl_manager_->feature_map_->getFeatureUsingOdom(fut_pos, fut_ori);
    if (feature_num > Utils::getGlobalParam().min_feature_num_plan_) {
      return t_stop;
    }

    t_stop += time_step;
  }

  return last_traj_->duration_;
}

double PAExplorationFSM::setStopTimeCollision(const double& t_collision) {
  auto& local_data = planner_manager_->local_data_;

  double t_pass = local_data.replan_begin_time_ - last_traj_->start_time_.toSec();
  double t_stop = t_pass + fp_->replan_time_ + fp_->replan_out_preset_;

  double t_search_start = t_pass;
  double t_search_end = min(t_collision, t_stop);

  const double time_step = 0.05;

  for (double t = t_search_end; t > t_search_start; t -= time_step) {
    const Vector3d fut_pos = last_traj_->position_traj_.evaluateDeBoorT(t);
    const Vector3d fut_acc = last_traj_->acceleration_traj_.evaluateDeBoorT(t);
    const double fut_yaw = last_traj_->yaw_traj_.getPos(t)[0];
    Quaterniond fut_ori = Utils::calcOrientation(fut_yaw, fut_acc);

    int feature_num = expl_manager_->feature_map_->getFeatureUsingOdom(fut_pos, fut_ori);
    if (feature_num > Utils::getGlobalParam().min_feature_num_plan_) {
      return t;
    }
  }

  return t_search_start + 1e-3;
}

void PAExplorationFSM::setStartState(START_STATE_TYPE replan_switch) {

  double t_r;

  if (last_traj_ == nullptr) {
    replan_switch = ODOM;
  }

  else {
    ros::Time t_now = ros::Time::now();
    t_r = (t_now - last_traj_->start_time_).toSec() + fp_->replan_time_;

    auto& local_data = planner_manager_->local_data_;

    if (odom_vel_.norm() < 0.01 || t_r > local_data.replan_stop_time_) {
      replan_switch = ODOM;
    }
  }

  if (replan_switch == ODOM) {
    ROS_INFO("[PAExplorationFSM::setStartState]: Start from odom");
    ROS_INFO_STREAM("Odom pos: " << odom_pos_.transpose() << " yaw: " << odom_yaw_);
    start_pos_ = odom_pos_;
    start_vel_ = odom_vel_;
    start_acc_.setZero();
    start_yaw_.setZero();
    start_yaw_(0) = odom_yaw_;
  }

  else {
    ROS_INFO("[PAExplorationFSM::setStartState]: Start from last traj");
    start_pos_ = last_traj_->position_traj_.evaluateDeBoorT(t_r);
    start_vel_ = last_traj_->velocity_traj_.evaluateDeBoorT(t_r);
    start_acc_ = last_traj_->acceleration_traj_.evaluateDeBoorT(t_r);
    start_yaw_(0) = last_traj_->yaw_traj_.getPos(t_r)[0];
    start_yaw_(1) = last_traj_->yaw_traj_.getVel(t_r)[0];
    start_yaw_(2) = last_traj_->yaw_traj_.getAcc(t_r)[0];
  }
}

void PAExplorationFSM::chooseBestViewpoint() {
  auto& ngd = expl_manager_->ngd_;
  auto& ed = expl_manager_->ed_;

  ngd->pos_ = ed->point_now;
  ngd->yaw_vec_ = ed->yaw_vector;
  ngd->frontier_cell_ = ed->frontier_now;
}

void PAExplorationFSM::transformViewpointFormat(
    const vector<Vector3d>& temp_path, const vector<double>& temp_yaw, Vector3d& point, vector<double>& yaw_vec_) {

  // Most of the time, there are only two points on the path, and when there are more than two points, the intermidate point is
  // most likely the viewpoints sampled by the feature

  if (temp_path.size() < 2) return;

  // Simply choose the second point on the path as the next viewpoint
  point = temp_path[1];
  yaw_vec_ = vector<double>{ temp_yaw[1] };

  // If the path has more than two points, the next local traj is likely to lack of
  // explorability, and we need to manually avoid local planning failures due to this reason
  en_explore_check_ = (temp_path.size() == 2);
}

void PAExplorationFSM::informReplan(const double& t_colli) {

  auto& local_data = planner_manager_->local_data_;

  // Avoid multiple calling traj_server
  if (local_data.replan_begin_time_ > 0)
    return;
  else
    local_data.replan_begin_time_ = ros::Time::now().toSec();

  if (t_colli < 0.0)
    local_data.replan_stop_time_ = setStopTimeReplan();
  else
    local_data.replan_stop_time_ = setStopTimeCollision(t_colli);
  replan_msg_.data = local_data.replan_stop_time_;

  replan_pub_.publish(replan_msg_);
}

}  // namespace perception_aware_planner
