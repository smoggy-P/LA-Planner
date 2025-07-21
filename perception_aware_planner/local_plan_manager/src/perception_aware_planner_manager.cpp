#include "local_plan_manager/perception_aware_planner_manager.h"

// #include "stepping_debug/stepping_debug.h"

#include "utils/utils.h"

#define ANSI_COLOR_YELLOW_BOLD "\033[1;33m"
#define ANSI_COLOR_GREEN_BOLD "\033[1;32m"
#define ANSI_COLOR_RED_BOLD "\033[1;31m"
#define NORMAL_FONT "\033[0m"

using namespace std;
using namespace Eigen;

namespace perception_aware_planner {

void LocalPlanner::initPlanModules(ros::NodeHandle& nh) {

  nh.param("manager/max_vel", pp_.max_vel_, -1.0);
  nh.param("manager/max_acc", pp_.max_acc_, -1.0);
  nh.param("manager/max_yawdot", pp_.max_yawdot_, -1.0);
  nh.param("manager/control_points_distance", pp_.ctrl_pt_dist, -1.0);
  nh.param("manager/bspline_degree", pp_.bspline_degree_, 3);
  nh.param("manager/min_observed_ratio", pp_.min_observed_ratio_, 0.6);
  nh.param("manager/adjust_end_state", pp_.adjust_end_state_, false);

  sdf_map_.reset(new SDFMap(false));
  sdf_map_->initMap(nh);
  edt_environment_.reset(new EDTEnvironment);
  edt_environment_->setMap(sdf_map_);

  kino_path_finder_.reset(new KinodynamicAstar);
  kino_path_finder_->setParam(nh);
  kino_path_finder_->setEnvironment(edt_environment_);
  kino_path_finder_->init();

  bspline_optimizers_.reset(new BsplineOptimizer);
  bspline_optimizers_->setParam(nh);
  bspline_optimizers_->setEnvironment(edt_environment_);

  yaw_initial_planner_.reset(new YawInitialPlanner(nh));

  yaw_traj_opt_.reset(new YawTrajOptimizer);
  yaw_traj_opt_->setParam(nh);
}

void LocalPlanner::updatePosTrajInfo() {
  local_data_.velocity_traj_ = local_data_.position_traj_.getDerivative();
  local_data_.acceleration_traj_ = local_data_.velocity_traj_.getDerivative();

  local_data_.start_pos_ = local_data_.position_traj_.evaluateDeBoorT(0.0);

  local_data_.traj_id_++;

  statistics_.dt_ = local_data_.position_traj_.getKnotSpan();
}

void LocalPlanner::updateYawTrajInfo() {
  // Due to the truncation mechanism in yaw_initial_planner, the duration needs to be updated
  statistics_.duration_ = local_data_.duration_ = local_data_.yaw_traj_.getTotalDuration();

  local_data_.end_pos_ = local_data_.position_traj_.evaluateDeBoorT(local_data_.duration_);
  local_data_.end_yaw_ = local_data_.yaw_traj_.getPos(local_data_.duration_)[0];

  local_data_.position_traj_.getMeanAndMaxVel(statistics_.mean_vel_, statistics_.max_vel_, local_data_.duration_);

  statistics_.max_yaw_rate_ = local_data_.yaw_traj_.getMaxVelRate();
  statistics_.dt_yaw_ = local_data_.yaw_traj_.getDurations()[0];
}

bool LocalPlanner::isCollision(const Vector3d& pos) {
  // return !(sdf_map_->getOccupancy(pos) == SDFMap::FREE);
  return (sdf_map_->getOccupancy(pos) == SDFMap::OCCUPIED);
}

bool LocalPlanner::checkTrajCollision(double& distance, double& t_colli) {
  double t_now = (ros::Time::now() - local_data_.start_time_).toSec();

  Eigen::Vector3d cur_pt = local_data_.position_traj_.evaluateDeBoorT(t_now);
  double radius = 0.0;
  Eigen::Vector3d fut_pt;
  double fut_t = 0.02;

  while (radius < 6.0 && t_now + fut_t < local_data_.duration_) {
    fut_pt = local_data_.position_traj_.evaluateDeBoorT(t_now + fut_t);
    if (sdf_map_->getInflateOccupancy(fut_pt) == 1) {
      distance = radius;
      t_colli = t_now + fut_t;
      ROS_WARN("Collision at pos:(%f, %f, %f),time: %f", fut_pt[0], fut_pt[1], fut_pt[2], t_colli);
      return false;
    }
    radius = (fut_pt - cur_pt).norm();
    fut_t += 0.02;
  }

  return true;
}

bool LocalPlanner::checkTrajExploration(const vector<Vector3d>& target_frontier) {

  vector<Vector3d> knots_pos;
  vector<Vector3d> knots_acc;
  vector<Vector3d> knots_yaw;
  vector<double> dummy;

  getYawTrajForVis(knots_pos, knots_acc, knots_yaw, dummy, local_data_.start_time_);

  set<int> observed_features;

  for (size_t i = 0; i < knots_pos.size(); i++) {

    Quaterniond ori = Utils::calcOrientation(knots_yaw[i](0), knots_acc[i]);

    set<int> observed_features_knots;
    sdf_map_->countVisibleCells(knots_pos[i], ori, target_frontier, observed_features_knots);

    observed_features.insert(observed_features_knots.begin(), observed_features_knots.end());
    // We hope to maintain good explorability at the end of the trajectory
    if (i == knots_pos.size() - 1) {
      double min_observed_ratio_end = 0.1 * pp_.min_observed_ratio_;
      if (observed_features_knots.size() < min_observed_ratio_end * target_frontier.size()) {
        ROS_WARN("[LocalPlanner::checkTrajExploration] Insufficient end observations: %ld (threshold: %f)",
            observed_features_knots.size(), min_observed_ratio_end * target_frontier.size());
        return false;
      }
    }
  }

  statistics_.observed_frontier_num_ = observed_features.size();

  double ratio = static_cast<double>(observed_features.size()) / target_frontier.size();

  if (ratio < pp_.min_observed_ratio_) {
    ROS_WARN("[LocalPlanner::checkTrajExploration] Poor exploration with total ratio: %.4f", ratio);
    return false;
  }

  return true;
}

void LocalPlanner::printStatistics(const vector<Vector3d>& target_frontier) {
  cout << ANSI_COLOR_GREEN_BOLD;
  cout << "====================Local Planner Statistics====================" << endl;
  if (statistics_.kinodynamic_astar_status_ == KinodynamicAstar::REACH_HORIZON) {
    cout << "Kinodynamic A* Status:       REACH_HORIZON" << endl;
  } else if (statistics_.kinodynamic_astar_status_ == KinodynamicAstar::REACH_END) {
    cout << "Kinodynamic A* Status:       REACH_END" << endl;
  } else if (statistics_.kinodynamic_astar_status_ == KinodynamicAstar::NEAR_END) {
    cout << "Kinodynamic A* Status:       NEAR_END" << endl;
  } else if (statistics_.kinodynamic_astar_status_ == KinodynamicAstar::NO_PATH) {
    cout << "Kinodynamic A* Status:       NO_PATH" << endl;
  }
  cout << fixed << setprecision(3);
  cout << "Time of Kinodynamic A*:      " << statistics_.time_kinodynamic_astar_ << " (sec)" << endl;
  cout << "Time of Pos Traj Optimize:   " << statistics_.time_pos_traj_opt_ << " (sec)" << endl;
  cout << "Time of Yaw Initial Planner: " << statistics_.time_yaw_initial_planner_ << " (sec)" << endl;
  cout << "Time of Yaw Traj Optimize:   " << statistics_.time_yaw_traj_opt_ << " (sec)" << endl;
  statistics_.time_total_ = statistics_.time_kinodynamic_astar_ + statistics_.time_pos_traj_opt_ +
                            statistics_.time_yaw_initial_planner_ + statistics_.time_yaw_traj_opt_;
  cout << "Time of Total Planning:      " << statistics_.time_total_ << " (sec)" << endl;

  cout << fixed << setprecision(3);
  double max_vel = Utils::getGlobalParam().max_vel_;

  if (statistics_.max_vel_ < max_vel * 1.5)
    cout << ANSI_COLOR_GREEN_BOLD;
  else
    cout << ANSI_COLOR_RED_BOLD;
  cout << "Max Vel on Pos Traj:         " << statistics_.max_vel_ << " (m/s)" << endl << ANSI_COLOR_GREEN_BOLD;

  double max_yaw_rate = Utils::getGlobalParam().max_yaw_rate_;
  if (statistics_.max_yaw_rate_ < max_yaw_rate)
    cout << ANSI_COLOR_GREEN_BOLD;
  else
    cout << ANSI_COLOR_RED_BOLD;
  cout << "Max Yaw Rate on Yaw Traj:    " << statistics_.max_yaw_rate_ << " (rad/s)" << endl << ANSI_COLOR_GREEN_BOLD;

  cout << "Knot Span:                   " << statistics_.dt_ << " (sec)" << endl;
  cout << "Knot Span(Yaw):              " << statistics_.dt_yaw_ << " (sec)" << endl;
  cout << "Traj Duration:               " << statistics_.duration_ << " (sec)" << endl;

  if (!target_frontier.empty()) {
    cout << "Observed Frontiers Num:      " << statistics_.observed_frontier_num_ << "/" << target_frontier.size() << endl;
  }
  cout.unsetf(ios::fixed);
  cout << "===============================================================" << endl;
  cout << NORMAL_FONT;
}

int LocalPlanner::planPosTraj(const Vector3d& start_pt, const Vector3d& start_vel, const Vector3d& start_acc,
    const Vector3d& end_pt, const Vector3d& end_vel, const double& time_lb) {

  if ((start_pt - end_pt).norm() < 1e-2) {
    cout << "Close goal" << endl;
    return PATH_SEARCH_ERROR;
  }

  // Step1: Call Hybrid A* to get initial waypoints
  auto time_start = ros::Time::now();

  kino_path_finder_->reset();

  kino_astar_status_ = kino_path_finder_->search(start_pt, start_vel, start_acc, end_pt, end_vel, true);
  if (kino_astar_status_ == KinodynamicAstar::NO_PATH) {
    cout << "[Kino replan]: search 1 fail." << endl;
    // Retry
    kino_path_finder_->reset();
    kino_astar_status_ = kino_path_finder_->search(start_pt, start_vel, start_acc, end_pt, end_vel, false);
    if (kino_astar_status_ == KinodynamicAstar::NO_PATH) {
      ROS_ERROR("[Kino replan]: Can't find path.");
      return PATH_SEARCH_ERROR;
    }
  }

  kino_path_ = kino_path_finder_->getKinoTraj(0.01);

  statistics_.kinodynamic_astar_status_ = kino_astar_status_;
  statistics_.time_kinodynamic_astar_ = (ros::Time::now() - time_start).toSec();

  // Step2: Optimize the trajectory using bspline optimizer
  auto time_start_2 = ros::Time::now();

  double dt = pp_.ctrl_pt_dist / pp_.max_vel_;
  vector<Vector3d> point_set, start_end_derivatives;
  kino_path_finder_->getSamples(dt, point_set, start_end_derivatives);

  Eigen::MatrixXd ctrl_pts;
  NonUniformBspline::parameterizeToBspline(dt, point_set, start_end_derivatives, pp_.bspline_degree_, ctrl_pts);
  NonUniformBspline init_traj(ctrl_pts, pp_.bspline_degree_, dt);

  vector<Vector3d> start, end;
  vector<bool> start_idx, end_idx;

  // if (kino_astar_status_ == KinodynamicAstar::REACH_END) {
  //   init_traj.getBoundaryStates(2, 2, start, end);
  //   start_idx = { true, true, true };
  //   end_idx = { true, true, true };
  // }

  // else {
  //   init_traj.getBoundaryStates(2, 0, start, end);
  //   start_idx = { true, true, true };
  //   end_idx = { true, false, false };
  // }

  init_traj.getBoundaryStates(2, 2, start, end);
  if (pp_.adjust_end_state_) {
    end[1] = end_vel;
    end[2] = Vector3d::Zero();
  }
  start_idx = { true, true, true };
  end_idx = { true, true, true };

  bspline_optimizers_->setBoundaryStates(start, end, start_idx, end_idx);
  if (time_lb > 0) bspline_optimizers_->setTimeLowerBound(time_lb);

  int cost_func = 0;
  cost_func |= BsplineOptimizer::SMOOTHNESS;
  cost_func |= BsplineOptimizer::FEASIBILITY;
  cost_func |= BsplineOptimizer::START;
  cost_func |= BsplineOptimizer::END;
  cost_func |= BsplineOptimizer::MINTIME;
  cost_func |= BsplineOptimizer::DISTANCE;

  bspline_optimizers_->optimize(ctrl_pts, dt, cost_func, 1, 1);
  if (!bspline_optimizers_->issuccess) return POSISION_OPT_ERROR;
  local_data_.position_traj_.setUniformBspline(ctrl_pts, pp_.bspline_degree_, dt);

  statistics_.time_pos_traj_opt_ = (ros::Time::now() - time_start_2).toSec();
  updatePosTrajInfo();

  return SUCCESS_FIND_POSISION_TRAJ;
}

int LocalPlanner::planYawTraj(const Vector3d& start_yaw, const vector<double>& end_yaw_vec,
    const vector<Vector3d>& frontier_cells, const Vector3d& final_goal, const bool go2final) {

  auto time_start = ros::Time::now();

  // Step1: Determine the time interval for the yaw trajectory
  double total_time = local_data_.position_traj_.getTimeSum();
  double max_tol_time = Utils::getGlobalParam().max_tol_time_;
  int piece_num = static_cast<int>(total_time / max_tol_time) + 1;
  double dt_yaw = total_time / piece_num;

  // Step2.1: Using graph search algorithm to obtain initial intermediate waypoints
  vector<Vector3d> pos_vec, acc_vec;
  local_data_.position_traj_.evaluateDeBoorTVec(dt_yaw, pos_vec);
  local_data_.acceleration_traj_.evaluateDeBoorTVec(dt_yaw, acc_vec);

  vector<double> yaw_waypoints;
  yaw_initial_planner_->setFeatureMap(feature_map_);
  yaw_initial_planner_->setSDFmap(sdf_map_);
  yaw_initial_planner_->setFinalGoal(final_goal);
  yaw_initial_planner_->setIfPlan2FinalGoal(go2final);
  yaw_initial_planner_->setPos(pos_vec);
  yaw_initial_planner_->setAcc(acc_vec);
  yaw_initial_planner_->setTargetFrontier(frontier_cells);

  if (!yaw_initial_planner_->search(start_yaw[0], end_yaw_vec, dt_yaw, yaw_waypoints)) {
    ROS_ERROR("Yaw Trajectory Planning Failed in Graph Search!!!");
    return YAW_INIT_ERROR;
  }

  // Step2.2: Set initial constraints
  Vector3d start_yaw3d = start_yaw;
  Utils::roundPi(start_yaw3d[0]);

  Matrix3d headState = Matrix3d::Zero();
  headState(0, 0) = start_yaw3d(0);
  headState(0, 1) = start_yaw3d(1);
  headState(0, 2) = start_yaw3d(2);

  // Step2.3: Set intermediate constraints
  double last_yaw = yaw_waypoints[0];
  MatrixXd initInnerPts(3, yaw_waypoints.size() - 2);
  for (size_t i = 1; i < yaw_waypoints.size() - 1; i++) {
    Vector3d waypt = Vector3d::Zero();
    waypt(0) = yaw_waypoints[i];
    Utils::calcNextYaw(last_yaw, waypt(0));
    last_yaw = waypt(0);
    initInnerPts.col(i - 1) = waypt;
  }

  // Step2.4: Set end constraints
  Eigen::Vector3d end_yaw3d(yaw_waypoints.back(), 0, 0);
  Utils::calcNextYaw(last_yaw, end_yaw3d(0));

  Matrix3d tailState = Matrix3d::Zero();
  tailState(0, 0) = end_yaw3d(0);

  VectorXd initT(yaw_waypoints.size() - 1);
  for (size_t i = 0; i < yaw_waypoints.size() - 1; i++) initT(i) = dt_yaw;

  statistics_.time_yaw_initial_planner_ = (ros::Time::now() - time_start).toSec();

  // Step3: Call the minco trajectory optimizer to optimize the yaw trajectory
  auto time_start_2 = ros::Time::now();

  YawOptData::Ptr opt_data = make_shared<YawOptData>();
  yaw_initial_planner_->prepareOptData(opt_data);
  yaw_traj_opt_->setOptData(opt_data);
  yaw_traj_opt_->setFeatureMap(feature_map_);

  if (!yaw_traj_opt_->optimizeTrajectory(headState, tailState, initInnerPts, initT)) return YAW_OPT_ERROR;
  yaw_traj_opt_->getTrajectory(local_data_.yaw_traj_);
  updateYawTrajInfo();

  statistics_.time_yaw_traj_opt_ = (ros::Time::now() - time_start_2).toSec();

  return SUCCESS_FIND_YAW_TRAJ;
}

traj_utils::MixTraj LocalPlanner::generateROSMsg() {

  traj_utils::MixTraj traj_msg;

  traj_msg.bspline_degree = pp_.bspline_degree_;
  traj_msg.traj_id = local_data_.traj_id_;
  traj_msg.start_time = local_data_.start_time_;
  traj_msg.real_traj_duration = local_data_.duration_;

  // Pos traj
  Eigen::MatrixXd pos_pts = local_data_.position_traj_.getControlPoint();
  for (int i = 0; i < pos_pts.rows(); ++i) {
    geometry_msgs::Point pt;
    pt.x = pos_pts(i, 0);
    pt.y = pos_pts(i, 1);
    pt.z = pos_pts(i, 2);
    traj_msg.pos_pts.push_back(pt);
  }

  Eigen::VectorXd knots = local_data_.position_traj_.getKnot();
  for (int i = 0; i < knots.rows(); ++i) {
    traj_msg.knots.push_back(knots(i));
  }

  // Yaw traj
  traj_msg.minco_order = 5;

  auto durs = local_data_.yaw_traj_.getDurations();
  int piece_num = local_data_.yaw_traj_.getPieceNum();
  traj_msg.duration_yaw.resize(piece_num);
  traj_msg.coef_yaw.resize(6 * piece_num);
  for (int i = 0; i < piece_num; ++i) {
    traj_msg.duration_yaw[i] = durs(i);

    auto cMat = local_data_.yaw_traj_[i].getCoeffMat();
    for (int j = 0; j < 6; j++) {
      traj_msg.coef_yaw[6 * i + j] = cMat(0, j);
    }
  }

  return traj_msg;
}

void LocalPlanner::getYawTrajForVis(vector<Vector3d>& pos_vec, vector<Vector3d>& acc_vec, vector<Vector3d>& yaw_vec,
    std::vector<double>& bound_vec, const ros::Time& time_now) {
  pos_vec.clear();
  acc_vec.clear();
  yaw_vec.clear();

  auto times = yaw_traj_opt_->getTimes();
  Eigen::VectorXd lb, ub;
  yaw_traj_opt_->getLocalizableCorridor(lb, ub);

  double t = 0;
  double t_start = (time_now - local_data_.start_time_).toSec();

  int i = 0;
  do {
    if (t >= t_start) {
      pos_vec.push_back(local_data_.position_traj_.evaluateDeBoorT(t));
      acc_vec.push_back(local_data_.acceleration_traj_.evaluateDeBoorT(t));
      yaw_vec.push_back(local_data_.yaw_traj_.getPos(t));
      if (i >= 1) {
        bound_vec.push_back(lb(i - 1));
        bound_vec.push_back(ub(i - 1));
      }
    }

    t += times(i);
    i++;

  } while (i < times.size());

  // for (int i = 0; i < lb.rows(); i++) {
  //   bound_vec.push_back(lb(i));
  //   bound_vec.push_back(ub(i));
  // }
}

}  // namespace perception_aware_planner
