#include "exploration_manager/perception_aware_exploration_manager.h"
#include "exploration_manager/failure_detector.h"

#include "local_plan_manager/perception_aware_planner_manager.h"

#include "utils/utils.h"

using namespace Eigen;

namespace perception_aware_planner {

PAExplorationManager::PAExplorationManager(shared_ptr<PAExplorationFSM> expl_fsm) : expl_fsm_(expl_fsm) {
}

void PAExplorationManager::initialize(ros::NodeHandle& nh) {
  planner_manager_.reset(new LocalPlanner);
  planner_manager_->initPlanModules(nh);

  edt_environment_ = planner_manager_->edt_environment_;
  sdf_map_ = edt_environment_->sdf_map_;

  ngd_.reset(new NextGoalData);
  ed_.reset(new ExplorationData);

  // Note: We need a globally known obstacle map to better simulate the feature map
  // This map is not available to the algorithm
  global_sdf_map_.reset(new SDFMap(true));
  global_sdf_map_->initMap(nh);

  feature_map_.reset(new FeatureMap);
  feature_map_->setMap(global_sdf_map_, sdf_map_);
  feature_map_->initMap(nh);

  planner_manager_->setFeatureMap(feature_map_);

  failure_detector_.reset(new FailureDetector(nh));
  failure_detector_->setFeatureMap(feature_map_);

  frontier_finder_.reset(new FrontierFinder(edt_environment_, feature_map_, nh));
}

bool PAExplorationManager::isNextFinal(const Vector3d& next_pos) {
  const auto& final_goal = expl_fsm_.lock()->final_goal_;
  double dis = (final_goal - next_pos).norm();
  if (sdf_map_->getOccupancy(final_goal) == SDFMap::FREE) {
    ROS_WARN("Final_goal is in free area!!!");
    cout << "--final_goal: " << final_goal.transpose() << endl;
    cout << "--next_pos: " << next_pos.transpose() << endl;

    if (dis < 0.1) {
      ROS_WARN("Go to Final Goal!!!!");
      return true;
    }

    else
      ROS_WARN("Go to temp viewpoint. Maybe can't find locatable graph to final goal!!!");
  }

  return false;
}

void PAExplorationManager::selectNextGoal() {
  auto& last_fail_reason = expl_fsm_.lock()->last_fail_reason;

  // Determine is it possible to go to fianl goal
  if (isNextFinal(ngd_->pos_)) {
    last_fail_reason = planToNextGoal(ngd_->pos_, ngd_->yaw_vec_, ngd_->frontier_cell_, true);
    if (last_fail_reason != NONE)
      ROS_WARN("Find final goal, but fail to Generate Path!!!");

    else {
      ROS_WARN("Successfully plan to final goal");
      ngd_->pos_ = planner_manager_->local_data_.end_pos_;
      ngd_->yaw_ = planner_manager_->local_data_.end_yaw_;
      ngd_->frontier_cell_.clear();
      if (planner_manager_->kino_astar_status_ == KinodynamicAstar::REACH_HORIZON)
        ngd_->type_ = GOTO_FINAL_GOAL;
      else
        ngd_->type_ = REACH_FINAL_GOAL;

      return;
    }
  }

  // Call local planner to move to next goal
  last_fail_reason = planToNextGoal(ngd_->pos_, ngd_->yaw_vec_, ngd_->frontier_cell_, false);
  ngd_->type_ = (last_fail_reason == NONE) ? TMP_VIEWPOINT : LOCAL_PLAN_FAIL;
  ngd_->pos_ = planner_manager_->local_data_.end_pos_;
  ngd_->yaw_ = planner_manager_->local_data_.end_yaw_;
}

VIEWPOINT_CHANGE_REASON PAExplorationManager::planToNextGoal(
    const Vector3d& next_pos, const vector<double>& next_yaw_vec, const vector<Vector3d>& frontier_cells, const bool& go2final) {

  const auto& start_pos = expl_fsm_.lock()->start_pos_;
  const auto& start_vel = expl_fsm_.lock()->start_vel_;
  const auto& start_acc = expl_fsm_.lock()->start_acc_;
  const auto& start_yaw = expl_fsm_.lock()->start_yaw_;
  const auto& final_goal = expl_fsm_.lock()->final_goal_;

  double diff = fabs(next_yaw_vec.front() - start_yaw[0]);
  double max_yaw_rate = Utils::getGlobalParam().max_yaw_rate_;
  double time_lb = min(diff, 2 * M_PI - diff) / max_yaw_rate;

  // Step1: Plan position trajectory
  int pos_traj_statu = planner_manager_->planPosTraj(start_pos, start_vel, start_acc, next_pos, Vector3d::Zero(), time_lb);

  if (pos_traj_statu == PATH_SEARCH_ERROR)
    return PATH_SEARCH_FAIL;
  else if (pos_traj_statu == POSISION_OPT_ERROR)
    return POSITION_OPT_FAIL;

  if (planner_manager_->local_data_.position_traj_.getTimeSum() < time_lb - 0.1) {
    ROS_ERROR("Time lower bound not satified!");
  }

  // Step2: Plan yaw trajectory
  int yaw_traj_statu = planner_manager_->planYawTraj(start_yaw, next_yaw_vec, frontier_cells, final_goal, go2final);
  if (yaw_traj_statu == YAW_INIT_ERROR)
    return YAW_INIT_FAIL;
  else if (yaw_traj_statu == YAW_OPT_ERROR)
    return YAW_OPT_FAIL;

  // Step3: Check the locability of the trajectory
  if (!failure_detector_->checkTraj(planner_manager_->local_data_)) return LOCABILITY_CHECK_FAIL;

  // Step4: Check the explorability of the trajectory
  if (en_explorability_check_ && !go2final && planner_manager_->kino_astar_status_ != KinodynamicAstar::REACH_HORIZON &&
      !planner_manager_->checkTrajExploration(frontier_cells))
    return EXPLORABILITY_CHECK_FAIL;

  // Step5: If pass all the above steps, that we get a feasible local trajectory
  planner_manager_->printStatistics(frontier_cells);

  return NONE;
}

}  // namespace perception_aware_planner
