#ifndef _PERCEPTION_AWARE_EXPLORATION_MANAGER_H_
#define _PERCEPTION_AWARE_EXPLORATION_MANAGER_H_

#include "exploration_manager/perception_aware_exploration_fsm.h"
#include "exploration_manager/failure_detector.h"

#include "traj_utils/planning_visualization.h"

#include <Eigen/Eigen>

#include <ros/ros.h>

using std::shared_ptr;
using std::unique_ptr;
using std::vector;
using std::weak_ptr;

namespace perception_aware_planner {

class EDTEnvironment;
class SDFMap;
class FeatureMap;
class LocalPlanner;
class PAExplorationFSM;
class FailureDetector;
class FrontierFinder;
struct ExplorationData;
struct NextGoalData;

class PAExplorationManager {
public:
  using Ptr = shared_ptr<PAExplorationManager>;

  PAExplorationManager(shared_ptr<PAExplorationFSM> expl_fsm);

  void initialize(ros::NodeHandle& nh);

  void setEnableExploreCheck(const bool enable) {
    en_explorability_check_ = enable;
  }

  void selectNextGoal();
  VIEWPOINT_CHANGE_REASON planToNextGoal(
      const Vector3d& next_pos, const vector<double>& next_yaw_vec, const vector<Vector3d>& frontier_cells, const bool& go2final);

  shared_ptr<NextGoalData> ngd_ = nullptr;
  shared_ptr<ExplorationData> ed_ = nullptr;
  shared_ptr<FailureDetector> failure_detector_ = nullptr;
  shared_ptr<LocalPlanner> planner_manager_ = nullptr;
  shared_ptr<FrontierFinder> frontier_finder_ = nullptr;
  shared_ptr<FeatureMap> feature_map_ = nullptr;
  shared_ptr<SDFMap> global_sdf_map_ = nullptr;

private:
  weak_ptr<PAExplorationFSM> expl_fsm_;  // Convenient for sharing some data with FSM

  shared_ptr<EDTEnvironment> edt_environment_ = nullptr;
  shared_ptr<SDFMap> sdf_map_ = nullptr;

  bool en_explorability_check_ = true;

  bool isNextFinal(const Vector3d& next_pos);
};

}  // namespace perception_aware_planner

#endif