#ifndef PERCEPTION_AWARE_PLANNER_MANAGER_H_
#define PERCEPTION_AWARE_PLANNER_MANAGER_H_

#include "active_perception/frontier_finder.h"

#include "path_searching/kinodynamic_astar.h"

#include "local_plan_manager/plan_container.hpp"
#include "local_plan_manager/yaw_initial_planner.h"

#include "traj_opt/bspline_optimizer.h"
#include "traj_opt/yaw_traj_opt.h"

#include "traj_utils/Bspline.h"
#include "traj_utils/MixTraj.h"

namespace perception_aware_planner {

enum LOCAL_PLAN_RESULT {
  SUCCESS_FIND_POSISION_TRAJ = 0,
  PATH_SEARCH_ERROR = 1,
  POSISION_OPT_ERROR = 2,
  SUCCESS_FIND_YAW_TRAJ = 3,
  YAW_INIT_ERROR = 4,
  YAW_OPT_ERROR = 5
};

struct Statistics {
  int kinodynamic_astar_status_;     // Status of kinodynamic A*
  double time_kinodynamic_astar_;    // Time cost for kinodynamic A*(s)
  double time_pos_traj_opt_;         // Time cost for pos traj optimization(s)
  double time_yaw_initial_planner_;  // Time cost for searching yaw waypoints(s)
  double time_yaw_traj_opt_;         // Time cost for yaw traj optimization(s)
  double time_total_;                // Time cost for total local planning(s)
  double mean_vel_;                  // Mean vel on local traj(m/s)
  double max_vel_;                   // Max vel on local traj(m/s)
  double mean_yaw_rate_;             // Mean yaw rate on local traj(rad/s)
  double max_yaw_rate_;              // Max yaw rate on local traj(rad/s)
  int observed_frontier_num_;        // Num of the observed frontier cells on local traj
  double dt_;                        // Knot span of pos traj(s)
  double dt_yaw_;                    // Knot span of yaw traj(s)
  double duration_;                  // Running time of local traj(s)
};

class LocalPlanner {

public:
  using Ptr = shared_ptr<LocalPlanner>;

  /* main planning interface */
  int planPosTraj(const Vector3d& start_pt, const Vector3d& start_vel, const Vector3d& start_acc, const Vector3d& end_pt,
      const Vector3d& end_vel, const double& time_lb);

  int planYawTraj(const Vector3d& start_yaw, const vector<double>& end_yaw_vec, const vector<Vector3d>& frontier_cells,
      const Vector3d& final_goal, const bool go2final);

  void getYawTrajForVis(std::vector<Eigen::Vector3d>& pos_vec, std::vector<Eigen::Vector3d>& acc_vec,
      std::vector<Vector3d>& yaw_vec, std::vector<double>& bound_vec, const ros::Time& time_now = ros::Time::now());

  void initPlanModules(ros::NodeHandle& nh);

  bool isCollision(const Vector3d& pos);
  bool checkTrajCollision(double& distance, double& t_colli);
  bool checkTrajExploration(const vector<Vector3d>& target_frontier);

  void setFeatureMap(const shared_ptr<FeatureMap>& feature_map) {
    feature_map_ = feature_map;
  }

  void printStatistics(const vector<Vector3d>& target_frontier);

  traj_utils::MixTraj generateROSMsg();

  PlanParameters pp_;
  Statistics statistics_;
  LocalTrajData local_data_;

  EDTEnvironment::Ptr edt_environment_ = nullptr;
  shared_ptr<SDFMap> sdf_map_ = nullptr;

  // kinodynamic path
  vector<Eigen::Vector3d> kino_path_;
  int kino_astar_status_;

private:
  /* main planning algorithms & modules */
  shared_ptr<FeatureMap> feature_map_ = nullptr;

  unique_ptr<KinodynamicAstar> kino_path_finder_ = nullptr;
  BsplineOptimizer::Ptr bspline_optimizers_ = nullptr;
  unique_ptr<YawInitialPlanner> yaw_initial_planner_ = nullptr;
  YawTrajOptimizer::Ptr yaw_traj_opt_ = nullptr;

  void updatePosTrajInfo();
  void updateYawTrajInfo();
};
}  // namespace perception_aware_planner

#endif