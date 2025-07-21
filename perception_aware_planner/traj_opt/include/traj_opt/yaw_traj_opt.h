#ifndef TRAJ_OPT_YAW_TRAJ_OPT_H_
#define TRAJ_OPT_YAW_TRAJ_OPT_H_

#include "plan_env/edt_environment.h"
#include "plan_env/feature_map.h"

#include "utils/utils.h"

#include "gcopter/minco.hpp"

#include <ros/ros.h>

#include <Eigen/Eigen>

using std::vector;

namespace perception_aware_planner {

class YawTrajOptimizer {

public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  using Ptr = std::shared_ptr<YawTrajOptimizer>;

  struct Param {
    double max_yaw_dot_;
    double max_yaw_dotdot_;

    double ld_feasi_;
    double ld_expl_;
    double ld_weight1_, ld_weight2_;

    int cps_num_perPiece_;

    double k1_;
    double k2_;
    double k3_;
  };

  Param param;

  void setParam(ros::NodeHandle& nh);

  void genLocalizableCorridor(const MatrixXd& initInnerPts);
  void getLocalizableCorridor(Eigen::VectorXd& lb_in, Eigen::VectorXd& ub_in) {
    lb_in = lb;
    ub_in = ub;
  }

  void setFeatureMap(const FeatureMap::Ptr& map) {
    feature_map_ = map;
  }

  void setOptData(const YawOptData::Ptr& data) {
    opt_data_ = data;
  }

  bool optimizeTrajectory(
      const Matrix3d& headState, const Matrix3d& tailState, const Eigen::MatrixXd& initInnerPts, const Eigen::VectorXd& initT);

  void getTrajectory(Trajectory<5>& traj) {
    minco.getTrajectory(traj);
  }

  Eigen::VectorXd getTimes() {
    return times;
  }

private:
  minco::MINCO_S3NU minco;

  int piece_num;

  Eigen::Matrix3Xd yaws;
  Eigen::VectorXd times;
  Eigen::Matrix3Xd gradYaws;
  Eigen::MatrixX3d gradCoeffs;

  Eigen::VectorXd lb;
  Eigen::VectorXd ub;

  YawOptData::Ptr opt_data_ = nullptr;

  FeatureMap::Ptr feature_map_ = nullptr;
  CameraParam::Ptr frontier_cam_ = nullptr;

  // xi([-inf,inf]) -> yaw([lb,ub])
  void forwardYaw(const Eigen::VectorXd& xi, Eigen::Matrix3Xd& yaw) {
    yaw.resize(3, piece_num - 1);
    yaw.setZero();

    for (int i = 0; i < piece_num - 1; ++i) {
      double ratio = 1 / (1 + exp(-xi(i)));
      yaw(0, i) = lb(i) + (ub(i) - lb(i)) * ratio;
    }
  }

  // yaw([lb,ub]) -> xi([-inf,inf])
  template <typename EIGENVEC>
  void backwardYaw(const Eigen::Matrix3Xd& yaw_mat, EIGENVEC& xi) {
    xi.resize(piece_num - 1);

    for (int i = 0; i < piece_num - 1; ++i) {
      double yaw = yaw_mat(0, i);
      double ratio = (yaw - lb(i)) / (ub(i) - lb(i));
      xi(i) = -log((1 / ratio) - 1);
    }
  }

  template <typename EIGENVECGD>
  void backwardGradYaw(const Eigen::VectorXd& xi, const Eigen::Matrix3Xd& gradYaws, EIGENVECGD& gradXi) {
    gradXi.resize(piece_num - 1);
    gradXi.setZero();

    for (int i = 0; i < piece_num - 1; ++i) {
      double dyaw_dxi = (ub(i) - lb(i)) * exp(-xi(i)) / (1 + exp(-xi(i))) / (1 + exp(-xi(i)));
      gradXi(i) = gradYaws(0, i) * dyaw_dxi;
    }
  }

  void ExplorationCostKnot(
      const double& yaw, const Eigen::Vector3d& pos, const Eigen::Vector3d& acc, const int layer, double& cost, double& grad);
  void calcExplorationCost(double& cost, Eigen::Matrix3Xd& gradYaws);

  void calcFeasibilityCost(const Eigen::VectorXd& T, double& cost_fea, const int& K);
  bool feasibilityGradCostV(const Eigen::Vector3d& v, Eigen::Vector3d& gradv, double& costv);
  bool feasibilityGradCostA(const Eigen::Vector3d& a, Eigen::Vector3d& grada, double& costa);

  static double costFunctionCallback(void* ptr, const double* x, double* g, const int n);
};

}  // namespace perception_aware_planner
#endif