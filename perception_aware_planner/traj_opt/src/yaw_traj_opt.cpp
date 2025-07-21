#include "traj_opt/yaw_traj_opt.h"

#include "traj_opt/lbfgs.hpp"

using namespace Eigen;

namespace perception_aware_planner {

void YawTrajOptimizer::setParam(ros::NodeHandle& nh) {

  nh.param("yaw_traj_opt/max_yaw_dot", param.max_yaw_dot_, -1.0);
  nh.param("yaw_traj_opt/max_yaw_dotdot", param.max_yaw_dotdot_, -1.0);

  nh.param("yaw_traj_opt/ld_feasi", param.ld_feasi_, -1.0);
  nh.param("yaw_traj_opt/ld_expl", param.ld_expl_, -1.0);

  nh.param("yaw_traj_opt/constraint_points_perPiece", param.cps_num_perPiece_, -1);

  // Refer to APACE
  nh.param("yaw_traj_opt/k1", param.k1_, -1.0);
  nh.param("yaw_traj_opt/k2", param.k2_, -1.0);
  nh.param("yaw_traj_opt/k3", param.k3_, -1.0);

  nh.param("yaw_traj_opt/ld_weight1", param.ld_weight1_, -1.0);
  nh.param("yaw_traj_opt/ld_weight2", param.ld_weight2_, -1.0);

  frontier_cam_ = Utils::getGlobalParam().frontier_cam_;
}

void YawTrajOptimizer::genLocalizableCorridor(const MatrixXd& initInnerPts) {
  lb.resize(initInnerPts.cols());
  ub.resize(initInnerPts.cols());

  feature_map_->genLocalizableCorridor(opt_data_, initInnerPts, lb, ub);

  // cout << "lb: " << lb.transpose() << endl;
  // cout << "ub: " << ub.transpose() << endl;
}

bool YawTrajOptimizer::optimizeTrajectory(
    const Matrix3d& headState, const Matrix3d& tailState, const MatrixXd& initInnerPts, const VectorXd& initT) {

  // Step1: Set the parameters of the lbfgs optimizer
  lbfgs::lbfgs_parameter_t lbfgs_params;
  lbfgs::lbfgs_load_default_parameters(&lbfgs_params);

  lbfgs_params.mem_size = 15;
  lbfgs_params.max_iterations = 300;
  lbfgs_params.g_epsilon = 10;
  lbfgs_params.max_step = 1e+20;
  lbfgs_params.min_step = 1e-20;

  // Step2: Prepare the data for optimization
  piece_num = initT.size();
  minco.setConditions(headState, tailState, piece_num);

  genLocalizableCorridor(initInnerPts);

  double x_init[piece_num - 1];
  Eigen::Map<Eigen::VectorXd> xi(x_init, piece_num - 1);
  backwardYaw(initInnerPts, xi);

  times = initT;

  // Step3: Optimize the trajectory
  double final_cost;
  int result = lbfgs::lbfgs_optimize(
      piece_num, x_init, &final_cost, YawTrajOptimizer::costFunctionCallback, nullptr, nullptr, this, &lbfgs_params);

  switch (result) {
    case lbfgs::LBFGS_CONVERGENCE:
    case lbfgs::LBFGSERR_MAXIMUMITERATION:
    case lbfgs::LBFGS_ALREADY_MINIMIZED:
    case lbfgs::LBFGS_STOP:
    // Note: These error codes are considered to be acceptable
    case lbfgs::LBFGSERR_ROUNDING_ERROR:
    case lbfgs::LBFGSERR_MINIMUMSTEP:
    case lbfgs::LBFGSERR_WIDTHTOOSMALL: {
      // ROS_INFO("Successfully optimize using lbfgs");
      forwardYaw(xi, yaws);

      vector<double> yaw_full;
      yaw_full.emplace_back(headState(0, 0));
      for (int i = 0; i < piece_num - 1; i++) {
        yaw_full.emplace_back(yaws(0, i));
      }
      yaw_full.emplace_back(tailState(0, 0));

      for (size_t i = 0; i < yaw_full.size(); ++i) {
        if (i != 0) {
          double y1 = yaw_full[i - 1];
          double y2 = yaw_full[i];
          double diff = fabs(y1 - y2);
          if (diff > M_PI) {
            ROS_ERROR("YawTrajOptimizer::optimizeTrajectory: yaw diff is too large, y1: %f, y2: %f", y1, y2);
            return false;
          }
        }
      }

      minco.setParameters(yaws, times);
      return true;
    }
    default: {
      ROS_ERROR("[yaw_opt]: Solver error in optimize_lbfgs!, return = %d, %s", result, lbfgs::lbfgs_strerror(result));
      return false;
    }
  }
}

bool YawTrajOptimizer::feasibilityGradCostV(const Eigen::Vector3d& v, Eigen::Vector3d& gradv, double& costv) {
  double v_penal = v.squaredNorm() - param.max_yaw_dot_ * param.max_yaw_dot_;
  if (v_penal > 0) {
    gradv = param.ld_feasi_ * 6 * v_penal * v_penal * v;
    costv = param.ld_feasi_ * v_penal * v_penal * v_penal;
    return true;
  }

  gradv.setZero();
  costv = 0.0;
  return false;
}

bool YawTrajOptimizer::feasibilityGradCostA(const Eigen::Vector3d& a, Eigen::Vector3d& grada, double& costa) {
  double a_penal = a.squaredNorm() - param.max_yaw_dotdot_ * param.max_yaw_dotdot_;
  if (a_penal > 0) {
    grada = param.ld_feasi_ * 6 * a_penal * a_penal * a;
    costa = param.ld_feasi_ * a_penal * a_penal * a_penal;
    return true;
  }

  grada.setZero();
  costa = 0.0;
  return false;
}

void YawTrajOptimizer::calcFeasibilityCost(const Eigen::VectorXd& T, double& cost_fea, const int& K) {

  cost_fea = 0.0;

  double s1, s2, s3, s4, s5;
  Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;

  for (int i = 0; i < piece_num; i++) {
    const Eigen::Matrix<double, 6, 3>& c = minco.getCoeffs().block<6, 3>(i * 6, 0);
    double step = T(i) / K;
    s1 = 0.0;

    for (int j = 0; j <= K; j++) {
      s2 = s1 * s1;
      s3 = s2 * s1;
      s4 = s2 * s2;
      s5 = s4 * s1;
      beta0 << 1.0, s1, s2, s3, s4, s5;
      beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
      beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
      beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
      beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1;

      Eigen::Vector3d vel = c.transpose() * beta1;
      Eigen::Vector3d acc = c.transpose() * beta2;

      double omg = (j == 0 || j == K) ? 0.5 : 1.0;

      Eigen::Vector3d gradv;
      double costv;
      if (feasibilityGradCostV(vel, gradv, costv)) {
        Eigen::Matrix<double, 6, 3> gradViolaVc = beta1 * gradv.transpose();
        gradCoeffs.block<6, 3>(i * 6, 0) += omg * step * gradViolaVc;
        cost_fea += omg * step * costv;
      }

      Eigen::Vector3d grada;
      double costa;
      if (feasibilityGradCostA(acc, grada, costa)) {
        Eigen::Matrix<double, 6, 3> gradViolaAc = beta2 * grada.transpose();
        gradCoeffs.block<6, 3>(i * 6, 0) += omg * step * gradViolaAc;
        cost_fea += omg * step * costa;
      }

      s1 += step;
    }
  }
}

void YawTrajOptimizer::ExplorationCostKnot(
    const double& yaw, const Vector3d& pos, const Vector3d& acc, const int layer, double& cost, double& grad) {

  cost = grad = 0.0;

  auto& status_vec = opt_data_->frontier_status_[layer];
  ROS_ASSERT(status_vec.size() == opt_data_->frontier_cells_.size());

  Vector3d gravity(0, 0, -9.81);
  double total_weight = 0.0;

  for (size_t i = 0; i < opt_data_->frontier_cells_.size(); i++) {
    if (status_vec[i] == NOT_AVAILABLE || status_vec[i] == HAS_BEEN_OBSERVED) continue;

    double w = 0.0;
    if (status_vec[i] == VISIBLE) {
      w = param.ld_weight1_;
    } else if (status_vec[i] == AVAILABLE) {
      w = param.ld_weight2_;
    } else {
      ROS_ERROR("Error Type for status_vec[i]: %d", status_vec[i]);
      ROS_BREAK();
    }

    Vector3d cell = opt_data_->frontier_cells_[i];

    // Refer to APACE
    // Calculate vectors n1, ny, n3 ,b and their gradients
    Vector3d n1, ny, n3, n2, b;
    n1 = acc - gravity;  // thrust
    ny << cos(yaw), sin(yaw), 0;
    n3 = n1.cross(ny);
    n2 = n3.cross(n1);
    b = cell - pos;

    Vector3d dn3_dyaw;
    dn3_dyaw << -n1(2) * cos(yaw), -n1(2) * sin(yaw), n1(1) * sin(yaw) + n1(0) * cos(yaw);

    // v1
    double k1 = param.k1_;
    double fov_vertical = frontier_cam_->fov_vertical * M_PI / 180;
    double alpha1 = (M_PI - fov_vertical) / 2.0;
    double sin_theta1 = n1.cross(b).norm() / (n1.norm() * b.norm());
    double v1 = Utils::sigmoid(k1, (sin_theta1 - sin(alpha1)));

    // v2
    double k2 = param.k2_;
    double cos_theta2 = n2.dot(b) / (n2.norm() * b.norm());
    double v2 = Utils::sigmoid(k2, cos_theta2);

    double v1v2 = v1 * v2;

    // v3
    double k3 = param.k3_;
    double fov_horizontal = frontier_cam_->fov_horizontal * M_PI / 180;
    double alpha3 = (M_PI - fov_horizontal) / 2.0;
    double sin_theta3 = n3.cross(b).norm() / (n3.norm() * b.norm());
    double v3 = Utils::sigmoid(k3, (sin_theta3 - sin(alpha3)));

    // grad of v3
    Vector3d c = n3.cross(b);
    double c_norm = c.norm();
    double n3_norm = n3.norm();
    double b_norm = b.norm();
    Vector3d dsin_theta3_dn3 = (pow(-n3_norm, 2) * b.cross(c) - pow(c_norm, 2) * n3) / (pow(n3_norm, 3) * b_norm * c_norm);
    double dv3_dsin_theta3 = k3 * exp(-k3 * (sin_theta3 - sin(alpha3))) * pow(v3, 2);

    // Combine gradients using chain rule
    double dv3_dyaw = dv3_dsin_theta3 * dsin_theta3_dn3.dot(dn3_dyaw);

    // cost for single cell
    double cost_per_cell = 1 - (v1v2 * v3);
    cost = (cost * total_weight + cost_per_cell * w) / (total_weight + w);

    // grad for single cell
    double grad_per_cell = -v1v2 * dv3_dyaw;
    grad = (grad * total_weight + grad_per_cell * w) / (total_weight + w);

    total_weight += w;
  }
}

void YawTrajOptimizer::calcExplorationCost(double& cost, Eigen::Matrix3Xd& gradYaws) {
  cost = 0.0;

  double cost_i;
  double grad_i;

  for (size_t i = 0; i < piece_num - 1; ++i) {
    double yaw = yaws(0, i);
    Vector3d knots_pos = opt_data_->pos_vec_[i];
    Vector3d knots_acc = opt_data_->acc_vec_[i];

    ExplorationCostKnot(yaw, knots_pos, knots_acc, i, cost_i, grad_i);

    cost += param.ld_expl_ * cost_i;
    gradYaws(0, i) += param.ld_expl_ * grad_i;
  }
}

double YawTrajOptimizer::costFunctionCallback(void* ptr, const double* x, double* g, const int n) {
  // Step1: Read optimization data
  YawTrajOptimizer* obj = reinterpret_cast<YawTrajOptimizer*>(ptr);
  int piece_num = obj->piece_num;

  Eigen::Map<const Eigen::VectorXd> xi(x, piece_num - 1);
  Eigen::Map<Eigen::VectorXd> gradXi(g, piece_num - 1);

  // Step2: Xi -> Yaws
  obj->forwardYaw(xi, obj->yaws);

  // Step3.1: Calculate the energy cost
  double energy_cost;
  obj->minco.setParameters(obj->yaws, obj->times);
  obj->minco.getEnergy(energy_cost);
  obj->minco.getEnergyPartialGradByCoeffs(obj->gradCoeffs);

  // Step3.2: Calculate the feasibility cost
  double feasi_cost = 0.0;
  obj->calcFeasibilityCost(obj->times, feasi_cost, obj->param.cps_num_perPiece_);

  // Step4: grad(Coeffs) -> grad(Yaws)
  Eigen::VectorXd partialGradByTimes(piece_num);
  partialGradByTimes.setZero();
  Eigen::VectorXd gradByTimes;
  obj->minco.propogateGrad(obj->gradCoeffs, partialGradByTimes, obj->gradYaws, gradByTimes);

  // Step5: Calculate the explore cost
  double expl_cost = 0.0;
  obj->calcExplorationCost(expl_cost, obj->gradYaws);

  // Step6: grad(Yaws) -> grad(Xi)
  obj->backwardGradYaw(xi, obj->gradYaws, gradXi);

  double cost = energy_cost + feasi_cost + expl_cost;

  obj->gradYaws.setZero();
  obj->gradCoeffs.setZero();

  return cost;
}

}  // namespace perception_aware_planner