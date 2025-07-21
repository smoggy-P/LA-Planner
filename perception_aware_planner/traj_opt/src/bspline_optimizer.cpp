#include "traj_opt/bspline_optimizer.h"
#include "traj_opt/lbfgs.hpp"

#include "plan_env/edt_environment.h"
#include "plan_env/sdf_map.h"

#include "utils/utils.h"

#include <nlopt.hpp>

using namespace std;
using namespace Eigen;

namespace perception_aware_planner {

const int BsplineOptimizer::SMOOTHNESS = (1 << 1);
const int BsplineOptimizer::DISTANCE = (1 << 2);
const int BsplineOptimizer::FEASIBILITY = (1 << 3);
const int BsplineOptimizer::START = (1 << 4);
const int BsplineOptimizer::END = (1 << 5);
const int BsplineOptimizer::GUIDE = (1 << 6);
const int BsplineOptimizer::WAYPOINTS = (1 << 7);
const int BsplineOptimizer::MINTIME = (1 << 8);

const int BsplineOptimizer::GUIDE_PHASE =
    BsplineOptimizer::SMOOTHNESS | BsplineOptimizer::GUIDE | BsplineOptimizer::START | BsplineOptimizer::END;

void BsplineOptimizer::setParam(ros::NodeHandle& nh) {

  nh.param("optimization/ld_smooth", ld_smooth_, -1.0);
  nh.param("optimization/ld_dist", ld_dist_, -1.0);
  nh.param("optimization/ld_feasi", ld_feasi_, -1.0);
  nh.param("optimization/ld_start", ld_start_, -1.0);
  nh.param("optimization/ld_end", ld_end_, -1.0);
  nh.param("optimization/ld_guide", ld_guide_, -1.0);
  nh.param("optimization/ld_waypt", ld_waypt_, -1.0);
  nh.param("optimization/ld_time", ld_time_, -1.0);

  nh.param("optimization/dist0", dist0_, -1.0);
  nh.param("optimization/max_vel", max_vel_, -1.0);
  nh.param("optimization/max_acc", max_acc_, -1.0);
  nh.param("optimization/dlmin", dlmin_, -1.0);
  nh.param("optimization/wnl", wnl_, -1.0);

  nh.param("optimization/max_iteration_num1", max_iteration_num_[0], -1);
  nh.param("optimization/max_iteration_num2", max_iteration_num_[1], -1);
  nh.param("optimization/max_iteration_num3", max_iteration_num_[2], -1);
  nh.param("optimization/max_iteration_num4", max_iteration_num_[3], -1);
  nh.param("optimization/max_iteration_time1", max_iteration_time_[0], -1.0);
  nh.param("optimization/max_iteration_time2", max_iteration_time_[1], -1.0);
  nh.param("optimization/max_iteration_time3", max_iteration_time_[2], -1.0);
  nh.param("optimization/max_iteration_time4", max_iteration_time_[3], -1.0);

  nh.param("optimization/algorithm1", algorithm1_, -1);
  nh.param("optimization/algorithm2", algorithm2_, -1);
  nh.param("optimization/use_lbfgs", use_lbfgs_, false);
  nh.param("manager/bspline_degree", bspline_degree_, 3);
}

void BsplineOptimizer::setEnvironment(const EDTEnvironment::Ptr& env) {
  this->edt_environment_ = env;
  dynamic_ = false;
}

void BsplineOptimizer::setCostFunction(const int& cost_code) {
  cost_function_ = cost_code;
}

void BsplineOptimizer::setGuidePath(const vector<Eigen::Vector3d>& guide_pt) {
  guide_pts_ = guide_pt;
}

void BsplineOptimizer::setWaypoints(const vector<Eigen::Vector3d>& waypts, const vector<int>& waypt_idx) {
  waypoints_ = waypts;
  waypt_idx_ = waypt_idx;
}

void BsplineOptimizer::enableDynamic(double time_start) {
  dynamic_ = true;
  start_time_ = time_start;
}

void BsplineOptimizer::setBoundaryStates(const vector<Eigen::Vector3d>& start, const vector<Eigen::Vector3d>& end) {
  start_state_ = start;
  end_state_ = end;
}

void BsplineOptimizer::setBoundaryStates(
    const vector<Vector3d>& start, const vector<Vector3d>& end, const vector<bool>& start_idx, const vector<bool>& end_idx) {
  start_state_ = start;
  end_state_ = end;
  start_con_index_ = start_idx;
  end_con_index_ = end_idx;
}

void BsplineOptimizer::setTimeLowerBound(const double& lb) {
  time_lb_ = lb;
}

void BsplineOptimizer::resetCostAndGrad() {
  f_smoothness_ = 0.0;
  f_distance_ = 0.0;
  f_feasibility_ = 0.0;
  f_start_ = 0.0;
  f_end_ = 0.0;
  f_guide_ = 0.0;
  f_waypoints_ = 0.0;
  f_time_ = 0.0;

  g_q_.resize(point_num_);
  g_smoothness_.resize(point_num_);
  g_distance_.resize(point_num_);
  g_feasibility_.resize(point_num_);
  g_start_.resize(point_num_);
  g_end_.resize(point_num_);
  g_guide_.resize(point_num_);
  g_waypoints_.resize(point_num_);
  g_time_.resize(point_num_);

  std::fill(g_q_.begin(), g_q_.end(), Eigen::Vector3d::Zero());
  std::fill(g_smoothness_.begin(), g_smoothness_.end(), Eigen::Vector3d::Zero());
  std::fill(g_distance_.begin(), g_distance_.end(), Eigen::Vector3d::Zero());
  std::fill(g_feasibility_.begin(), g_feasibility_.end(), Eigen::Vector3d::Zero());
  std::fill(g_start_.begin(), g_start_.end(), Eigen::Vector3d::Zero());
  std::fill(g_end_.begin(), g_end_.end(), Eigen::Vector3d::Zero());
  std::fill(g_guide_.begin(), g_guide_.end(), Eigen::Vector3d::Zero());
  std::fill(g_waypoints_.begin(), g_waypoints_.end(), Eigen::Vector3d::Zero());
  std::fill(g_time_.begin(), g_time_.end(), Eigen::Vector3d::Zero());
}

void BsplineOptimizer::optimize(
    Eigen::MatrixXd& points, double& dt, const int& cost_function, const int& max_num_id, const int& max_time_id) {

  if (start_state_.empty()) {
    ROS_ERROR("Initial state undefined!");
    return;
  }

  control_points_ = points;
  knot_span_ = dt;
  max_num_id_ = max_num_id;
  max_time_id_ = max_time_id;
  setCostFunction(cost_function);

  // Set necessary data and flag
  dim_ = control_points_.cols();
  if (dim_ == 1)
    order_ = 3;
  else
    order_ = bspline_degree_;
  point_num_ = control_points_.rows();
  optimize_time_ = cost_function_ & MINTIME;
  variable_num_ = optimize_time_ ? dim_ * point_num_ + 1 : dim_ * point_num_;
  if (variable_num_ <= 0) {
    ROS_ERROR("Empty varibale to optimization solver.");
    return;
  }

  pt_dist_ = 0.0;
  for (int i = 0; i < control_points_.rows() - 1; ++i) {
    pt_dist_ += (control_points_.row(i + 1) - control_points_.row(i)).norm();
  }
  pt_dist_ /= point_num_;

  iter_num_ = 0;
  min_cost_ = std::numeric_limits<double>::max();

  resetCostAndGrad();

  // optimize();
  if (use_lbfgs_)
    optimize_lbfgs();
  else
    optimize();

  points = control_points_;
  dt = knot_span_;
  start_state_.clear();
  start_con_index_.clear();
  end_con_index_.clear();
  time_lb_ = -1;
}

void BsplineOptimizer::optimize() {
  // Optimize all control points and maybe knot span dt
  // Use NLopt solver

  nlopt::opt opt(nlopt::algorithm(isQuadratic() ? algorithm1_ : algorithm2_), variable_num_);
  opt.set_min_objective(BsplineOptimizer::costFunction, this);
  opt.set_xtol_rel(1e-4);

  // opt.set_maxeval(max_iteration_num_[max_num_id_]);
  // opt.set_maxtime(max_iteration_time_[max_time_id_]);
  // opt.set_xtol_rel(1e-5);

  // Set axis aligned bounding box for optimization
  Eigen::Vector3d bmin, bmax;
  edt_environment_->sdf_map_->getBox(bmin, bmax);
  for (int k = 0; k < 3; ++k) {
    bmin[k] += 0.1;
    bmax[k] -= 0.1;
    // cout << "k: " << k << " bmin[k]: " << bmin[k] << " bmax[k]: " << bmax[k] << endl;
  }

  vector<double> q(variable_num_);

  // Variables for control points
  for (int i = 0; i < point_num_; ++i) {
    for (int j = 0; j < dim_; ++j) {
      double cij = control_points_(i, j);
      if (dim_ != 1) cij = max(min(cij, bmax[j % 3]), bmin[j % 3]);
      q[dim_ * i + j] = cij;
    }
  }

  // Variables for knot span
  if (optimize_time_) q[variable_num_ - 1] = knot_span_;

  if (dim_ != 1) {
    vector<double> lb(variable_num_), ub(variable_num_);
    const double bound = 10.0;
    for (int i = 0; i < 3 * point_num_; ++i) {
      lb[i] = q[i] - bound;
      ub[i] = q[i] + bound;
      lb[i] = max(lb[i], bmin[i % 3]);
      ub[i] = min(ub[i], bmax[i % 3]);
    }
    if (optimize_time_) {
      lb[variable_num_ - 1] = 0.0;
      ub[variable_num_ - 1] = 5.0;
    }
    opt.set_lower_bounds(lb);
    opt.set_upper_bounds(ub);
  }

  double final_cost;
  try {
    opt.optimize(q, final_cost);
    for (int i = 0; i < point_num_; ++i)
      for (int j = 0; j < dim_; ++j) control_points_(i, j) = best_variable_[dim_ * i + j];
    if (optimize_time_) knot_span_ = best_variable_[variable_num_ - 1];

    issuccess = true;
  }

  catch (std::exception& e) {
    cout << e.what() << endl;
    issuccess = false;
    ROS_ERROR("[BsplineOptimizer::optimize] Optimized_fail!!!!!!! final_cost: %.4f", final_cost);
    for (int i = 0; i < point_num_; ++i)
      for (int j = 0; j < dim_; ++j) control_points_(i, j) = best_variable_[dim_ * i + j];
    if (optimize_time_) knot_span_ = best_variable_[variable_num_ - 1];
  }
}

bool BsplineOptimizer::optimize_lbfgs() {
  // Refer to the lbfgs optimizer of ego-planner
  Eigen::Vector3d bmin, bmax;
  edt_environment_->sdf_map_->getBox(bmin, bmax);
  for (int k = 0; k < 3; ++k) {
    bmin[k] += 0.1;
    bmax[k] -= 0.1;
  }
  double final_cost;
  bool flag_safe = true;
  // do
  {
    double q[variable_num_];
    for (int i = 0; i < point_num_; ++i) {
      for (int j = 0; j < dim_; ++j) {
        double cij = control_points_(i, j);
        if (dim_ != 1) cij = std::max(std::min(cij, bmax[j % 3]), bmin[j % 3]);
        q[dim_ * i + j] = cij;
      }
    }
    q[variable_num_ - 1] = knot_span_;

    // Initialize the lbfgs optimizer
    lbfgs::lbfgs_parameter_t lbfgs_params;
    lbfgs::lbfgs_load_default_parameters(&lbfgs_params);

    lbfgs_params.mem_size = 15;
    lbfgs_params.max_iterations = 300;
    lbfgs_params.g_epsilon = 10;
    lbfgs_params.max_step = 1e+20;
    lbfgs_params.min_step = 1e-20;

    int result = lbfgs::lbfgs_optimize(
        variable_num_, q, &final_cost, BsplineOptimizer::costFunctionLBFGS, NULL, NULL, this, &lbfgs_params);

    switch (result) {
      case lbfgs::LBFGS_CONVERGENCE:
      case lbfgs::LBFGSERR_MAXIMUMITERATION:
      case lbfgs::LBFGS_ALREADY_MINIMIZED:
      case lbfgs::LBFGS_STOP:
      // Note: These error codes are considered to be acceptable
      case lbfgs::LBFGSERR_ROUNDING_ERROR:
      case lbfgs::LBFGSERR_MINIMUMSTEP: {
        // pass
        break;
      }

      default: {
        flag_safe = false;
        ROS_ERROR("[bspline_opt]: Solver error in optimize_lbfgs!, return = %d, %s", result, lbfgs::lbfgs_strerror(result));
        break;
      }
    }

    for (int i = 0; i < point_num_; ++i)
      for (int j = 0; j < dim_; ++j) control_points_(i, j) = best_variable_[dim_ * i + j];
    if (optimize_time_) knot_span_ = best_variable_[variable_num_ - 1];
    issuccess = flag_safe;
    return flag_safe;
  }
}

void BsplineOptimizer::calcSmoothnessCost(const vector<Vector3d>& q, double& cost, vector<Vector3d>& gradient_q) {
  cost = 0.0;
  Eigen::Vector3d zero(0, 0, 0);
  std::fill(gradient_q.begin(), gradient_q.end(), zero);
  Eigen::Vector3d jerk, temp_j;

  for (size_t i = 0; i < q.size() - 3; i++) {
    /* evaluate jerk */
    // 3-rd order derivative = 1/(ts)^3*(q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i])

    // Test jerk cost
    Eigen::Vector3d ji = (q[i + 3] - 3 * q[i + 2] + 3 * q[i + 1] - q[i]) / pt_dist_;
    double cost_this = ji.squaredNorm();
    cost += cost_this;
    temp_j = 2 * ji / pt_dist_;

    gradient_q[i + 0] += -temp_j;
    gradient_q[i + 1] += 3.0 * temp_j;
    gradient_q[i + 2] += -3.0 * temp_j;
    gradient_q[i + 3] += temp_j;
  }
}

void BsplineOptimizer::calcDistanceCost(const vector<Eigen::Vector3d>& q, double& cost, vector<Eigen::Vector3d>& gradient_q) {
  cost = 0.0;
  Eigen::Vector3d zero(0, 0, 0);
  std::fill(gradient_q.begin(), gradient_q.end(), zero);

  double dist;
  Eigen::Vector3d dist_grad;
  for (size_t i = 0; i < q.size(); i++) {
    edt_environment_->evaluateEDTWithGrad(q[i], -1.0, dist, dist_grad);
    if (dist_grad.norm() > 1e-4) dist_grad.normalize();

    if (dist < dist0_) {
      cost += pow(dist - dist0_, 2);
      gradient_q[i] += 2.0 * (dist - dist0_) * dist_grad;
    }
  }
}

void BsplineOptimizer::calcFeasibilityCost(
    const vector<Vector3d>& q, const double& dt, double& cost, vector<Vector3d>& gradient_q, double& gt) {

  cost = 0.0;
  Eigen::Vector3d zero(0, 0, 0);
  std::fill(gradient_q.begin(), gradient_q.end(), zero);
  gt = 0.0;

  // Abbreviation of params
  const double dt_inv = 1 / dt;
  const double dt_inv2 = dt_inv * dt_inv;
  for (size_t i = 0; i < q.size() - 1; ++i) {
    // Control point of velocity
    Eigen::Vector3d vi = (q[i + 1] - q[i]) * dt_inv;
    for (int k = 0; k < 3; ++k) {
      // Calculate cost for each axis
      double vd = fabs(vi[k]) - max_vel_;
      if (vd > 0.0) {
        cost += pow(vd, 2);
        double sign = vi[k] > 0 ? 1.0 : -1.0;
        double tmp = 2 * vd * sign * dt_inv;
        gradient_q[i][k] += -tmp;
        gradient_q[i + 1][k] += tmp;
        if (optimize_time_) gt += tmp * (-vi[k]);
      }
    }
  }

  // Acc feasibility cost
  for (size_t i = 0; i < q.size() - 2; ++i) {
    Eigen::Vector3d ai = (q[i + 2] - 2 * q[i + 1] + q[i]) * dt_inv2;
    for (int k = 0; k < 3; ++k) {
      double ad = fabs(ai[k]) - max_acc_;
      if (ad > 0.0) {
        cost += pow(ad, 2);
        double sign = ai[k] > 0 ? 1.0 : -1.0;
        double tmp = 2 * ad * sign * dt_inv2;
        gradient_q[i][k] += tmp;
        gradient_q[i + 1][k] += -2 * tmp;
        gradient_q[i + 2][k] += tmp;
        if (optimize_time_) gt += tmp * ai[k] * (-2) * dt;
      }
    }
  }
}

void BsplineOptimizer::calcStartCost(
    const vector<Vector3d>& q, const double& dt, double& cost, vector<Vector3d>& gradient_q, double& gt) {

  if (start_con_index_.size() != 3) ROS_ERROR("Start state constraint is not set!");

  cost = 0.0;
  Eigen::Vector3d zero(0, 0, 0);
  // std::fill(gradient_q.begin(), gradient_q.end(), zero);
  for (int i = 0; i < 3; ++i) gradient_q[i] = zero;
  gt = 0.0;

  Eigen::Vector3d q1, q2, q3, dq;
  q1 = q[0];
  q2 = q[1];
  q3 = q[2];

  // Start position
  if (start_con_index_[0]) {
    if (start_state_.size() < 1) ROS_ERROR_STREAM("(start pos),start state size: " << start_state_.size());

    static const double w_pos = 10.0;
    dq = 1 / 6.0 * (q1 + 4 * q2 + q3) - start_state_[0];
    cost += w_pos * dq.squaredNorm();
    gradient_q[0] += w_pos * 2 * dq * (1 / 6.0);
    gradient_q[1] += w_pos * 2 * dq * (4 / 6.0);
    gradient_q[2] += w_pos * 2 * dq * (1 / 6.0);
  }

  // Start velocity
  if (start_con_index_[1]) {
    if (start_state_.size() < 2) ROS_ERROR_STREAM("(start vel),start state size: " << start_state_.size());

    dq = 1 / (2 * dt) * (q3 - q1) - start_state_[1];
    cost += dq.squaredNorm();
    gradient_q[0] += 2 * dq * (-1.0) / (2 * dt);
    gradient_q[2] += 2 * dq * 1.0 / (2 * dt);
    if (optimize_time_) gt += dq.dot(q3 - q1) / (-dt * dt);
  }

  // Start acceleration
  if (start_con_index_[2]) {
    if (start_state_.size() < 3) ROS_ERROR_STREAM("(start acc),start state size: " << start_state_.size());

    dq = 1 / (dt * dt) * (q1 - 2 * q2 + q3) - start_state_[2];
    cost += dq.squaredNorm();
    gradient_q[0] += 2 * dq * 1.0 / (dt * dt);
    gradient_q[1] += 2 * dq * (-2.0) / (dt * dt);
    gradient_q[2] += 2 * dq * 1.0 / (dt * dt);
    if (optimize_time_) gt += dq.dot(q1 - 2 * q2 + q3) / (-dt * dt * dt);
  }
}

void BsplineOptimizer::calcEndCost(
    const vector<Vector3d>& q, const double& dt, double& cost, vector<Vector3d>& gradient_q, double& gt) {

  if (end_con_index_.size() != 3) ROS_ERROR("End state constraint is not set!");

  cost = 0.0;
  Eigen::Vector3d zero(0, 0, 0);
  // std::fill(gradient_q.begin(), gradient_q.end(), zero);
  for (size_t i = q.size() - 3; i < q.size(); ++i) gradient_q[i] = zero;
  gt = 0.0;

  Eigen::Vector3d q_3, q_2, q_1, dq;
  q_3 = q[q.size() - 3];
  q_2 = q[q.size() - 2];
  q_1 = q[q.size() - 1];

  // End position
  if (end_con_index_[0]) {
    if (end_state_.size() < 1) ROS_ERROR_STREAM("(end pos),end state size: " << end_state_.size());

    dq = 1 / 6.0 * (q_1 + 4 * q_2 + q_3) - end_state_[0];
    cost += dq.squaredNorm();
    gradient_q[q.size() - 1] += 2 * dq * (1 / 6.0);
    gradient_q[q.size() - 2] += 2 * dq * (4 / 6.0);
    gradient_q[q.size() - 3] += 2 * dq * (1 / 6.0);
  }

  // End velocity
  if (end_con_index_[1]) {
    if (end_state_.size() < 2) ROS_ERROR_STREAM("(end vel),end state size: " << end_state_.size());

    dq = 1 / (2 * dt) * (q_1 - q_3) - end_state_[1];
    cost += dq.squaredNorm();
    gradient_q[q.size() - 1] += 2 * dq * 1.0 / (2 * dt);
    gradient_q[q.size() - 3] += 2 * dq * (-1.0) / (2 * dt);
    if (optimize_time_) gt += dq.dot(q_1 - q_3) / (-dt * dt);
  }

  // End acceleration
  if (end_con_index_[2]) {
    if (end_state_.size() < 3) ROS_ERROR_STREAM("(end acc),end state size: " << end_state_.size());

    dq = 1 / (dt * dt) * (q_1 - 2 * q_2 + q_3) - end_state_[2];
    cost += dq.squaredNorm();
    gradient_q[q.size() - 1] += 2 * dq * 1.0 / (dt * dt);
    gradient_q[q.size() - 2] += 2 * dq * (-2.0) / (dt * dt);
    gradient_q[q.size() - 3] += 2 * dq * 1.0 / (dt * dt);
    if (optimize_time_) gt += dq.dot(q_1 - 2 * q_2 + q_3) / (-dt * dt * dt);
  }
}

void BsplineOptimizer::calcWaypointsCost(const vector<Eigen::Vector3d>& q, double& cost, vector<Eigen::Vector3d>& gradient_q) {
  cost = 0.0;
  Eigen::Vector3d zero(0, 0, 0);
  std::fill(gradient_q.begin(), gradient_q.end(), zero);

  Vector3d q1, q2, q3, dq;

  for (size_t i = 0; i < waypoints_.size(); ++i) {
    Vector3d waypt = waypoints_[i];
    int idx = waypt_idx_[i];

    q1 = q[idx];
    q2 = q[idx + 1];
    q3 = q[idx + 2];

    dq = (q1 + 4 * q2 + q3) / 6 - waypt;
    cost += dq.squaredNorm();

    gradient_q[idx] += dq * (2.0 / 6.0);      // 2*dq*(1/6)
    gradient_q[idx + 1] += dq * (8.0 / 6.0);  // 2*dq*(4/6)
    gradient_q[idx + 2] += dq * (2.0 / 6.0);
  }
}

/* use the uniformly sampled points on a geomertic path to guide the
 * trajectory. For each control points to be optimized, it is assigned a
 * guiding point on the path and the distance between them is penalized */
void BsplineOptimizer::calcGuideCost(const vector<Eigen::Vector3d>& q, double& cost, vector<Eigen::Vector3d>& gradient_q) {
  cost = 0.0;
  Eigen::Vector3d zero(0, 0, 0);
  std::fill(gradient_q.begin(), gradient_q.end(), zero);

  int end_idx = q.size() - order_;

  for (int i = order_; i < end_idx; i++) {
    Vector3d gpt = guide_pts_[i - order_];
    cost += (q[i] - gpt).squaredNorm();
    gradient_q[i] += 2 * (q[i] - gpt);
  }
}

void BsplineOptimizer::calcTimeCost(const double& dt, double& cost, double& gt) {
  // Min time
  double duration = (point_num_ - order_) * dt;
  cost = duration;
  gt = double(point_num_ - order_);

  // Time lower bound
  if (time_lb_ > 0 && duration < time_lb_) {
    static const double w_lb = 10;
    cost += w_lb * pow(duration - time_lb_, 2);
    gt += w_lb * 2 * (duration - time_lb_) * (point_num_ - order_);
  }
}

void BsplineOptimizer::combineCost(const std::vector<double>& x, vector<double>& grad, double& f_combine) {

  for (int i = 0; i < point_num_; ++i) {
    for (int j = 0; j < dim_; ++j) g_q_[i][j] = x[dim_ * i + j];
    for (int j = dim_; j < 3; ++j) g_q_[i][j] = 0.0;
  }
  const double dt = optimize_time_ ? x[variable_num_ - 1] : knot_span_;

  f_combine = 0.0;
  grad.resize(variable_num_);
  fill(grad.begin(), grad.end(), 0.0);

  // Cost1：Smoothness Cost
  if (cost_function_ & SMOOTHNESS) {
    calcSmoothnessCost(g_q_, f_smoothness_, g_smoothness_);
    f_combine += ld_smooth_ * f_smoothness_;
    for (int i = 0; i < point_num_; i++)
      for (int j = 0; j < dim_; j++) grad[dim_ * i + j] += ld_smooth_ * g_smoothness_[i](j);
  }

  // Cost2：Collision Cost
  if (cost_function_ & DISTANCE) {
    calcDistanceCost(g_q_, f_distance_, g_distance_);
    f_combine += ld_dist_ * f_distance_;
    for (int i = 0; i < point_num_; i++)
      for (int j = 0; j < dim_; j++) grad[dim_ * i + j] += ld_dist_ * g_distance_[i](j);
  }

  // Cost3：Physical Feasibility Cost
  if (cost_function_ & FEASIBILITY) {
    double gt_feasibility = 0.0;
    calcFeasibilityCost(g_q_, dt, f_feasibility_, g_feasibility_, gt_feasibility);
    f_combine += ld_feasi_ * f_feasibility_;
    for (int i = 0; i < point_num_; i++)
      for (int j = 0; j < dim_; j++) grad[dim_ * i + j] += ld_feasi_ * g_feasibility_[i](j);
    if (optimize_time_) grad[variable_num_ - 1] += ld_feasi_ * gt_feasibility;
  }

  // Cost4：Start State Cost
  if (cost_function_ & START) {
    double gt_start = 0.0;
    calcStartCost(g_q_, dt, f_start_, g_start_, gt_start);
    f_combine += ld_start_ * f_start_;
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < dim_; j++) grad[dim_ * i + j] += ld_start_ * g_start_[i](j);
    if (optimize_time_) grad[variable_num_ - 1] += ld_start_ * gt_start;
  }

  // Cost5：End State Cost
  if (cost_function_ & END) {
    double gt_end = 0.0;
    calcEndCost(g_q_, dt, f_end_, g_end_, gt_end);
    f_combine += ld_end_ * f_end_;
    for (int i = point_num_ - 3; i < point_num_; i++)
      for (int j = 0; j < dim_; j++) grad[dim_ * i + j] += ld_end_ * g_end_[i](j);

    if (optimize_time_) grad[variable_num_ - 1] += ld_end_ * gt_end;
  }

  // Cost6：Guide Cost(Control Points)
  if (cost_function_ & GUIDE) {
    calcGuideCost(g_q_, f_guide_, g_guide_);
    f_combine += ld_guide_ * f_guide_;
    for (int i = 0; i < point_num_; i++)
      for (int j = 0; j < dim_; j++) grad[dim_ * i + j] += ld_guide_ * g_guide_[i](j);
  }

  // Cost7：Waypoints Cost(Knot Points)
  if (cost_function_ & WAYPOINTS) {
    calcWaypointsCost(g_q_, f_waypoints_, g_waypoints_);
    f_combine += ld_waypt_ * f_waypoints_;
    for (int i = 0; i < point_num_; i++)
      for (int j = 0; j < dim_; j++) grad[dim_ * i + j] += ld_waypt_ * g_waypoints_[i](j);
  }

  // Cost8：Minimize Time Cost
  if (cost_function_ & MINTIME) {
    double gt_time = 0.0;
    calcTimeCost(dt, f_time_, gt_time);
    f_combine += ld_time_ * f_time_;
    grad[variable_num_ - 1] += ld_time_ * gt_time;
  }
}

double BsplineOptimizer::costFunction(const std::vector<double>& x, std::vector<double>& grad, void* func_data) {
  BsplineOptimizer* opt = reinterpret_cast<BsplineOptimizer*>(func_data);
  double cost;
  opt->combineCost(x, grad, cost);
  opt->iter_num_++;

  /* save the min cost result */
  if (cost < opt->min_cost_) {
    opt->min_cost_ = cost;
    opt->best_variable_ = x;
  }
  return cost;
}

double BsplineOptimizer::costFunctionLBFGS(void* func_data, const double* x, double* grad, const int n) {
  BsplineOptimizer* opt = reinterpret_cast<BsplineOptimizer*>(func_data);

  std::vector<double> x_vec(x, x + n);
  std::vector<double> grad_vec(n);

  double f_combine;
  opt->combineCost(x_vec, grad_vec, f_combine);

  memcpy(grad, grad_vec.data(), n * sizeof(grad[0]));

  opt->iter_num_++;

  if (f_combine < opt->min_cost_) {
    opt->min_cost_ = f_combine;
    opt->best_variable_ = x_vec;  // store the best variable
  }

  return f_combine;
}

Eigen::MatrixXd BsplineOptimizer::getControlPoints() {
  return this->control_points_;
}

bool BsplineOptimizer::isQuadratic() {
  if (cost_function_ == GUIDE_PHASE) {
    return true;
  }

  else if (cost_function_ == SMOOTHNESS) {
    return true;
  }

  else if (cost_function_ == (SMOOTHNESS | WAYPOINTS)) {
    return true;
  }

  return false;
}

}  // namespace perception_aware_planner