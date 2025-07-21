#ifndef PLAN_MANAGE_UTILS_HPP
#define PLAN_MANAGE_UTILS_HPP

#include <Eigen/Eigen>
#include <vector>
#include <set>
#include <iostream>

#include <ros/ros.h>

using Eigen::Matrix3d;
using Eigen::Matrix4d;
using Eigen::Quaterniond;
using Eigen::Vector2d;
using Eigen::Vector3d;
using std::set;
using std::vector;

namespace perception_aware_planner {

enum FrontierStatus {
  NOT_AVAILABLE,      // Can't be observed, even if change the yaw angle
  HAS_BEEN_OBSERVED,  // Has been observed by previous nodes
  AVAILABLE,          // May be observed by optimizing the yaw angle
  VISIBLE             // Can be observed by current node
};

struct YawOptData {
  using Ptr = std::shared_ptr<YawOptData>;

  vector<Vector3d> pos_vec_;
  vector<Vector3d> acc_vec_;

  vector<set<int>> target_covis_features_;

  vector<Vector3d> frontier_cells_;
  vector<vector<FrontierStatus>> frontier_status_;

  // Deprecated
  vector<char> final_goal_status_;
  vector<vector<Vector3d>> observed_features_;
};

enum CamType { FRONTIER, FEATURE };

class CameraParam {
public:
  using Ptr = std::shared_ptr<CameraParam>;

  double cx;
  double cy;
  double fx;
  double fy;
  int width;
  int height;
  double fov_horizontal;
  double fov_vertical;
  double visual_min;
  double visual_max;
  double fov_horizontal_rad_half;
  double fov_vertical_rad_half;
  Eigen::Matrix4d sensor2body;

  void init(ros::NodeHandle& nh, const CamType type) {

    std::string str = (type == FEATURE) ? "feature" : "frontier";
    std::string prefix = "camera/" + str + "/";

    nh.param(prefix + "visual_min", visual_min, 0.1);
    nh.param(prefix + "visual_max", visual_max, 10.0);
    nh.param(prefix + "cam_cx", cx, 321.046);
    nh.param(prefix + "cam_cy", cy, 243.449);
    nh.param(prefix + "cam_fx", fx, 387.229);
    nh.param(prefix + "cam_fy", fy, 387.229);
    nh.param(prefix + "cam_width", width, 640);
    nh.param(prefix + "cam_height", height, 480);

    std::vector<double> cam02body;
    if (nh.getParam(prefix + "cam02body", cam02body)) {
      if (cam02body.size() == 16) {
        sensor2body << cam02body[0], cam02body[1], cam02body[2], cam02body[3], cam02body[4], cam02body[5], cam02body[6],
            cam02body[7], cam02body[8], cam02body[9], cam02body[10], cam02body[11], cam02body[12], cam02body[13], cam02body[14],
            cam02body[15];
      } else {
        ROS_ERROR("Parameter 'cam02body' size is incorrect. "
                  "Expected "
                  "16 values.");
      }
    } else {
      ROS_ERROR("Failed to get parameter 'camera/cam02body'.");
    }

    fov_horizontal = 2 * atan(width / (2 * fx)) * 180 / M_PI;
    fov_vertical = 2 * atan(height / (2 * fy)) * 180 / M_PI;
    fov_vertical_rad_half = (fov_vertical / 2.0) * (M_PI / 180.0);
    fov_horizontal_rad_half = (fov_horizontal / 2.0) * (M_PI / 180.0);

    bool show_param;
    nh.param(prefix + "show_param", show_param, false);
    if (show_param) printParameters();
  }

  void printParameters() {
    std::cout << "------------------Camera Parameters------------------:" << std::endl;
    std::cout << "cx: " << cx << std::endl;
    std::cout << "cy: " << cy << std::endl;
    std::cout << "fx: " << fx << std::endl;
    std::cout << "fy: " << fy << std::endl;
    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;
    std::cout << "FOV Horizontal: " << fov_horizontal << " degrees" << std::endl;
    std::cout << "FOV Vertical: " << fov_vertical << " degrees" << std::endl;
    std::cout << "Camera Visual Max: " << visual_max << std::endl;
    std::cout << "Camera Visual Min: " << visual_min << std::endl;
    std::cout << "Half FOV Horizontal Rad: " << fov_horizontal_rad_half << std::endl;
    std::cout << "Half FOV Vertical Rad: " << fov_vertical_rad_half << std::endl;
    std::cout << "-----------------------------------------------------:" << std::endl;
  }

  bool inVisbleDepth(const Vector3d& camera_p, const Vector3d& target_p) {
    double dis = (target_p - camera_p).norm();
    return dis > visual_min && dis < visual_max;
  }

  bool inVisbleDepthAtLevel(const Vector3d& camera_p, const Vector3d& target_p) {
    if (!inVisbleDepth(camera_p, target_p)) return false;

    Eigen::Vector3d direction = target_p - camera_p;
    // Calculate the projection of the direction vector on the horizontal plane(x-y plane) and the angle
    Eigen::Vector3d horizontal_projection = direction;
    horizontal_projection.z() = 0;
    double angle = std::atan2(std::abs(direction.z()), horizontal_projection.norm());

    return (angle < fov_vertical_rad_half);
  }

  bool inFOV(const Vector3d& camera_p, const Vector3d& target_p) {
    return inVisbleDepth(camera_p, target_p);
  }

  bool inFOV(const Vector3d& camera_p, const Vector3d& target_p, const Quaterniond& camera_q) {
    // if (!inVisbleDepth(camera_p, target_p)) return false;

    Eigen::Vector3d target_in_camera = camera_q.inverse() * (target_p - camera_p);
    double x = target_in_camera.x();
    double y = target_in_camera.y();
    double z = target_in_camera.z();
    if (z <= visual_min || z >= visual_max) return false;
    double fov_x = atan2(x, z) * 180 / M_PI;
    double fov_y = atan2(y, z) * 180 / M_PI;
    bool within_horizontal_fov = std::abs(fov_x) <= fov_horizontal / 2.0;
    bool within_vertical_fov = std::abs(fov_y) <= fov_vertical / 2.0;

    return within_horizontal_fov && within_vertical_fov;
  }

  bool inFOVOdom(const Vector3d& odom_p, const double& odom_yaw, const Vector3d& target_p) {
    Eigen::Vector3d camera_pose;
    Eigen::Quaterniond camera_orient;
    fromOdom2Cam(odom_p, Eigen::Quaterniond(Eigen::AngleAxisd(odom_yaw, Eigen::Vector3d::UnitZ())), camera_pose, camera_orient);
    return inFOV(camera_pose, target_p, camera_orient);
  }

  void fromOdom2Cam(const Vector3d& odom_pos, const Quaterniond& odom_orient, Vector3d& camera_pos, Quaterniond& camera_orient) {
    Matrix4d Pose4d_receive = Matrix4d::Identity();
    Pose4d_receive.block<3, 3>(0, 0) = odom_orient.toRotationMatrix();
    Pose4d_receive.block<3, 1>(0, 3) = odom_pos;
    Matrix4d camera_Pose4d = Pose4d_receive * sensor2body;
    camera_pos = camera_Pose4d.block<3, 1>(0, 3);
    Eigen::Matrix3d cam_rot_matrix = camera_Pose4d.block<3, 3>(0, 0);
    camera_orient = Eigen::Quaterniond(cam_rot_matrix);
  }

  void fromOdom2Cam(const Vector3d& odom_pos, Vector3d& camera_pos) {
    Matrix4d Pose4d_receive = Eigen::Matrix4d::Identity();
    Pose4d_receive.block<3, 1>(0, 3) = odom_pos;
    Matrix4d camera_Pose4d = Pose4d_receive * sensor2body;
    camera_pos = camera_Pose4d.block<3, 1>(0, 3);
  }

  void getYawRangeUsingReferPos(
      const Vector3d& pos, const Vector3d& refer_pos, const vector<double>& sample_yaw, vector<bool>& yaw_available) {

    yaw_available.resize(sample_yaw.size());
    std::fill(yaw_available.begin(), yaw_available.end(), false);

    Eigen::Vector3d pos_transformed;
    fromOdom2Cam(pos, pos_transformed);
    if (!inVisbleDepthAtLevel(pos_transformed, refer_pos)) return;
    Eigen::Vector2d yaw_range = calculateYawRange(pos_transformed, refer_pos);

    for (size_t i = 0; i < sample_yaw.size(); ++i) {
      double yaw = sample_yaw[i];
      // Case1：yaw_range(0) < yaw_range(1)，not jump across “π”
      if (yaw_range(0) < yaw_range(1)) {
        if (yaw >= yaw_range(0) && yaw <= yaw_range(1)) {
          yaw_available[i] = true;
        }
      }
      // Case2：yaw_range(0) > yaw_range(1)，jump across “π”
      else {
        if (yaw >= yaw_range(0) || yaw <= yaw_range(1)) {
          yaw_available[i] = true;
        }
      }
    }
  }

  // Calculate the feasible yaw range from camera_p to target_p
  Eigen::Vector2d calculateYawRange(const Eigen::Vector3d& camera_p, const Eigen::Vector3d& target_p) {
    Eigen::Vector2d relative_position_xy(target_p.x() - camera_p.x(), target_p.y() - camera_p.y());
    double yaw_angle = atan2(relative_position_xy.y(), relative_position_xy.x());

    double min_yaw = yaw_angle - fov_horizontal_rad_half;
    double max_yaw = yaw_angle + fov_horizontal_rad_half;
    while (min_yaw < -M_PI) {
      min_yaw += 2 * M_PI;
    }
    while (max_yaw > M_PI) {
      max_yaw -= 2 * M_PI;
    }

    return Eigen::Vector2d(min_yaw, max_yaw);
  }
};

struct GlobalParam {
  // Localizability constraints
  int min_feature_num_act_;
  int min_covisible_feature_num_act_;
  int min_feature_num_plan_;
  int min_covisible_feature_num_plan_;
  double max_tol_time_;

  // Feasibility constraints
  double max_vel_;
  double max_acc_;
  double max_yaw_rate_;

  // Param of camera
  CameraParam::Ptr frontier_cam_ = nullptr;
  CameraParam::Ptr feature_cam_ = nullptr;
};

class Utils {
public:
  static void initialize(ros::NodeHandle& nh) {
    nh.param("global/min_feature_num_act", param_g_.min_feature_num_act_, -1);
    nh.param("global/min_covisible_feature_num_act", param_g_.min_covisible_feature_num_act_, -1);
    nh.param("global/min_feature_num_plan", param_g_.min_feature_num_plan_, -1);
    nh.param("global/min_covisible_feature_num_plan", param_g_.min_covisible_feature_num_plan_, -1);
    nh.param("global/max_tol_time", param_g_.max_tol_time_, -1.0);

    nh.param("global/max_vel", param_g_.max_vel_, -1.0);
    nh.param("global/max_acc", param_g_.max_acc_, -1.0);
    nh.param("global/max_yaw_rate", param_g_.max_yaw_rate_, -1.0);

    param_g_.frontier_cam_.reset(new CameraParam());
    param_g_.frontier_cam_->init(nh, FRONTIER);

    param_g_.feature_cam_.reset(new CameraParam());
    param_g_.feature_cam_->init(nh, FEATURE);
  }

  static GlobalParam getGlobalParam() {
    return param_g_;
  }

  static double sigmoid(const double k, const double x) {
    return 1 / (1 + std::exp(-k * x));
  }

  static Quaterniond calcOrientation(const double& yaw, const Vector3d& acc) {

    Vector3d thrust_dir = getThrustDirection(acc);
    Vector3d ny(cos(yaw), sin(yaw), 0);
    Vector3d yB = thrust_dir.cross(ny).normalized();
    Vector3d xB = yB.cross(thrust_dir).normalized();

    const Matrix3d R_W_B((Matrix3d() << xB, yB, thrust_dir).finished());
    const Quaterniond desired_attitude(R_W_B);

    return desired_attitude;
  }

  static void roundPi(double& value) {
    while (value < -M_PI) value += 2 * M_PI;
    while (value > M_PI) value -= 2 * M_PI;
  }

  static void calcNextYaw(const double& last_yaw, double& yaw) {
    // round yaw to [-PI, PI]
    double round_last = last_yaw;
    roundPi(round_last);

    double diff = yaw - round_last;
    if (fabs(diff) <= M_PI) {
      yaw = last_yaw + diff;
    } else if (diff > M_PI) {
      yaw = last_yaw + diff - 2 * M_PI;
    } else if (diff < -M_PI) {
      yaw = last_yaw + diff + 2 * M_PI;
    }
  }

  static int getSameCount(const std::set<int>& s1, const std::set<int>& s2) {
    std::set<int> s;
    s.insert(s1.begin(), s1.end());
    s.insert(s2.begin(), s2.end());

    int ret = s1.size() + s2.size() - s.size();
    return ret;
  }

  static Vector3d calculateTopMiddlePoint(const vector<Vector3d>& points) {
    if (points.empty()) {
      std::cerr << "Error: The input vector is empty!" << std::endl;
      return Eigen::Vector3d::Zero();
    }

    double x_sum = 0.0, y_sum = 0.0;
    double max_z = points[0].z();
    for (const auto& point : points) {
      x_sum += point.x();
      y_sum += point.y();
      if (point.z() > max_z) max_z = point.z();
    }
    double x_avg = x_sum / points.size();
    double y_avg = y_sum / points.size();
    return Vector3d(x_avg, y_avg, max_z);
  }

private:
  static Vector3d getThrustDirection(const Vector3d& acc) {
    Vector3d gravity(0, 0, -9.81);
    Vector3d thrust_dir = (acc - gravity).normalized();
    return thrust_dir;
  }

  static GlobalParam param_g_;
};

}  // namespace perception_aware_planner

#endif