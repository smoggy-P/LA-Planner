#include "local_plan_manager/backward.hpp"

#include "quadrotor_msgs/PositionCommand.h"

#include "traj_utils/non_uniform_bspline.h"
#include "traj_utils/MixTraj.h"
#include "traj_utils/PolyTraj.h"
#include "traj_utils/polynomial_traj.h"
#include "traj_utils/pub_fov_marker_tf.h"

#include "gcopter/gcopter.hpp"

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Float64.h>
#include <visualization_msgs/Marker.h>

namespace backward {
backward::SignalHandling sh;
}

using namespace std;
using namespace Eigen;

using perception_aware_planner::NonUniformBspline;

ros::Publisher pos_cmd_pub, traj_pub;
nav_msgs::Odometry odom;
quadrotor_msgs::PositionCommand cmd;
quadrotor_msgs::PositionCommand emergency_stop_cmd;
bool flag_emergency_stop = false;

// Info of generated traj
vector<NonUniformBspline> traj_;
std::shared_ptr<Trajectory<5>> yaw_traj_ = nullptr;

double traj_duration_;
ros::Time start_time_;
int traj_id_ = 0;

// Info of replan
bool receive_traj_ = false;
double replan_time_;
double replan_out_;

// Executed traj, commanded and real ones
vector<Eigen::Vector3d> traj_cmd_, traj_real_;
vector<Eigen::Quaterniond> traj_real_yaw_;

// Pub FOV marker
FOVMarker* fovMarkerPtr = nullptr;

typedef Eigen::Matrix<double, 3, 6> CoefficientMat;
typedef Eigen::Matrix<double, 3, 5> VelCoefficientMat;
typedef Eigen::Matrix<double, 3, 4> AccCoefficientMat;

void displayTrajWithColor(const vector<Vector3d>& path, const double resolution, const Vector4d& color) {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::SPHERE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = 0;
  traj_pub.publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;
  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);
  mk.scale.x = resolution;
  mk.scale.y = resolution;
  mk.scale.z = resolution;
  geometry_msgs::Point pt;
  for (int i = 0; i < int(path.size()); i++) {
    pt.x = path[i](0);
    pt.y = path[i](1);
    pt.z = path[i](2);
    mk.points.push_back(pt);
  }
  traj_pub.publish(mk);
  ros::Duration(0.001).sleep();
}

void emergencyStopCallback(std_msgs::Empty msg) {
  if (flag_emergency_stop) return;

  ROS_WARN("emergencyStopCallback");

  flag_emergency_stop = true;

  emergency_stop_cmd.kx = { 5.7, 5.7, 6.2 };
  emergency_stop_cmd.kv = { 3.4, 3.4, 4.0 };

  emergency_stop_cmd.header.frame_id = "world";
  emergency_stop_cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;

  emergency_stop_cmd.position.x = traj_real_.back().x();
  emergency_stop_cmd.position.y = traj_real_.back().y();
  emergency_stop_cmd.position.z = traj_real_.back().z();
  emergency_stop_cmd.velocity.x = emergency_stop_cmd.velocity.y = emergency_stop_cmd.velocity.z = 0.0;
  emergency_stop_cmd.acceleration.x = emergency_stop_cmd.acceleration.y = emergency_stop_cmd.acceleration.z = 0.0;

  double yaw = traj_real_yaw_.back().toRotationMatrix().eulerAngles(0, 1, 2)[2];
  emergency_stop_cmd.yaw = yaw;
  emergency_stop_cmd.yaw_dot = 0.0;

  traj_cmd_.clear();
  traj_real_.clear();
  traj_real_yaw_.clear();
}

void replanCallback(const std_msgs::Float64::ConstPtr& msg) {

  if (msg->data < 0 || msg->data > traj_duration_) {
    ROS_ERROR("[Traj Server]: Unavailable replan_stop_time_");
    // ROS_BREAK();
    ros::Time time_now = ros::Time::now();
    double t_stop = (time_now - start_time_).toSec() + replan_out_ + replan_time_;
    traj_duration_ = std::min(t_stop, traj_duration_);
  }

  else {
    traj_duration_ = msg->data;
  }
}

void newCallback(std_msgs::Empty msg) {
  // Clear the executed traj data
  traj_cmd_.clear();
  traj_real_.clear();
  traj_real_yaw_.clear();
}

void odomCallbck(const nav_msgs::Odometry& msg) {
  if (msg.child_frame_id == "X" || msg.child_frame_id == "O") return;
  odom = msg;
  fovMarkerPtr->publishFOV(msg);
  traj_real_.emplace_back(odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z);
  traj_real_yaw_.emplace_back(
      odom.pose.pose.orientation.w, odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z);

  if (traj_real_.size() > 50000) traj_real_.erase(traj_real_.begin(), traj_real_.begin() + 1000);
  if (traj_real_yaw_.size() > 10000) traj_real_yaw_.erase(traj_real_yaw_.begin(), traj_real_yaw_.begin() + 1000);
}

void visCallback(const ros::TimerEvent& e) {
  if (flag_emergency_stop) return;  // Prevent the original visuzliation msg from being cleared by emergency stop

  displayTrajWithColor(traj_real_, 0.05, Eigen::Vector4d(0, 0, 1, 1));
  // displayTrajWithColor(traj_cmd_, 0.05, Eigen::Vector4d(0, 0, 1, 1));
}

void trajectoryCallback(const traj_utils::MixTrajPtr& msg) {

  if (msg->minco_order != 5) {
    ROS_ERROR("[traj_server] Only support trajectory order equals 5 now!");
    return;
  }

  if (msg->duration_yaw.size() * (msg->minco_order + 1) != msg->coef_yaw.size()) {
    ROS_ERROR("[traj_server] WRONG trajectory parameters, ");
    return;
  }

  // Parse the pos traj msg
  Eigen::MatrixXd pos_pts(msg->pos_pts.size(), 3);
  Eigen::VectorXd knots(msg->knots.size());
  for (size_t i = 0; i < msg->knots.size(); ++i) {
    knots(i) = msg->knots[i];
  }
  for (size_t i = 0; i < msg->pos_pts.size(); ++i) {
    pos_pts(i, 0) = msg->pos_pts[i].x;
    pos_pts(i, 1) = msg->pos_pts[i].y;
    pos_pts(i, 2) = msg->pos_pts[i].z;
  }
  NonUniformBspline pos_traj(pos_pts, msg->bspline_degree, 0.1);
  pos_traj.setKnot(knots);

  start_time_ = msg->start_time;
  traj_id_ = msg->traj_id;

  traj_.clear();
  traj_.push_back(pos_traj);
  traj_.push_back(traj_[0].getDerivative());
  traj_.push_back(traj_[1].getDerivative());
  traj_.push_back(traj_[2].getDerivative());

  traj_duration_ = msg->real_traj_duration;

  // Parse the yaw traj msg
  int piece_num = msg->duration_yaw.size();
  vector<double> duration_yaw(piece_num);
  vector<CoefficientMat> cMats_yaw(piece_num);
  for (int i = 0; i < piece_num; ++i) {
    int i6 = i * 6;
    cMats_yaw[i].row(0) << msg->coef_yaw[i6 + 0], msg->coef_yaw[i6 + 1], msg->coef_yaw[i6 + 2], msg->coef_yaw[i6 + 3],
        msg->coef_yaw[i6 + 4], msg->coef_yaw[i6 + 5];
    cMats_yaw[i].row(1) << 0, 0, 0, 0, 0, 0;
    cMats_yaw[i].row(2) << 0, 0, 0, 0, 0, 0;
    duration_yaw[i] = msg->duration_yaw[i];
  }

  yaw_traj_.reset(new Trajectory<5>(duration_yaw, cMats_yaw));

  receive_traj_ = true;
}

void cmdCallback(const ros::TimerEvent& e) {
  // No publishing before receive traj data
  if (!receive_traj_) return;

  if (flag_emergency_stop) {
    emergency_stop_cmd.header.stamp = ros::Time::now();
    emergency_stop_cmd.trajectory_id = ++traj_id_;
    pos_cmd_pub.publish(emergency_stop_cmd);

    return;
  }

  ros::Time time_now = ros::Time::now();
  double t_cur = (time_now - start_time_).toSec();

  Vector3d pos = Vector3d::Zero();
  Vector3d vel = Vector3d::Zero();
  Vector3d acc = Vector3d::Zero();
  Vector3d jer = Vector3d::Zero();
  double yaw = 0.0;
  double yawdot = 0.0;

  if (t_cur < traj_duration_ && t_cur >= 0.0) {
    // Current time within range of planned traj
    pos = traj_[0].evaluateDeBoorT(t_cur);
    vel = traj_[1].evaluateDeBoorT(t_cur);
    acc = traj_[2].evaluateDeBoorT(t_cur);
    jer = traj_[3].evaluateDeBoorT(t_cur);
    yaw = yaw_traj_->getPos(t_cur)[0];
    yawdot = yaw_traj_->getVel(t_cur)[0];
  }

  else if (t_cur >= traj_duration_) {
    // Current time exceed range of planned traj
    // keep publishing the final position and yaw
    pos = traj_[0].evaluateDeBoorT(traj_duration_);
    vel.setZero();
    acc.setZero();
    yaw = yaw_traj_->getPos(traj_duration_)[0];
    yawdot = 0.0;
  }

  else {
    std::cout << "[Traj server]: invalid time." << std::endl;
  }

  cmd.header.stamp = time_now;
  cmd.trajectory_id = traj_id_;
  cmd.position.x = pos(0);
  cmd.position.y = pos(1);
  cmd.position.z = pos(2);
  cmd.velocity.x = vel(0);
  cmd.velocity.y = vel(1);
  cmd.velocity.z = vel(2);
  cmd.acceleration.x = acc(0);
  cmd.acceleration.y = acc(1);
  cmd.acceleration.z = acc(2);
  cmd.yaw = yaw;
  cmd.yaw_dot = yawdot;
  pos_cmd_pub.publish(cmd);

  // Record info of the executed traj
  if (traj_cmd_.size() == 0) {
    // Add the first position
    traj_cmd_.push_back(pos);
  } else if ((pos - traj_cmd_.back()).norm() > 1e-6) {
    // Add new different commanded position
    traj_cmd_.push_back(pos);
  }

  // if (traj_cmd_.size() > 100000)
  //   traj_cmd_.erase(traj_cmd_.begin(), traj_cmd_.begin() + 1000);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "traj_server");
  ros::NodeHandle node;
  ros::NodeHandle nh("~");

  FOVMarker fovMarker;
  fovMarker.init(nh);
  fovMarkerPtr = &fovMarker;

  ros::Subscriber emergency_stop_sub = node.subscribe("planning/emergency_stop", 10, emergencyStopCallback);
  ros::Subscriber pos_traj_sub = node.subscribe("planning/trajectory", 10, trajectoryCallback);
  ros::Subscriber replan_sub = node.subscribe("planning/replan", 10, replanCallback);
  ros::Subscriber new_sub = node.subscribe("planning/new", 10, newCallback);
  ros::Subscriber odom_sub = node.subscribe("/odom_world", 50, odomCallbck);

  pos_cmd_pub = node.advertise<quadrotor_msgs::PositionCommand>("planning/pos_cmd", 50);
  traj_pub = node.advertise<visualization_msgs::Marker>("planning/travel_traj", 10);

  ros::Timer cmd_timer = node.createTimer(ros::Duration(0.01), cmdCallback);
  ros::Timer vis_timer = node.createTimer(ros::Duration(0.25), visCallback);

  nh.param("traj_server/replan_time", replan_time_, 0.1);
  nh.param("traj_server/replan_out", replan_out_, -1.0);

  Eigen::Vector3d init_pos;
  nh.param("traj_server/init_x", init_pos[0], 0.0);
  nh.param("traj_server/init_y", init_pos[1], 0.0);
  nh.param("traj_server/init_z", init_pos[2], 0.0);

  ROS_WARN("[Traj server]: init...");
  ros::Duration(1.0).sleep();

  // Control parameter
  cmd.kx = { 5.7, 5.7, 6.2 };
  cmd.kv = { 3.4, 3.4, 4.0 };

  // Init cmd msg
  cmd.header.stamp = ros::Time::now();
  cmd.header.frame_id = "world";
  cmd.trajectory_flag = quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_READY;
  cmd.trajectory_id = traj_id_;
  cmd.position.x = init_pos[0];
  cmd.position.y = init_pos[1];
  cmd.position.z = init_pos[2];
  cmd.velocity.x = 0.0;
  cmd.velocity.y = 0.0;
  cmd.velocity.z = 0.0;
  cmd.acceleration.x = 0.0;
  cmd.acceleration.y = 0.0;
  cmd.acceleration.z = 0.0;
  cmd.yaw = 0.0;
  cmd.yaw_dot = 0.0;

  // Move upward and downward
  for (int i = 0; i < 100; ++i) {
    cmd.position.z += 0.01;
    pos_cmd_pub.publish(cmd);
    ros::Duration(0.01).sleep();
  }
  for (int i = 0; i < 100; ++i) {
    cmd.position.z -= 0.01;
    pos_cmd_pub.publish(cmd);
    ros::Duration(0.01).sleep();
  }

  ROS_WARN("[Traj server]: ready.");
  ros::spin();

  return 0;
}
