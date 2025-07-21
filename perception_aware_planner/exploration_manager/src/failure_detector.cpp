#include "exploration_manager/failure_detector.h"

#include "traj_utils/non_uniform_bspline.h"

#include "utils/utils.h"

using namespace Eigen;

namespace perception_aware_planner {

FailureDetector::FailureDetector(ros::NodeHandle& nh) {

  param_.max_tol_time_ = Utils::getGlobalParam().max_tol_time_;

  nh.param("failure_detect/min_check_freq", param_.min_check_freq_, 100.0);
  nh.param("failure_detect/output_path", output_path_, string(""));

  if (!output_path_.empty()) {
    f_realtime_odom_.open(output_path_ + "realtime_odom.txt");
    f_realtime_vis_.open(output_path_ + "realtime_vis.txt");
    f_traj_odom_.open(output_path_ + "traj_odom.txt");
  }
}

FailureDetector::~FailureDetector() {
  if (f_realtime_odom_.is_open()) f_realtime_odom_.close();
  if (f_realtime_vis_.is_open()) f_realtime_vis_.close();
  if (f_traj_odom_.is_open()) f_traj_odom_.close();
}

// Record the odometry data during running
void FailureDetector::saveTraj_TUM(std::ofstream& fout, const double timestamp, const Vector3d& p, const Quaterniond& q) {
  if (!fout.is_open()) return;

  // TUM data format
  std::string timestamp_str = std::to_string(timestamp);
  std::vector<double> data = { p(0), p(1), p(2), q.x(), q.y(), q.z(), q.w() };

  fout << std::fixed;
  fout << std::setprecision(9) << timestamp_str;
  for (const auto& d : data) fout << " " << d;

  fout.unsetf(std::ios_base::fixed);

  fout << std::endl;
}

// Record the vis && covis feature nums during running
void FailureDetector::saveTraj_Vis(std::ofstream& fout, const double timestamp, const int vis_num, const int covis_num) {
  if (!fout.is_open()) return;

  // TUM data format
  std::string timestamp_str = std::to_string(timestamp);
  std::vector<int> data = { vis_num, covis_num };

  fout << std::fixed;
  fout << std::setprecision(9) << timestamp_str;
  fout.unsetf(std::ios_base::fixed);
  for (const auto& d : data) fout << " " << d;

  fout << std::endl;
}

bool FailureDetector::checkSingleFrameVisibility(const Vector3d& pos, const Quaterniond& orient) {
  int feature_num = feature_map_->getFeatureUsingOdom(pos, orient);
  int min_feature_num = Utils::getGlobalParam().min_feature_num_act_;
  return feature_num > min_feature_num;
}

bool FailureDetector::checkRealTime(const double timestamp, const Vector3d& pos, const Quaterniond& orient) {

  double t_now = timestamp;
  saveTraj_TUM(f_realtime_odom_, t_now, pos, orient);

  static double t_last_localizable = t_now;

  if (!slide_window_act_.empty()) {
    double delta_t = t_now - slide_window_act_.back().first;

    // Our detection method requires a very high check frequency, otherwise it will return an incorrect result
    if (delta_t > 1.0 / param_.min_check_freq_) {
      // If you see this error, you need to increase the buffer size of "odometryCallback" in FSM
      ROS_ERROR("Too low check freq: %fHz", 1.0 / delta_t);
      ROS_BREAK();
      return true;
    }
  }

  set<int> feature_with_id;
  int feature_num = feature_map_->getFeatureUsingOdom(pos, orient, feature_with_id);

  int size = slide_window_act_.size();
  for (int i = 0; i < size; i++) {
    double t = slide_window_act_.front().first;
    if (t_now - t > param_.max_tol_time_) {
      slide_window_act_.pop_front();
    }

    else
      break;
  }

  set<int> feature_in_sw;
  for (const auto& e : slide_window_act_) {
    feature_in_sw.insert(e.second.begin(), e.second.end());
  }

  int covis_feature_num = Utils::getSameCount(feature_in_sw, feature_with_id);

  auto min_feature_num = Utils::getGlobalParam().min_feature_num_act_;
  auto min_covis_feature_num = Utils::getGlobalParam().min_covisible_feature_num_act_;
  saveTraj_Vis(f_realtime_vis_, t_now, feature_num, covis_feature_num);

  slide_window_act_.push_back(make_pair(t_now, feature_with_id));
  if (slide_window_act_.size() == 1) {
    ROS_WARN("[Failure Detector]: Reset the slide window");
    return true;
  }

  if (feature_num <= min_feature_num || covis_feature_num <= min_covis_feature_num) {
    // ROS_ERROR("[Failure Detector]: CheckLocalizability Fail!!!Feature num: %d/%d, Covis Feature num: %d/%d; Timestamp: %fs",
    //     feature_num, min_feature_num, covis_feature_num, min_covis_feature_num, t_now);

    if (t_now - t_last_localizable > param_.max_tol_time_) {
      return false;
    }
  }

  else {
    t_last_localizable = t_now;
  }

  return true;
}

bool FailureDetector::checkTraj(const LocalTrajData& local_traj) {
  double t_last_localizable = 0.0, t_now = 0.0;

  Eigen::VectorXd times = local_traj.yaw_traj_.getDurations();

  deque<pair<double, set<int>>> slide_window;

  for (int i = 0; i < times.size(); i++) {
    Vector3d fut_pt = local_traj.position_traj_.evaluateDeBoorT(t_now);
    double fut_yaw = local_traj.yaw_traj_.getPos(t_now)[0];
    Vector3d fut_acc = local_traj.acceleration_traj_.evaluateDeBoorT(t_now);

    Quaterniond fut_orient = Utils::calcOrientation(fut_yaw, fut_acc);

    set<int> feature_with_id;
    int feature_num = feature_map_->getFeatureUsingOdom(fut_pt, fut_orient, feature_with_id);

    int size = slide_window.size();
    for (int i = 0; i < size; i++) {
      double t = slide_window.front().first;
      if (t_now - t > param_.max_tol_time_) {
        slide_window.pop_front();
      }

      else
        break;
    }

    set<int> feature_in_sw;
    for (const auto& e : slide_window) {
      feature_in_sw.insert(e.second.begin(), e.second.end());
    }

    slide_window.push_back(make_pair(t_now, feature_with_id));
    int covis_feature_num = Utils::getSameCount(feature_in_sw, feature_with_id);

    auto min_feature_num = Utils::getGlobalParam().min_feature_num_plan_;
    auto min_covis_feature_num = Utils::getGlobalParam().min_covisible_feature_num_plan_;

    if (feature_num <= min_feature_num || covis_feature_num <= min_covis_feature_num) {
      if (t_now - t_last_localizable > param_.max_tol_time_) {
        return false;
      }
    }

    else {
      t_last_localizable = t_now;
    }

    t_now += times(i);
  }

  // Add an extra check for the end state (hovering horizontally)
  const auto& end_pos = local_traj.end_pos_;
  const auto& end_yaw = local_traj.end_yaw_;
  Quaterniond end_ori = Utils::calcOrientation(end_yaw, Vector3d::Zero());

  int feature_num = feature_map_->getFeatureUsingOdom(end_pos, end_ori);
  auto min_feature_num = Utils::getGlobalParam().min_feature_num_plan_;

  if (feature_num <= min_feature_num) {
    ROS_ERROR("[Failure Detector::checkTraj]: Fail in checking the end state!!!");
    return false;
  }

  return true;
}

}  // namespace perception_aware_planner