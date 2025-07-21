#ifndef _PLANNING_VISUALIZATION_H_
#define _PLANNING_VISUALIZATION_H_

#include "traj_utils/non_uniform_bspline.h"

#include <ros/ros.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/kdtree/kdtree_flann.h>

namespace perception_aware_planner {

enum VIEWPOINT_CHANGE_REASON {
  NONE,
  PATH_SEARCH_FAIL,
  POSITION_OPT_FAIL,
  YAW_INIT_FAIL,
  YAW_OPT_FAIL,
  LOCABILITY_CHECK_FAIL,
  EXPLORABILITY_CHECK_FAIL,
  COLLISION_CHECK_FAIL
};

class PlanningVisualization {
private:
  enum TRAJECTORY_PLANNING_ID {
    GOAL = 1,
    PATH = 200,
    BSPLINE = 300,
    BSPLINE_CTRL_PT = 400,
    POLY_TRAJ = 500,
    ASTAR_PATH = 800,
    UNREACHABLE_VIEWPOINT = 900,
    GO_VIEWPOINT = 1000,
    UNREACHABLE_KINOASTAR = 1100,
    DEAD_FRONTOER = 1300,
    ACTIVE_FRONTIER = 1400,
    UNREACHABLE_POSTTAJ = 1600,
    SHOW_FEATURE_TEXT = 1700,
    YAW_FOV = 1900,
    YAW_CORRIDOR = 2000,
    YAW_ARROW = 2100,
    FRONTIER_ASRAT_PATH = 2400,
    BEST_FRONTIER_VPS = 2500,
    FEATURE_CLUSTER = 2600,
    VIEWPOINT = 30000
  };

  ros::NodeHandle node;
  ros::Publisher traj_pub_;          // 0, bspline(pos) trajectory
  ros::Publisher yaw_fov_pub_;       // 1, yaw fov trajectory
  ros::Publisher topo_pub_;          // 2, deprecated
  ros::Publisher predict_pub_;       // 3, deprecated
  ros::Publisher cluster_pub_;       // 4, feature clusters
  ros::Publisher frontier_pub_;      // 5, frontier
  ros::Publisher viewpoint_pub_;     // 6, viewpoint
  ros::Publisher text_pub_;          // 7, text
  ros::Publisher yaw_corridor_pub_;  // 8, localizable corridor
  ros::Publisher yaw_arrow_pub_;     // 9, yaw arrow trajectory
  vector<ros::Publisher> pubs_;

public:
  PlanningVisualization(ros::NodeHandle& nh);

  void drawGoal(const Eigen::Vector3d& goal, const double resolution, const Eigen::Vector4d& color, int id = 0);

  void drawAtarPath(const vector<Eigen::Vector3d>& paths, const Eigen::Vector4d& color, const int& id);

  void drawFrontiers(const vector<vector<Eigen::Vector3d>>& frontiers, const Eigen::Vector4d& color, const bool is_dead);
  void drawFrontiersScore(const vector<vector<Eigen::Vector3d>>& clusters, const vector<vector<double>>& score,
      const vector<Eigen::Vector3d>& pos, const Eigen::Vector4d& color, const bool is_dead);

  void setFanMarker(visualization_msgs::Marker& marker, const double radius, const double scale, const Eigen::Vector3d& pos,
      const Eigen::Vector3d& acc, const double lb, const double ub, const string& ns, const Eigen::Vector4d& color,
      const int& id);
  void setFOVmarker(visualization_msgs::Marker& marker, const double& fov_depth, const Eigen::Vector3d& pos,
      const Eigen::Matrix3d& R_cam, const string& ns, const Eigen::Vector4d& color, const int& id);

  void drawViewpoints(
      const vector<std::pair<Eigen::Vector3d, double>>& viewpoints, const Eigen::Vector4d& color, const int type = 0);
  void drawTargetViewpoint(const Eigen::Vector3d& pos, const double yaw, const Eigen::Vector4d& color);

  int unreachable_num_ = 0;
  VIEWPOINT_CHANGE_REASON fail_reason = NONE;

  void drawFrontiersUnreachable(const vector<Eigen::Vector3d>& cells, const Eigen::Vector3d& pos, const double& yaw,
      const vector<Eigen::Vector3d>& fail_pos_path, NonUniformBspline& fail_pos_traj, const double duration);

  void clearUnreachableMarker();
  void drawFeatureNum(const int feature_num, const int feature_num_thr, const Eigen::Vector3d& odom_pos);

  // new interface
  void fillBasicInfo(visualization_msgs::Marker& mk, const Eigen::Vector3d& scale, const Eigen::Vector4d& color, const string& ns,
      const int& id, const int& shape);
  void fillGeometryInfo(visualization_msgs::Marker& mk, const vector<Eigen::Vector3d>& list);
  void fillGeometryInfo(
      visualization_msgs::Marker& mk, const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2);

  void displayText(
      const Eigen::Vector3d& position, const std::string& text, const Eigen::Vector4d& color, double scale, int id, int pub_id);
  void drawSpheres(const vector<Eigen::Vector3d>& list, const double& scale, const Eigen::Vector4d& color, const string& ns,
      const int& id, const int& pub_id);
  void drawCubes(const vector<Eigen::Vector3d>& list, const double& scale, const Eigen::Vector4d& color, const string& ns,
      const int& id, const int& pub_id);
  void drawLines(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2, const double& scale,
      const Eigen::Vector4d& color, const string& ns, const int& id, const int& pub_id);
  void drawLines(const vector<Eigen::Vector3d>& list, const double& scale, const Eigen::Vector4d& color, const string& ns,
      const int& id, const int& pub_id);
  void drawBox(const Eigen::Vector3d& center, const Eigen::Vector3d& scale, const Eigen::Vector4d& color, const string& ns,
      const int& id, const int& pub_id);
  void drawClusters(const vector<Eigen::Vector3d>& centers, const vector<Eigen::Vector3d>& scales, const Eigen::Vector4d& color);

  // draw basic shapes
  void displaySphereList(
      const vector<Eigen::Vector3d>& list, double resolution, const Eigen::Vector4d& color, int id, int pub_id = 0);
  void displayCubeList(
      const vector<Eigen::Vector3d>& list, double resolution, const Eigen::Vector4d& color, int id, int pub_id = 0);
  void displayLineList(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2, double line_width,
      const Eigen::Vector4d& color, int id, int pub_id = 0);
  void displayArrowList(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2, double line_width,
      const Eigen::Vector4d& color, int id, int pub_id);

  // draw a piece-wise straight line path
  void drawGeometricPath(const vector<Eigen::Vector3d>& path, double resolution, const Eigen::Vector4d& color, int id = 0);
  // draw a bspline trajectory
  void drawBspline(NonUniformBspline& bspline, const double duration, double size, const Eigen::Vector4d& color,
      bool show_ctrl_pts = false, double size2 = 0.1, const Eigen::Vector4d& color2 = Eigen::Vector4d(1, 1, 0, 1), int id = 0);
  // draw a yaw trajectory
  void drawYawFOVTraj(const vector<Eigen::Vector3d>& pos, const vector<Eigen::Vector3d>& acc, const vector<Eigen::Vector3d>& yaw,
      const Eigen::Vector4d& color);
  // draw a yaw trajectory using arrows
  void drawYawArrow(
      const vector<Eigen::Vector3d>& pos, const vector<Eigen::Vector3d>& acc, const vector<Eigen::Vector3d>& yaw, int id = 0);
  // draw the yaw corridor
  void drawYawCorridor(const vector<Eigen::Vector3d>& pos, const vector<Eigen::Vector3d>& acc, const vector<Eigen::Vector3d>& yaw,
      const vector<double>& bound, const Eigen::Vector4d& color);

  Eigen::Vector4d getColor(const double& h, double alpha = 1.0);

  using Ptr = std::shared_ptr<PlanningVisualization>;

  struct Color {
    double r_;
    double g_;
    double b_;
    double a_;

    Color() : r_(0), g_(0), b_(0), a_(1) {
    }
    Color(double r, double g, double b) : Color(r, g, b, 1.) {
    }
    Color(double r, double g, double b, double a) : r_(r), g_(g), b_(b), a_(a) {
    }
    Color(int r, int g, int b) {
      r_ = static_cast<double>(r) / 255.;
      g_ = static_cast<double>(g) / 255.;
      b_ = static_cast<double>(b) / 255.;
      a_ = 1.;
    }
    Color(int r, int g, int b, int a) {
      r_ = static_cast<double>(r) / 255.;
      g_ = static_cast<double>(g) / 255.;
      b_ = static_cast<double>(b) / 255.;
      a_ = static_cast<double>(a) / 255.;
    }

    static Eigen::Vector4d toEigen(const Color& color) {
      return Eigen::Vector4d(color.r_, color.g_, color.b_, color.a_);
    }

    static const Eigen::Vector4d White() {
      return toEigen(Color(255, 255, 255));
    }
    static const Eigen::Vector4d Black() {
      return toEigen(Color(0, 0, 0));
    }
    static const Eigen::Vector4d Gray() {
      return toEigen(Color(127, 127, 127));
    }
    static const Eigen::Vector4d Red() {
      return toEigen(Color(255, 0, 0));
    }
    static const Eigen::Vector4d DeepRed() {
      return toEigen(Color(127, 0, 0));
    }
    static const Eigen::Vector4d Green() {
      return toEigen(Color(0, 255, 0));
    }
    static const Eigen::Vector4d DeepGreen() {
      return toEigen(Color(0, 127, 0));
    }
    static const Eigen::Vector4d SpringGreen() {
      return toEigen(Color(0, 255, 127));
    }
    static const Eigen::Vector4d Blue() {
      return toEigen(Color(0, 0, 255));
    }
    static const Eigen::Vector4d DeepBlue() {
      return toEigen(Color(0, 0, 127));
    }
    static const Eigen::Vector4d Yellow() {
      return toEigen(Color(255, 255, 0));
    }
    static const Eigen::Vector4d Orange() {
      return toEigen(Color(255, 127, 0));
    }
    static const Eigen::Vector4d Purple() {
      return toEigen(Color(127, 0, 255));
    }
    static const Eigen::Vector4d Teal() {
      return toEigen(Color(0, 255, 255));
    }
    static const Eigen::Vector4d TealTransparent() {
      return toEigen(Color(0, 255, 255, 200));
    }
    static const Eigen::Vector4d Pink() {
      return toEigen(Color(255, 0, 127));
    }
    static const Eigen::Vector4d Magenta() {
      return toEigen(Color(255, 0, 255));
    }
  };
};
}  // namespace perception_aware_planner
#endif