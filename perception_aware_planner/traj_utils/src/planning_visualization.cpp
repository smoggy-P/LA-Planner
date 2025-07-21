#include "traj_utils/planning_visualization.h"

#include "utils/utils.h"

using namespace std;
using namespace Eigen;

namespace perception_aware_planner {

PlanningVisualization::PlanningVisualization(ros::NodeHandle& nh) {
  node = nh;

  traj_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/trajectory", 100);
  pubs_.push_back(traj_pub_);

  yaw_fov_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/yaw", 100);
  pubs_.push_back(yaw_fov_pub_);

  topo_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/topo_path", 100);
  pubs_.push_back(topo_pub_);

  predict_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/prediction", 100);
  pubs_.push_back(predict_pub_);

  cluster_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/feature_cluster", 100);
  pubs_.push_back(cluster_pub_);

  frontier_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/frontier", 10000);
  pubs_.push_back(frontier_pub_);

  viewpoint_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/viewpoints", 1000);
  pubs_.push_back(viewpoint_pub_);

  text_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/text", 100);
  pubs_.push_back(text_pub_);

  yaw_corridor_pub_ = node.advertise<visualization_msgs::Marker>("/planning_vis/corridor", 100);
  pubs_.push_back(yaw_corridor_pub_);

  yaw_arrow_pub_ = node.advertise<visualization_msgs::MarkerArray>("/planning_vis/yaw_arrow", 100);
  pubs_.push_back(yaw_arrow_pub_);
}

void PlanningVisualization::fillBasicInfo(visualization_msgs::Marker& mk, const Vector3d& scale, const Vector4d& color,
    const string& ns, const int& id, const int& shape) {

  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.id = id;
  mk.ns = ns;
  mk.type = shape;

  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);

  mk.scale.x = scale[0];
  mk.scale.y = scale[1];
  mk.scale.z = scale[2];
}

void PlanningVisualization::fillGeometryInfo(visualization_msgs::Marker& mk, const vector<Eigen::Vector3d>& list) {
  geometry_msgs::Point pt;
  for (int i = 0; i < int(list.size()); i++) {
    pt.x = list[i](0);
    pt.y = list[i](1);
    pt.z = list[i](2);
    mk.points.push_back(pt);
  }
}

void PlanningVisualization::fillGeometryInfo(
    visualization_msgs::Marker& mk, const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2) {
  geometry_msgs::Point pt;
  for (int i = 0; i < int(list1.size()); ++i) {
    pt.x = list1[i](0);
    pt.y = list1[i](1);
    pt.z = list1[i](2);
    mk.points.push_back(pt);

    pt.x = list2[i](0);
    pt.y = list2[i](1);
    pt.z = list2[i](2);
    mk.points.push_back(pt);
  }
}

void PlanningVisualization::drawAtarPath(const vector<Eigen::Vector3d>& paths, const Eigen::Vector4d& color, const int& id) {
  drawSpheres(paths, 0.1, color, "path", FRONTIER_ASRAT_PATH + id % 100, 0);
}

void PlanningVisualization::drawFrontiers(
    const vector<vector<Eigen::Vector3d>>& frontiers, const Eigen::Vector4d& color, const bool is_dead) {
  static size_t last_size[2] = { 0, 0 };
  int base_id = (is_dead) ? DEAD_FRONTOER : ACTIVE_FRONTIER;
  for (size_t i = frontiers.size(); i < last_size[is_dead ? 1 : 0]; ++i)
    drawCubes({}, 0.1, color, "frontier", base_id + i % 100, 5);
  for (size_t i = 0; i < frontiers.size(); ++i) drawCubes(frontiers[i], 0.1, color, "frontier", base_id + i % 100, 5);
  last_size[is_dead ? 1 : 0] = frontiers.size();
}

void PlanningVisualization::drawFrontiersScore(const vector<vector<Vector3d>>& clusters, const vector<vector<double>>& score,
    const vector<Vector3d>& pos, const Vector4d& color, const bool is_dead) {

  ROS_ASSERT(clusters.size() == score.size() && clusters.size() == pos.size());

  static size_t last_size[2] = { 0, 0 };
  Eigen::Vector4d color_black(0.0, 0.0, 0.0, 0.8);
  int base_id = (is_dead) ? DEAD_FRONTOER : ACTIVE_FRONTIER;
  for (size_t i = clusters.size(); i < last_size[is_dead ? 1 : 0]; ++i) {
    drawCubes({}, 0.1, color, "frontier", base_id + i % 100, 5);
    displayText(Vector3d::Zero(), "", color_black, 0.0, base_id + i % 100, 7);
  }

  for (size_t i = 0; i < clusters.size(); ++i) {
    Eigen::Vector4d color_i = color;
    if (!is_dead) {
      double intensity = static_cast<double>(i) / clusters.size();
      color_i << intensity, 0, 1.0 - intensity, color(3);
    }

    drawCubes(clusters[i], 0.1, color_i, "frontier", base_id + i % 100, 5);

    // draw score
    ROS_ASSERT(score[i].size() == 4);
    std::ostringstream vp_score, com_score, final_score, total_score;
    vp_score << std::fixed << std::setprecision(2) << score[i][0];
    com_score << std::fixed << std::setprecision(2) << score[i][1];
    final_score << std::fixed << std::setprecision(2) << score[i][2];
    total_score << std::fixed << std::setprecision(2) << score[i][3];
    std::string text = " Sort " + std::to_string(i) + " Score: " + total_score.str();
    displayText(pos[i] + Vector3d(0, 0, 0.1), text, color_i, 0.2, base_id + i % 100, 7);
  }

  last_size[is_dead ? 1 : 0] = clusters.size();
}

void PlanningVisualization::drawViewpoints(
    const vector<pair<Vector3d, double>>& viewpoints, const Vector4d& color, const int type) {
  if (type >= 100) return;

  static size_t last_type_size[100] = { 0 };

  for (size_t id = 0; id < viewpoints.size(); ++id) {
    visualization_msgs::Marker marker;
    Eigen::AngleAxisd yaw_angle(viewpoints[id].second, Vector3d(0, 0, 1));
    Eigen::Vector4d color_id = color;
    if (type == 0) {
      double intensity = static_cast<double>(id) / viewpoints.size();
      color_id = Eigen::Vector4d(intensity, 0, 1.0 - intensity, color(3));
    }
    setFOVmarker(marker, 0.3, viewpoints[id].first, yaw_angle.toRotationMatrix(), "viewpoint_FOV_marker", color_id,
        id % 1000 + VIEWPOINT + type * 1000);
    // marker.action = visualization_msgs::Marker::DELETE;
    // pubs_[6].publish(marker);
    marker.action = visualization_msgs::Marker::ADD;
    pubs_[6].publish(marker);
  }

  for (size_t id = viewpoints.size(); id < last_type_size[type]; ++id) {
    visualization_msgs::Marker marker;
    marker.action = visualization_msgs::Marker::DELETE;
    marker.id = id % 1000 + VIEWPOINT + type * 1000;
    marker.ns = "viewpoint_FOV_marker";
    pubs_[6].publish(marker);
  }

  last_type_size[type] = viewpoints.size();
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawTargetViewpoint(const Vector3d& pos, const double yaw, const Vector4d& color) {
  visualization_msgs::Marker marker;
  auto R_cam = Eigen::AngleAxisd(yaw, Vector3d(0, 0, 1)).toRotationMatrix();
  setFOVmarker(marker, 0.5, pos, R_cam, "viewpoint_FOV_marker", color, GO_VIEWPOINT);
  pubs_[6].publish(marker);
}

void PlanningVisualization::setFanMarker(visualization_msgs::Marker& marker, const double radius, const double scale,
    const Vector3d& pos, const Vector3d& acc, const double lb, const double ub, const string& ns, const Vector4d& color,
    const int& id) {

  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = scale * 0.1;
  marker.color.r = color(0);
  marker.color.g = color(1);
  marker.color.b = color(2);
  marker.color.a = color(3);
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;

  int step = 20;
  vector<double> angles;
  double res = (ub - lb) / step;
  for (double yaw = lb; yaw < ub + 1e-3; yaw += res) {
    angles.push_back(yaw);
  }

  marker.pose.position.x = 0.0;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 0.0;

  geometry_msgs::Point p;
  p.x = pos.x();
  p.y = pos.y();
  p.z = pos.z();
  marker.points.push_back(p);

  for (const auto& yaw : angles) {

    Quaterniond ori = Utils::calcOrientation(yaw, acc);
    Vector3d dir = ori * Vector3d(radius, 0, 0);

    Vector3d pdir = pos + dir;
    geometry_msgs::Point p;
    p.x = pdir.x();
    p.y = pdir.y();
    p.z = pdir.z();
    marker.points.push_back(p);
  }

  p.x = pos.x();
  p.y = pos.y();
  p.z = pos.z();
  marker.points.push_back(p);
}

void PlanningVisualization::setFOVmarker(visualization_msgs::Marker& marker, const double& fov_depth, const Vector3d& pos,
    const Matrix3d& R_cam, const string& ns, const Vector4d& color, const int& id) {
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = ns;
  marker.id = id;
  marker.type = visualization_msgs::Marker::LINE_LIST;
  marker.action = visualization_msgs::Marker::ADD;
  marker.scale.x = fov_depth * 0.1;
  marker.color.r = color(0);
  marker.color.g = color(1);
  marker.color.b = color(2);
  marker.color.a = color(3);
  marker.pose.orientation.w = 1.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;

  double fov_horizontal = 79.1396 * M_PI / 180.0;
  double fov_vertical = 63.5803 * M_PI / 180.0;
  double half_fov_width = tan(fov_horizontal / 2.0) * fov_depth;
  double half_fov_height = tan(fov_vertical / 2.0) * fov_depth;

  std::vector<Eigen::Vector3d> vertices_camera_frame = {
    { fov_depth, -half_fov_width, -half_fov_height },
    { fov_depth, half_fov_width, -half_fov_height },
    { fov_depth, half_fov_width, half_fov_height },
    { fov_depth, -half_fov_width, half_fov_height },
  };
  std::vector<Eigen::Vector3d> world_vertices;
  for (const auto& vertex : vertices_camera_frame) {
    Eigen::Vector3d world_vertex = R_cam * vertex + pos;
    world_vertices.push_back(world_vertex);
  }
  geometry_msgs::Point zero;
  zero.x = pos.x();
  zero.y = pos.y();
  zero.z = pos.z();
  for (size_t i = 0; i < world_vertices.size(); ++i) {
    geometry_msgs::Point pt1;
    pt1.x = world_vertices[i].x();
    pt1.y = world_vertices[i].y();
    pt1.z = world_vertices[i].z();
    marker.points.push_back(pt1);

    geometry_msgs::Point pt2;
    pt2.x = world_vertices[(i + 1) % world_vertices.size()].x();
    pt2.y = world_vertices[(i + 1) % world_vertices.size()].y();
    pt2.z = world_vertices[(i + 1) % world_vertices.size()].z();
    marker.points.push_back(pt2);
  }

  for (size_t i = 0; i < world_vertices.size(); ++i) {
    marker.points.push_back(zero);
    geometry_msgs::Point pt;
    pt.x = world_vertices[i].x();
    pt.y = world_vertices[i].y();
    pt.z = world_vertices[i].z();
    marker.points.push_back(pt);
  }
}

void PlanningVisualization::drawFrontiersUnreachable(const vector<Vector3d>& cells, const Vector3d& pos, const double& yaw,
    const vector<Vector3d>& fail_pos_path, NonUniformBspline& fail_pos_traj, const double duration) {

  Vector3d top_middle = Utils::calculateTopMiddlePoint(cells);
  top_middle(2) += 0.1 * unreachable_num_;

  double intensity = 1.0 - 0.05 * unreachable_num_;
  if (intensity < 0.1) intensity = 0.1;
  // Vector4d color(intensity, 0.0, 0.0, 0.4);
  Vector4d color_purple(intensity, 0, intensity, 0.5);

  double path_size = 0.04;
  double traj_size = 0.02;

  // draw this unreachable viewpoint
  Eigen::AngleAxisd yaw_angle(yaw, Eigen::Vector3d(0, 0, 1));
  visualization_msgs::Marker marker;
  setFOVmarker(marker, 0.25, pos, yaw_angle.toRotationMatrix(), "unreachable_viewpoint", color_purple,
      UNREACHABLE_VIEWPOINT + unreachable_num_ % 100);
  pubs_[5].publish(marker);

  // draw text and path/traj
  std::string error_reason;

  switch (fail_reason) {
    case PATH_SEARCH_FAIL:
      error_reason = "PATH_SEARCH_FAIL";
      break;
    case POSITION_OPT_FAIL:
      error_reason = "POSITION_OPT_FAIL";
      drawGeometricPath(fail_pos_path, path_size, color_purple, UNREACHABLE_KINOASTAR + unreachable_num_ % 100);
      break;
    case YAW_INIT_FAIL:
      error_reason = "YAW_INIT_FAIL";
      drawBspline(fail_pos_traj, duration, traj_size, color_purple, true, path_size, Vector4d(1, 1, 0, 1),
          UNREACHABLE_POSTTAJ + unreachable_num_ % 100);
      break;
    case YAW_OPT_FAIL:
      error_reason = "YAW_OPT_FAIL";
      drawBspline(fail_pos_traj, duration, traj_size, color_purple, true, path_size, Vector4d(1, 1, 0, 1),
          UNREACHABLE_POSTTAJ + unreachable_num_ % 100);
      break;
    case LOCABILITY_CHECK_FAIL:
      error_reason = "LOCABILITY_CHECK_FAIL";
      drawBspline(fail_pos_traj, duration, traj_size, color_purple, true, path_size, Vector4d(1, 1, 0, 1),
          UNREACHABLE_POSTTAJ + unreachable_num_ % 100);
      break;
    case EXPLORABILITY_CHECK_FAIL:
      error_reason = "EXPLORABILITY_CHECK_FAIL";
      drawBspline(fail_pos_traj, duration, traj_size, color_purple, true, path_size, Vector4d(1, 1, 0, 1),
          UNREACHABLE_POSTTAJ + unreachable_num_ % 100);
      break;
    case COLLISION_CHECK_FAIL:
      error_reason = "COLLISION_CHECK_FAIL";
      drawBspline(fail_pos_traj, duration, traj_size, color_purple, true, path_size, Vector4d(1, 1, 0, 1),
          UNREACHABLE_POSTTAJ + unreachable_num_ % 100);
      break;
    default:
      ROS_BREAK();
      break;
  }
  std::string text = "id: " + std::to_string(unreachable_num_) + "  " + error_reason;
  color_purple(3) = 1.0;
  displayText(top_middle, text, color_purple, 0.2, UNREACHABLE_VIEWPOINT + unreachable_num_ % 100, 7);
  unreachable_num_++;
}

void PlanningVisualization::clearUnreachableMarker() {
  vector<Eigen::Vector3d> empty_vector_Vector3d;
  Vector4d black_color(0, 0, 0, 1);
  Vector3d zero_pos(0, 0, 0);
  NonUniformBspline empty;
  std::string text = "";
  for (int i = 0; i < unreachable_num_; ++i) {
    visualization_msgs::Marker marker;
    marker.action = visualization_msgs::Marker::DELETE;
    marker.id = UNREACHABLE_VIEWPOINT + i % 100;
    marker.ns = "unreachable_viewpoint";
    pubs_[5].publish(marker);
    displayText(zero_pos, text, black_color, 0.3, UNREACHABLE_VIEWPOINT + i % 100, 7);
    drawBspline(empty, 0.0, 0.1, black_color, true, 0.15, Vector4d(1, 1, 0, 1), UNREACHABLE_POSTTAJ + i % 100);
    drawGeometricPath(empty_vector_Vector3d, 0.02, black_color, UNREACHABLE_KINOASTAR + i % 100);
  }
  unreachable_num_ = 0;
}

void PlanningVisualization::drawFeatureNum(const int feature_num, const int feature_num_thr, const Vector3d& odom_pos) {
  std::string text = "Feature_view: " + std::to_string(feature_num);
  Eigen::Vector4d color;
  if (feature_num < feature_num_thr) {
    // Display red color when less than threshold
    color = Eigen::Vector4d(1.0, 0.0, 0.0, 1.0);
  }

  else {
    // Gradually transition from red to green when greater than threshold
    double ratio = std::min(1.0, double(feature_num - feature_num_thr) / (2.0 * feature_num_thr));
    color = Eigen::Vector4d(1.0 - ratio, ratio, 0.0, 1.0);
  }

  displayText(odom_pos + Vector3d(0.0, 0.0, 1.0), text, color, 0.3, SHOW_FEATURE_TEXT, 7);
}

void PlanningVisualization::drawClusters(const vector<Vector3d>& centers, const vector<Vector3d>& scales, const Vector4d& color) {

  static size_t last_node_size = 0;

  for (size_t i = 0; i < centers.size(); ++i) {
    drawBox(centers[i], scales[i], color, "feature_clusters", FEATURE_CLUSTER + i % 100, 4);
    Eigen::Vector3d show_pos = centers[i] + Vector3d(0, 0, scales[i][2] / 2.0);
    std::string text = "Feature Cluster ID: " + std::to_string(i);
    displayText(show_pos, text, color, 0.2, FEATURE_CLUSTER + i % 100, 7);
  }

  for (size_t i = centers.size(); i < last_node_size; ++i) {  // clear out of date
    drawBox(Vector3d::Zero(), Vector3d::Zero(), color, "feature_clusters", FEATURE_CLUSTER + i % 100, 4);
    displayText(Vector3d::Zero(), "", color, 0.2, FEATURE_CLUSTER + i % 100, 7);
  }
  last_node_size = centers.size();
}

void PlanningVisualization::drawBox(
    const Vector3d& center, const Vector3d& scale, const Vector4d& color, const string& ns, const int& id, const int& pub_id) {

  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(0.05, 0.05, 0.05), color, ns, id, visualization_msgs::Marker::LINE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  // vertex of box
  Eigen::Vector3d half_scale = scale * 0.5;
  std::vector<Eigen::Vector3d> vertices = { center + Eigen::Vector3d(-half_scale[0], -half_scale[1], -half_scale[2]),
    center + Eigen::Vector3d(half_scale[0], -half_scale[1], -half_scale[2]),
    center + Eigen::Vector3d(half_scale[0], half_scale[1], -half_scale[2]),
    center + Eigen::Vector3d(-half_scale[0], half_scale[1], -half_scale[2]),
    center + Eigen::Vector3d(-half_scale[0], -half_scale[1], half_scale[2]),
    center + Eigen::Vector3d(half_scale[0], -half_scale[1], half_scale[2]),
    center + Eigen::Vector3d(half_scale[0], half_scale[1], half_scale[2]),
    center + Eigen::Vector3d(-half_scale[0], half_scale[1], half_scale[2]) };

  // edge of box
  std::vector<std::pair<int, int>> edges = {
    { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 },  // bottom
    { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 },  // top
    { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }   // vertical
  };

  // fill geometry info
  for (const auto& edge : edges) {
    geometry_msgs::Point p1, p2;
    p1.x = vertices[edge.first][0];
    p1.y = vertices[edge.first][1];
    p1.z = vertices[edge.first][2];
    p2.x = vertices[edge.second][0];
    p2.y = vertices[edge.second][1];
    p2.z = vertices[edge.second][2];
    mk.points.push_back(p1);
    mk.points.push_back(p2);
  }

  mk.action = visualization_msgs::Marker::ADD;

  // publish new marker
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawSpheres(const vector<Eigen::Vector3d>& list, const double& scale, const Eigen::Vector4d& color,
    const string& ns, const int& id, const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id, visualization_msgs::Marker::SPHERE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  // pub new marker
  fillGeometryInfo(mk, list);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawCubes(const vector<Eigen::Vector3d>& list, const double& scale, const Eigen::Vector4d& color,
    const string& ns, const int& id, const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id, visualization_msgs::Marker::CUBE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  // pub new marker
  if (!list.empty()) {
    fillGeometryInfo(mk, list);
    mk.action = visualization_msgs::Marker::ADD;
    pubs_[pub_id].publish(mk);
  }
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawLines(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2,
    const double& scale, const Eigen::Vector4d& color, const string& ns, const int& id, const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id, visualization_msgs::Marker::LINE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  if (list1.size() == 0) return;

  // pub new marker
  fillGeometryInfo(mk, list1, list2);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::drawLines(const vector<Eigen::Vector3d>& list, const double& scale, const Eigen::Vector4d& color,
    const string& ns, const int& id, const int& pub_id) {
  visualization_msgs::Marker mk;
  fillBasicInfo(mk, Eigen::Vector3d(scale, scale, scale), color, ns, id, visualization_msgs::Marker::LINE_LIST);

  // clean old marker
  mk.action = visualization_msgs::Marker::DELETE;
  pubs_[pub_id].publish(mk);

  if (list.empty()) return;

  // split the single list into two
  vector<Eigen::Vector3d> list1, list2;
  for (size_t i = 0; i < list.size() - 1; ++i) {
    list1.push_back(list[i]);
    list2.push_back(list[i + 1]);
  }

  // pub new marker
  fillGeometryInfo(mk, list1, list2);
  mk.action = visualization_msgs::Marker::ADD;
  pubs_[pub_id].publish(mk);
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::displaySphereList(
    const vector<Vector3d>& list, double resolution, const Vector4d& color, int id, int pub_id) {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::SPHERE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  if (!list.empty()) {
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
    for (size_t i = 0; i < list.size(); i++) {
      pt.x = list[i](0);
      pt.y = list[i](1);
      pt.z = list[i](2);
      mk.points.push_back(pt);
    }
    pubs_[pub_id].publish(mk);
  }
  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::displayCubeList(
    const vector<Eigen::Vector3d>& list, double resolution, const Eigen::Vector4d& color, int id, int pub_id) {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::CUBE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

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
  for (int i = 0; i < int(list.size()); i++) {
    pt.x = list[i](0);
    pt.y = list[i](1);
    pt.z = list[i](2);
    mk.points.push_back(pt);
  }
  pubs_[pub_id].publish(mk);

  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::displayLineList(const vector<Eigen::Vector3d>& list1, const vector<Eigen::Vector3d>& list2,
    double line_width, const Eigen::Vector4d& color, int id, int pub_id) {
  visualization_msgs::Marker mk;
  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::LINE_LIST;
  mk.action = visualization_msgs::Marker::DELETE;
  mk.id = id;
  pubs_[pub_id].publish(mk);

  mk.action = visualization_msgs::Marker::ADD;
  mk.pose.orientation.x = 0.0;
  mk.pose.orientation.y = 0.0;
  mk.pose.orientation.z = 0.0;
  mk.pose.orientation.w = 1.0;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);
  mk.scale.x = line_width;

  geometry_msgs::Point pt;
  for (int i = 0; i < int(list1.size()); ++i) {
    pt.x = list1[i](0);
    pt.y = list1[i](1);
    pt.z = list1[i](2);
    mk.points.push_back(pt);

    pt.x = list2[i](0);
    pt.y = list2[i](1);
    pt.z = list2[i](2);
    mk.points.push_back(pt);
  }
  pubs_[pub_id].publish(mk);

  ros::Duration(0.0005).sleep();
}

void PlanningVisualization::displayArrowList(
    const vector<Vector3d>& list1, const vector<Vector3d>& list2, double line_width, const Vector4d& color, int id, int pub_id) {

  if (pubs_[pub_id].getNumSubscribers() == 0) return;
  ROS_ASSERT(list1.size() == list2.size());

  static size_t max_id = 0;

  visualization_msgs::MarkerArray markerArray;
  if (list1.empty() || list2.empty()) {
    for (int i = id; i < 100 + id; ++i) {
      visualization_msgs::Marker mk;
      mk.header.frame_id = "world";
      mk.header.stamp = ros::Time::now();
      mk.id = i;
      mk.type = visualization_msgs::Marker::ARROW;
      mk.action = visualization_msgs::Marker::DELETE;
      markerArray.markers.push_back(mk);
    }
    pubs_[pub_id].publish(markerArray);
    return;
  }
  max_id = max(max_id, list1.size());
  for (size_t i = id; i < list1.size() + id; ++i) {
    // max_id = max(max_id, i);

    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = i;
    mk.type = visualization_msgs::Marker::ARROW;
    mk.action = visualization_msgs::Marker::ADD;
    mk.pose.orientation.w = 1.0;
    mk.scale.x = line_width;      // Arrow shaft diameter
    mk.scale.y = line_width * 3;  // Arrow head diameter
    mk.scale.z = 0.0;             // Arrow head length (0.0 for auto-compute)
    mk.color.r = color(0);
    mk.color.g = color(1);
    mk.color.b = color(2);
    mk.color.a = color(3);

    geometry_msgs::Point pt;
    int idx = i - id;
    pt.x = list1[idx](0);
    pt.y = list1[idx](1);
    pt.z = list1[idx](2);
    mk.points.push_back(pt);
    pt.x = list2[idx](0);
    pt.y = list2[idx](1);
    pt.z = list2[idx](2);
    mk.points.push_back(pt);

    markerArray.markers.push_back(mk);
  }
  for (size_t i = id + list1.size(); i < id + list1.size() + 100; ++i) {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.id = i;
    mk.type = visualization_msgs::Marker::ARROW;
    mk.action = visualization_msgs::Marker::DELETE;
    markerArray.markers.push_back(mk);
  }

  pubs_[pub_id].publish(markerArray);
}

void PlanningVisualization::drawGeometricPath(const vector<Vector3d>& path, double resolution, const Vector4d& color, int id) {
  displaySphereList(path, resolution, color, PATH + id % 100);
}

void PlanningVisualization::drawBspline(NonUniformBspline& bspline, const double duration, double size, const Vector4d& color,
    bool show_ctrl_pts, double size2, const Vector4d& color2, int id1) {

  if (bspline.getControlPoint().size() == 0) {
    visualization_msgs::Marker mk;
    fillBasicInfo(mk, Vector3d::Ones(), Vector4d::Ones(), "bspline", id1, visualization_msgs::Marker::SPHERE_LIST);
    mk.action = visualization_msgs::Marker::DELETE;
    pubs_[0].publish(mk);
    fillBasicInfo(mk, Vector3d::Ones(), Vector4d::Ones(), "bspline", id1 + 50, visualization_msgs::Marker::SPHERE_LIST);
    pubs_[0].publish(mk);
    return;
  }

  vector<Vector3d> traj_pts;
  for (double t = 0.0; t <= duration; t += 0.01) traj_pts.push_back(bspline.evaluateDeBoorT(t));

  drawSpheres(traj_pts, size, color, "bspline", id1, 0);

  // draw the control point
  if (show_ctrl_pts) {
    Eigen::MatrixXd ctrl_pts = bspline.getControlPoint();
    vector<Eigen::Vector3d> ctp;
    for (int i = 0; i < int(ctrl_pts.rows()); ++i) {
      Eigen::Vector3d pt = ctrl_pts.row(i).transpose();
      ctp.push_back(pt);
    }
    // displaySphereList(ctp, size2, color2, BSPLINE_CTRL_PT + id2 % 100);
    drawSpheres(ctp, size2, color2, "bspline", id1 + 50, 0);
  }
}

void PlanningVisualization::drawYawFOVTraj(
    const vector<Vector3d>& pos, const vector<Vector3d>& acc, const vector<Vector3d>& yaw, const Vector4d& color) {

  for (size_t i = 0; i < pos.size(); i++) {
    visualization_msgs::Marker marker;
    const auto& pos_waypt = pos[i];
    const auto& acc_waypt = acc[i];
    const auto& yaw_waypt = yaw[i][0];
    Quaterniond ori = Utils::calcOrientation(yaw_waypt, acc_waypt);

    setFOVmarker(marker, 0.3, pos_waypt, ori.toRotationMatrix(), "yaw_fov", color, YAW_FOV + i % 100);
    // marker.action = visualization_msgs::Marker::DELETE;
    // pubs_[1].publish(marker);
    // ros::Duration(0.0001).sleep();
    marker.action = visualization_msgs::Marker::ADD;
    pubs_[1].publish(marker);
    ros::Duration(0.0001).sleep();
  }

  for (size_t i = pos.size(); i < 100; i++) {
    visualization_msgs::Marker marker;
    marker.action = visualization_msgs::Marker::DELETE;
    marker.id = YAW_FOV + i % 100;
    marker.ns = "yaw_fov";
    pubs_[1].publish(marker);
    ros::Duration(0.0001).sleep();
  }
}

void PlanningVisualization::drawYawArrow(
    const vector<Vector3d>& pos, const vector<Vector3d>& acc, const vector<Vector3d>& yaw, int id) {
  ROS_ASSERT(pos.size() == yaw.size());

  vector<Vector3d> pts1;
  vector<Vector3d> pts2;
  for (size_t i = 0; i < pos.size(); i++) {
    const auto& pos_waypt = pos[i];
    pts1.push_back(pos_waypt);
    const auto& acc_waypt = acc[i];
    const auto& yaw_waypt = yaw[i][0];

    Quaterniond ori = Utils::calcOrientation(yaw_waypt, acc_waypt);
    Vector3d dir = ori * Vector3d(1.0, 0, 0);

    Vector3d pdir = pos_waypt + dir;
    pts2.push_back(pdir);
  }

  displayArrowList(pts1, pts2, 0.05, Vector4d(0.0, 1.0, 0.0, 1.0), YAW_ARROW + id % 100, 9);
}

void PlanningVisualization::drawYawCorridor(const vector<Vector3d>& pos, const vector<Vector3d>& acc, const vector<Vector3d>& yaw,
    const vector<double>& bound, const Vector4d& color) {

  for (size_t i = 0; i < pos.size(); i++) {
    visualization_msgs::Marker marker;
    const auto& pos_waypt = pos[i];
    const auto& acc_waypt = acc[i];
    const auto& lb_waypt = bound[2 * i];
    const auto& ub_waypt = bound[2 * i + 1];

    setFanMarker(marker, 0.35, 0.25, pos_waypt, acc_waypt, lb_waypt, ub_waypt, "yaw_corridor", Vector4d(1.0, 0.0, 0.0, 1.0),
        YAW_CORRIDOR + i % 100);
    marker.action = visualization_msgs::Marker::DELETE;
    pubs_[8].publish(marker);
    ros::Duration(0.0001).sleep();
    marker.action = visualization_msgs::Marker::ADD;
    pubs_[8].publish(marker);
    ros::Duration(0.0001).sleep();
  }

  for (size_t i = pos.size(); i < 100; i++) {
    visualization_msgs::Marker marker;
    marker.action = visualization_msgs::Marker::DELETE;
    marker.id = YAW_CORRIDOR + i % 100;
    marker.ns = "yaw_corridor";
    pubs_[8].publish(marker);
    ros::Duration(0.0001).sleep();
  }
}

void PlanningVisualization::drawGoal(const Vector3d& goal, const double resolution, const Vector4d& color, int id) {
  vector<Vector3d> goal_vec = { goal };
  displaySphereList(goal_vec, resolution, color, GOAL + id % 100);
}

void PlanningVisualization::displayText(
    const Vector3d& position, const string& text, const Vector4d& color, double scale, int id, int pub_id) {
  visualization_msgs::Marker mk;

  mk.header.frame_id = "world";
  mk.header.stamp = ros::Time::now();
  mk.type = visualization_msgs::Marker::TEXT_VIEW_FACING;  // Text marker type
  mk.action = visualization_msgs::Marker::ADD;             // Add marker
  mk.ns = "text_marker";                                   // Namespace to distinguish different types of markers
  mk.id = id;                                              // Marker ID for unique identification

  mk.pose.position.x = position(0);
  mk.pose.position.y = position(1);
  mk.pose.position.z = position(2) + 0.1;

  mk.text = text;

  mk.color.r = color(0);
  mk.color.g = color(1);
  mk.color.b = color(2);
  mk.color.a = color(3);

  mk.scale.z = scale;

  if (text.empty()) {
    mk.action = visualization_msgs::Marker::DELETE;
    pubs_[pub_id].publish(mk);
  }

  else {
    mk.action = visualization_msgs::Marker::ADD;
    pubs_[pub_id].publish(mk);
  }

  ros::Duration(0.0005).sleep();
}

Eigen::Vector4d PlanningVisualization::getColor(const double& h, double alpha) {
  double h1 = h;
  if (h1 < 0.0 || h1 > 1.0) {
    std::cout << "h out of range" << std::endl;
    h1 = 0.0;
  }

  double lambda = 0.0;
  Eigen::Vector4d color1, color2;
  if (h1 >= -1e-4 && h1 < 1.0 / 6) {
    lambda = (h1 - 0.0) * 6;
    color1 = Eigen::Vector4d(1, 0, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 1, 1);
  } else if (h1 >= 1.0 / 6 && h1 < 2.0 / 6) {
    lambda = (h1 - 1.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 0, 1, 1);
  } else if (h1 >= 2.0 / 6 && h1 < 3.0 / 6) {
    lambda = (h1 - 2.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 0, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 1, 1);
  } else if (h1 >= 3.0 / 6 && h1 < 4.0 / 6) {
    lambda = (h1 - 3.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 1, 1);
    color2 = Eigen::Vector4d(0, 1, 0, 1);
  } else if (h1 >= 4.0 / 6 && h1 < 5.0 / 6) {
    lambda = (h1 - 4.0 / 6) * 6;
    color1 = Eigen::Vector4d(0, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 1, 0, 1);
  } else if (h1 >= 5.0 / 6 && h1 <= 1.0 + 1e-4) {
    lambda = (h1 - 5.0 / 6) * 6;
    color1 = Eigen::Vector4d(1, 1, 0, 1);
    color2 = Eigen::Vector4d(1, 0, 0, 1);
  }

  Eigen::Vector4d fcolor = (1 - lambda) * color1 + lambda * color2;
  fcolor(3) = alpha;

  return fcolor;
}
}  // namespace perception_aware_planner