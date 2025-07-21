#include "active_perception/localization_aware_graph_search.h"

#include "utils/utils.h"

namespace perception_aware_planner {

void LocalizationNode::print() const {
  std::cout << "Node " << id_ << ": State (" << state_.transpose() << "), g_value: " << g_value_ << std::endl;
}

void LocalizationNode::printNeighbors() const {
  std::cout << "Neighbors of Node " << id_ << ": ";
  for (const auto& neighbor : neighbors_) {
    std::cout << neighbor->id_ << " ";
  }
  std::cout << std::endl;
}

double LocalizationNode::costTo(const shared_ptr<LocalizationNode>& neighbor) const {
  return computePathCost(state_, neighbor->state_, param_->max_vel_, param_->max_yaw_rate_);
}

double LocalizationNode::computePathCost(
    const Vector4d& state_1, const Vector4d& state_2, const double& max_vel, const double& max_yaw_rate) {
  double distance = (state_1.head<3>() - state_2.head<3>()).norm();
  double yaw_diff = fabs(state_1(3) - state_2(3));
  yaw_diff = std::min(yaw_diff, 2 * M_PI - yaw_diff);

  double min_path_time = distance / max_vel;
  double min_yaw_time = yaw_diff / max_yaw_rate;

  return std::max(min_path_time, min_yaw_time);
}

double LocalizationNode::computeFeatureCost(
    const bitset<MAX_FEATURES>& features1, const bitset<MAX_FEATURES>& features2, const double& min_covisible_feature_num) {
  int different_bits = static_cast<int>((features1 ^ features2).count());
  double score = 0;
  if (different_bits < min_covisible_feature_num)
    score = static_cast<double>(std::pow(min_covisible_feature_num - different_bits, 2));
  return score;
}

bool LocalizationNode::checkLocalbility(const shared_ptr<LocalizationNode>& neighbor, const double& min_covisible_feature_num) {
  return static_cast<int>((visible_features_ & neighbor->visible_features_).count()) > min_covisible_feature_num;
}

void LocalizationAwareGraphSearch::init(ros::NodeHandle& nh) {
  param_.max_vel_ = Utils::getGlobalParam().max_vel_;
  param_.max_yaw_rate_ = Utils::getGlobalParam().max_yaw_rate_;
  param_.min_feature_num_ = Utils::getGlobalParam().min_feature_num_plan_;
  param_.min_covisible_feature_num_ = Utils::getGlobalParam().min_covisible_feature_num_plan_;
}

void LocalizationAwareGraphSearch::addNode(
    const Vector3d& pos, const double& yaw, const vector<int>& feature_ids, const NodeType& type) {
  auto node = make_shared<LocalizationNode>();

  node->state_.head<3>() = pos;
  node->state_(3) = yaw;

  for (const auto& feature_id : feature_ids) {
    if (feature_id >= 0 && feature_id < MAX_FEATURES) {
      node->visible_features_.set(feature_id);  // Set the corresponding bit
    }

    else
      std::cerr << "Feature index " << feature_id << " is out of range (0-" << (MAX_FEATURES - 1) << ").\n";
  }

  node->type_ = type;
  node->param_ = &param_;

  graph_.addNode(node);
}

void LocalizationAwareGraphSearch::addEdge(int from, int to) {
  graph_.addEdge(from, to);
}

void LocalizationAwareGraphSearch::addFeatureEdges() {
  int node_count = getNodeCount();

  // Traverse all nodes and connect them according to the rules
  for (int i = 0; i < node_count; ++i) {
    for (int j = 0; j < node_count; ++j) {

      if (i == j) continue;
      // Perhaps we can reduce computational complexity here...
      const auto& node_from = graph_.getNode(i);
      const auto& node_to = graph_.getNode(j);

      // Check if the two nodes can be connected
      if (!canConnect(node_from->type_, node_to->type_)) continue;
      if (!node_from->checkLocalbility(node_to, param_.min_covisible_feature_num_)) continue;

      // Add the edge
      addEdge(i, j);
    }
  }
}

bool LocalizationAwareGraphSearch::SearchViewpoint(const Vector3d& pos, const double& yaw, const vector<int>& feature_ids,
    const NodeType& type,                                                                // input
    vector<Eigen::Vector3d>& node_path, vector<double>& node_yaw, vector<double>& cost)  // outout
{
  int node_count = getNodeCount();
  if (node_count == 0) return false;
  addNode(pos, yaw, feature_ids, type);

  for (int i = 0; i < node_count; ++i) {
    const auto& node_from = graph_.getNode(i);
    const auto& node_to = graph_.getNode(node_count);

    if (!node_from->checkLocalbility(node_to, param_.min_covisible_feature_num_)) continue;

    if (canConnect(node_from->type_, node_to->type_)) addEdge(i, node_count);
    if (canConnect(node_to->type_, node_from->type_)) addEdge(node_count, i);
  }
  outputPath(search(0, node_count), node_path, node_yaw, cost);
  graph_.removeNode(node_count);

  return !node_path.empty();
}

vector<shared_ptr<LocalizationNode>> LocalizationAwareGraphSearch::search(int start_id, int goal_id) {
  vector<shared_ptr<LocalizationNode>> path;
  graph_.DijkstraSearch(start_id, goal_id, path);
  return path;
}

void LocalizationAwareGraphSearch::outputPath(const vector<shared_ptr<LocalizationNode>>& path,
    vector<Eigen::Vector3d>& node_path, vector<double>& node_yaw, vector<double>& cost) {
  node_path.clear();
  node_yaw.clear();
  cost.clear();

  for (const auto& node : path) {
    node_path.push_back(node->state_.head<3>());
    node_yaw.push_back(node->state_(3));
    cost.push_back(node->g_value_);
  }
}

void LocalizationAwareGraphSearch::clear() {
  graph_ = GraphSearch<LocalizationNode>();
}

}  // namespace perception_aware_planner
