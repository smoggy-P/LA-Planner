#ifndef LOCALIZATION_AWARE_GRAPH_SEARCH_H
#define LOCALIZATION_AWARE_GRAPH_SEARCH_H

#include "active_perception/graph_search.h"

#include <Eigen/Core>
#include <vector>
#include <memory>
#include <iostream>
#include <bitset>
#include <ros/ros.h>

using Eigen::Vector3d;
using Eigen::Vector4d;
using std::bitset;
using std::cout;
using std::endl;
using std::make_shared;
using std::pair;
using std::shared_ptr;
using std::vector;

namespace perception_aware_planner {

#define HUGE_COST_ 999999
const int MAX_FEATURES = 2048;  // Assuming the number of features does not exceed this value

struct Dijkstra_Param {
  double max_vel_;
  double max_yaw_rate_;
  int min_feature_num_;
  int min_covisible_feature_num_;
};

// Use a bit mask for node type
enum class NodeType : uint8_t {
  START = 0b00001,
  FINAL = 0b00010,
  FEATURE_SAMPLE = 0b00100,
  FRONTIER_SAMPLE = 0b01000,
  FINAL_SAMPLE = 0b10000
};

// Rules for connection
constexpr uint8_t connection_rules[] = {
  0b11110,  // START           ---> FEATURE_SAMPLE, FRONTIER_SAMPLE, FINAL_SAMPLE, FINAL
  0b00000,  // FINAL           ---> NONE
  0b11100,  // FEATURE_SAMPLE  ---> FEATURE_SAMPLE, FRONTIER_SAMPLE, FINAL_SAMPLE
  0b11110,  // FRONTIER_SAMPLE ---> FEATURE_SAMPLE, FRONTIER_SAMPLE, FINAL_SAMPLE, FINAL
  0b00010   // FINAL_SAMPLE    ---> FINAL
};

// Check whether the type of connection match the rule
inline bool canConnect(const NodeType from, const NodeType to) {
  uint8_t from_mask = static_cast<uint8_t>(from);
  uint8_t to_mask = static_cast<uint8_t>(to);
  return (connection_rules[static_cast<size_t>(std::log2(from_mask))] & to_mask) != 0;
}

inline bool isSameType(const NodeType from, const NodeType to) {
  return (static_cast<uint8_t>(from) == static_cast<uint8_t>(to));
}

// vertex type for graph search
class LocalizationNode {
public:
  int id_ = -1;                                     // Graph ID
  NodeType type_;                                   // Type
  Vector4d state_;                                  // (x, y, z, yaw)
  bitset<MAX_FEATURES> visible_features_;           // Set of visible features
  double g_value_ = INFINITY;                       // Cost
  bool closed_ = false;                             // Is it visited
  shared_ptr<LocalizationNode> parent_ = nullptr;   // Parent node
  vector<shared_ptr<LocalizationNode>> neighbors_;  // Neighbor nodes
  const Dijkstra_Param* param_;                     // Paramerter for Dijkstra

  void print() const;
  void printNeighbors() const;

  double costTo(const shared_ptr<LocalizationNode>& neighbor) const;
  bool checkLocalbility(const shared_ptr<LocalizationNode>& neighbor, const double& min_covisible_feature_num);

private:
  static double computeFeatureCost(
      const bitset<MAX_FEATURES>& features1, const bitset<MAX_FEATURES>& features2, const double& min_covisible_feature_num);
  static double computePathCost(
      const Vector4d& state_1, const Vector4d& state_2, const double& max_vel, const double& max_yaw_rate);
};

// Class for graph search
class LocalizationAwareGraphSearch {
private:
  GraphSearch<LocalizationNode> graph_;
  Dijkstra_Param param_;

public:
  void init(ros::NodeHandle& nh);
  void clear();

  void addNode(const Vector3d& pos, const double& yaw, const vector<int>& feature_ids, const NodeType& type);
  void addEdge(int from, int to);
  void addFeatureEdges();

  // Algorithm entrance
  vector<shared_ptr<LocalizationNode>> search(int start_id, int goal_id);
  void outputPath(const vector<shared_ptr<LocalizationNode>>& path, vector<Eigen::Vector3d>& node_path, vector<double>& node_yaw,
      vector<double>& cost);
  bool SearchViewpoint(const Vector3d& pos, const double& yaw, const vector<int>& feature_ids,
      const NodeType& type,                                                                 // input
      vector<Eigen::Vector3d>& node_path, vector<double>& node_yaw, vector<double>& cost);  // outout
  int getNodeCount() const {
    return graph_.getNodeCount();
  }

  int getEdgesCount() const {
    return graph_.getEdgesCount();
  }
};

}  // namespace perception_aware_planner

#endif  // LOCALIZATION_AWARE_GRAPH_SEARCH_H
