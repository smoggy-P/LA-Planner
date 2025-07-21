#ifndef PERCEPTION_AWARE_GRAPH_SEARCH_H_
#define PERCEPTION_AWARE_GRAPH_SEARCH_H_

#include <vector>
#include <unordered_map>
#include <queue>
#include <list>
#include <memory>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <Eigen/Eigen>

using Eigen::Vector3d;
using std::cout;
using std::list;
using std::priority_queue;
using std::queue;
using std::shared_ptr;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

namespace perception_aware_planner {
// GraphSearch that operates on different types of node using Dijkstra algorithm
template <typename NodeT>
class GraphSearch {
public:
  void print();
  void addNode(const shared_ptr<NodeT>& node);
  void removeNode(const int& remove_id);
  void addEdge(const int& from, const int& to);
  void DijkstraSearch(const int& start, const int& goal, vector<shared_ptr<NodeT>>& path);

  int getNodeCount() const {
    return node_num_;
  }

  int getEdgesCount() const {
    return edge_num_;
  }

  shared_ptr<NodeT> getNode(const int id) const;

private:
  vector<shared_ptr<NodeT>> nodes_;
  int node_num_ = 0;
  int edge_num_ = 0;
};

template <typename NodeT>
class NodeCompare {
public:
  bool operator()(const shared_ptr<NodeT>& node1, const shared_ptr<NodeT>& node2) {
    return node1->g_value_ > node2->g_value_;
  }
};

template <typename NodeT>
void GraphSearch<NodeT>::print() {
  for (auto v : nodes_) {
    v->print();
    v->printNeighbors();
  }
}

template <typename NodeT>
void GraphSearch<NodeT>::addNode(const shared_ptr<NodeT>& node) {
  nodes_.push_back(node);
  nodes_.back()->id_ = node_num_++;
}

template <typename NodeT>
void GraphSearch<NodeT>::removeNode(const int& remove_id) {
  // Step1: Check is it valid
  if (remove_id < 0 || remove_id >= node_num_) {
    std::cerr << "[GraphSearch::removeNode] Invalid remove_id: " << remove_id << std::endl;
    return;
  }

  // Step2: Remove all edges that point to remove_id
  int removed_edge_count = 0;
  for (auto& node : nodes_) {
    if (!node) continue;
    auto& neighs = node->neighbors_;

    auto before_erase_size = neighs.size();
    neighs.erase(std::remove_if(neighs.begin(), neighs.end(),
                     [remove_id](const std::shared_ptr<NodeT>& neigh) { return neigh && neigh->id_ == remove_id; }),
        neighs.end());
    auto after_erase_size = neighs.size();
    removed_edge_count += (before_erase_size - after_erase_size);
  }

  edge_num_ -= removed_edge_count;

  if (remove_id == node_num_ - 1) {
    nodes_.pop_back();
    node_num_--;
    return;
  }

  // Step3: Maintain continous storage of the array
  auto node_to_remove = nodes_[remove_id];
  auto node_last = nodes_.back();
  nodes_[remove_id] = node_last;
  node_last->id_ = remove_id;

  // Step4: Update all the neighbors that point to the last node
  int replaced_id = node_num_ - 1;
  for (auto& node : nodes_) {
    if (!node) continue;
    for (auto& neigh : node->neighbors_) {
      if (neigh && neigh->id_ == replaced_id) {
        neigh = node_last;
      }
    }
  }

  // Step5: Remove the last node
  nodes_.pop_back();
  node_num_--;
}

template <typename NodeT>
void GraphSearch<NodeT>::addEdge(const int& from, const int& to) {
  nodes_[from]->neighbors_.push_back(nodes_[to]);
  ++edge_num_;
}

template <typename NodeT>
shared_ptr<NodeT> GraphSearch<NodeT>::getNode(const int id) const {
  return nodes_[id];
}

template <typename NodeT>
void GraphSearch<NodeT>::DijkstraSearch(const int& start, const int& goal, vector<shared_ptr<NodeT>>& path) {

  priority_queue<shared_ptr<NodeT>, vector<shared_ptr<NodeT>>, NodeCompare<NodeT>> open_set;

  shared_ptr<NodeT> start_v = nodes_[start];
  shared_ptr<NodeT> end_v = nodes_[goal];
  start_v->g_value_ = 0.0;
  open_set.push(start_v);

  while (!open_set.empty()) {
    auto vc = open_set.top();
    open_set.pop();
    vc->closed_ = true;

    // Check if reach target
    if (vc == end_v) {
      shared_ptr<NodeT> vit = vc;
      while (vit != nullptr) {
        path.push_back(vit);
        vit = vit->parent_;
      }
      reverse(path.begin(), path.end());
      return;
    }

    for (auto vb : vc->neighbors_) {
      // Check if in close set
      if (vb->closed_) continue;

      // Add new node or update node in open set
      double g_tmp = vc->g_value_ + vc->costTo(vb);
      if (g_tmp < vb->g_value_) {
        vb->g_value_ = g_tmp;
        vb->parent_ = vc;
        open_set.push(vb);
      }
    }
  }
}
}  // namespace perception_aware_planner

#endif