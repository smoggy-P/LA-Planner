#include "local_plan_manager/yaw_initial_planner.h"

#include <execution>

using namespace std;

namespace perception_aware_planner {

YawInitialPlanner::YawInitialPlanner(ros::NodeHandle& nh) {

  param_.max_yaw_rate_ = Utils::getGlobalParam().max_yaw_rate_;
  param_.min_feature_num_ = Utils::getGlobalParam().min_feature_num_plan_;
  param_.min_covisible_feature_num_ = Utils::getGlobalParam().min_covisible_feature_num_plan_;

  nh.param("yaw_initial/piece_num", param_.piece_num_, -1);
  nh.param("yaw_initial/ld_smoothness", param_.ld_smoothness_, 0.0);
  nh.param("yaw_initial/ld_expl", param_.ld_expl_, 0.0);
  nh.param("yaw_initial/ld_frontier", param_.ld_frontier_, 0.0);
  nh.param("yaw_initial/ld_final_goal", param_.ld_final_goal_, 0.0);

  param_.basic_cost_ = pow(2 * M_PI / param_.piece_num_, 2);

  param_.yaw_samples_.resize(param_.piece_num_);
  for (int i = 0; i < param_.piece_num_; ++i) {
    param_.yaw_samples_[i] = i * 2 * M_PI / param_.piece_num_ - M_PI;
  }

  frontier_cam_ = Utils::getGlobalParam().frontier_cam_;
  feature_cam_ = Utils::getGlobalParam().feature_cam_;
}

// Preprocess the frontier to speed up building graph
void YawInitialPlanner::preprocessFrontier() {
  size_t num_layers = pos_.size();

  target_frontier_aft_preprocess_.clear();
  target_frontier_aft_preprocess_.resize(num_layers);

  for (size_t i = 0; i < num_layers; i++) {
    Vector3d pos = pos_[i];
    for (size_t j = 0; j < target_frontier_.size(); j++) {
      Vector3d cell = target_frontier_[j];
      if (sdf_map_->getVisibility(pos, cell)) target_frontier_aft_preprocess_[i].emplace_back(j);
    }
  }
}

void YawInitialPlanner::yaw2id(const YawVertex::Ptr& v) {
  ROS_ASSERT(v->yaw_ >= -M_PI && v->yaw_ <= M_PI);
  v->yaw_id_ = static_cast<int>((v->yaw_ + M_PI) / (2 * M_PI / param_.piece_num_));
}

void YawInitialPlanner::id2yaw(const YawVertex::Ptr& v) {
  ROS_ASSERT(v->yaw_id_ >= 0 && v->yaw_id_ < param_.piece_num_);
  v->yaw_ = v->yaw_id_ * 2 * M_PI / param_.piece_num_ - M_PI;
}

void YawInitialPlanner::addVertex(const YawVertex::Ptr& vertex) {
  vertice_.push_back(vertex);
  vertex->graph_id_ = graph_id_++;
}

double YawInitialPlanner::calEdgeCost(const YawVertex::Ptr& from, const YawVertex::Ptr& to) {

  double diff = from->calEdgeDiff(to);
  double diff_square = diff * diff;
  diff_square += param_.basic_cost_;

  int new_frontier_num = to->frontiers_id_.size() - Utils::getSameCount(from->frontiers_id_, to->frontiers_id_);

  double expl_gain = 0.0;
  if (!go2final_)
    expl_gain = param_.ld_frontier_ * new_frontier_num + (to->if_vis_final_goal_ ? param_.ld_final_goal_ : 0);
  else
    expl_gain = param_.ld_frontier_ * new_frontier_num;

  double edge_cost = diff_square * (1 + std::exp(-expl_gain));

  return edge_cost;
}

void YawInitialPlanner::checkIfVisGoal(const YawVertex::Ptr& v) {
  v->if_vis_final_goal_ = sdf_map_->getVisibility(pos_[v->layer_], v->yaw_, final_goal_);
}

bool YawInitialPlanner::checkFeasibility(const YawVertex::Ptr& v1, const YawVertex::Ptr& v2) {
  int diff_id = abs(v1->yaw_id_ - v2->yaw_id_);
  int diff_id_round = std::min(diff_id, param_.piece_num_ - diff_id);

  return (diff_id_round <= param_.max_diff_yaw_id_);
}

bool YawInitialPlanner::graphSearch() {
  start_vert_->g_value_ = 0.0;

  set<int> open_set_id;
  set<int> close_set;

  std::vector<YawVertex::Ptr> open_set;
  open_set.push_back(start_vert_);
  open_set_id.emplace(start_vert_->graph_id_);

  while (!open_set.empty()) {
    auto it = std::min_element(open_set.begin(), open_set.end(),
        [](const YawVertex::Ptr& a, const YawVertex::Ptr& b) { return a->g_value_ < b->g_value_; });

    auto vc = *it;
    open_set.erase(it);
    open_set_id.erase(vc->graph_id_);
    close_set.emplace(vc->graph_id_);

    // reach target
    if (vc == end_vert_) {
      YawVertex::Ptr vit = vc;
      while (vit != nullptr) {
        vert_path_.push_back(vit);
        vit = vit->parent_;
      }
      reverse(vert_path_.begin(), vert_path_.end());
      return true;
    }

    for (auto& vb : vc->edges_) {
      // skip vertex in close set
      if (close_set.find(vb->graph_id_) != close_set.end()) continue;

      // update new or open vertex
      double g_tmp = vc->g_value_ + calEdgeCost(vc, vb);

      if (open_set_id.find(vb->graph_id_) == open_set_id.end()) {
        open_set_id.emplace(vb->graph_id_);
        open_set.push_back(vb);
      }

      else if (g_tmp > vb->g_value_)
        continue;

      vb->parent_ = vc;
      vb->g_value_ = g_tmp;
      vb->frontiers_id_path_ = vc->frontiers_id_path_;
      vb->frontiers_id_path_.insert(vb->frontiers_id_.begin(), vb->frontiers_id_.end());
    }
  }

  return false;
}

void YawInitialPlanner::setVisbleFrontiers(const YawVertex::Ptr& v) {
  v->frontiers_id_.clear();

  Quaterniond ori = Utils::calcOrientation(v->yaw_, acc_[v->layer_]);
  sdf_map_->countVisibleCells(
      pos_[v->layer_], ori, target_frontier_, target_frontier_aft_preprocess_[v->layer_], v->frontiers_id_);
}

void YawInitialPlanner::setVisbleFeatures(const YawVertex::Ptr& v) {
  v->features_id_.clear();

  vector<pair<int, Vector3d>> features;
  Quaterniond ori = Utils::calcOrientation(v->yaw_, acc_[v->layer_]);
  feature_map_->getFeatureUsingOdom(pos_[v->layer_], ori, features);
  for (const auto& feature : features) {
    v->features_id_.push_back(feature.first);
  }
}

int YawInitialPlanner::getCoVisibleNum(const YawVertex::Ptr& v1, const YawVertex::Ptr& v2) {

  int commonFeatureCount = 0;
  for (const auto& id1 : v1->features_id_) {
    for (const auto& id2 : v2->features_id_) {
      if (id1 == id2) {
        commonFeatureCount++;
      }
    }
  }

  return commonFeatureCount;
}

void YawInitialPlanner::getCoVisibleSet(const YawVertex::Ptr& v1, const YawVertex::Ptr& v2, set<int>& co_vis_id) {

  co_vis_id.clear();

  for (const auto& id1 : v1->features_id_) {
    for (const auto& id2 : v2->features_id_) {
      if (id1 == id2) {
        co_vis_id.insert(id1);
      }
    }
  }
}

double YawInitialPlanner::calcVisibility(const YawVertex::Ptr& v, const int id) {

  const Vector3d feature = feature_map_->getFeatureByID(id);

  const double& yaw = v->yaw_;
  const Vector3d& pos = pos_[v->layer_];
  const Vector3d& acc = acc_[v->layer_];

  // Calculate vectors n1, ny, n3 ,b and their gradients
  Vector3d gravity(0, 0, -9.81);

  Vector3d n1, ny, n3, n2, b;
  n1 = acc - gravity;  // thrust
  ny << cos(yaw), sin(yaw), 0;
  n3 = n1.cross(ny);
  n2 = n3.cross(n1);
  b = feature - pos;

  // v2
  double k = 20;
  double cos_theta2 = n2.dot(b) / (n2.norm() * b.norm());
  double v2 = Utils::sigmoid(k, cos_theta2);

  // v3
  double fov_horizontal = feature_cam_->fov_horizontal * M_PI / 180;
  double alpha3 = (M_PI - fov_horizontal) / 2.0;
  double sin_theta3 = n3.cross(b).norm() / (n3.norm() * b.norm());
  double v3 = Utils::sigmoid(k, (sin_theta3 - sin(alpha3)));

  double visib = v2 * v3;

  return visib;
}

double YawInitialPlanner::calcCoVisibility(const YawVertex::Ptr& v1, const YawVertex::Ptr& v2, const int id) {
  double visib1 = calcVisibility(v1, id);
  double visib2 = calcVisibility(v2, id);
  double covisb = visib1 * visib2;

  return covisb;
}

void YawInitialPlanner::selectTargetCoVisibleID(
    const YawVertex::Ptr& v1, const YawVertex::Ptr& v2, const set<int>& s_in, set<int>& s_out) {

  s_out.clear();

  // Calculate the covisible cost for each feature
  std::priority_queue<pair<int, double>, std::vector<pair<int, double>>, CovisibleCostComparator> score_queue;

  for (const auto& id : s_in) {
    double score = calcCoVisibility(v1, v2, id);
    score_queue.emplace(id, score);
  }

  // Choose too many features will due to the limitation of optimization
  int min_covisible_feature_num_ = param_.min_covisible_feature_num_;

  for (int i = 0; i < min_covisible_feature_num_; ++i) {
    auto id_score = score_queue.top();
    score_queue.pop();

    // cout << "i, score: " << i << " " << id_score.second << endl;
    s_out.insert(id_score.first);
  }
}

bool YawInitialPlanner::handleEndLayer() {
  // Assume that the end layer has been sorted in descending order

  size_t v_num = end_vert_vec_.size();
  Vector3d acc_zero = Vector3d::Zero();

  for (size_t i = 0; i < v_num; i++) {
    auto& v = end_vert_vec_[i];

    // Check if there are sufficient features in stationary(hovering) state
    Quaterniond ori = Utils::calcOrientation(v->yaw_, acc_zero);
    int feature_num = feature_map_->getFeatureUsingOdom(pos_[v->layer_], ori);
    if (feature_num <= param_.min_feature_num_) continue;

    // TODO: Add consideration for explorability

    // We only need the best one for graph search
    end_vert_ = v;
    addVertex(end_vert_);
    return true;
  }

  return false;
}

void YawInitialPlanner::reset() {
  graph_id_ = 0;
  vertice_.clear();
  vert_path_.clear();
}

// Bidirectional stepping to search for the localizable end yaw
bool YawInitialPlanner::refineEndYaw(YawVertex::Ptr& v) {
  const double range = 4 * M_PI / 180;
  const double step = 0.5 * M_PI / 180;

  double origin_yaw = v->yaw_;

  for (double yaw = origin_yaw; yaw <= origin_yaw + range + 1e-2; yaw += step) {
    Utils::roundPi(yaw);
    YawVertex::Ptr vert(new YawVertex(yaw, v->layer_));
    yaw2id(vert);
    setVisbleFeatures(vert);
    if (static_cast<int>(vert->features_id_.size()) > param_.min_feature_num_) {
      v = vert;
      return true;
    }
  }

  for (double yaw = origin_yaw; yaw >= origin_yaw - range - 1e-2; yaw -= step) {
    Utils::roundPi(yaw);
    YawVertex::Ptr vert(new YawVertex(yaw, v->layer_));
    yaw2id(vert);
    setVisbleFeatures(vert);
    if (static_cast<int>(vert->features_id_.size()) > param_.min_feature_num_) {
      v = vert;
      return true;
    }
  }

  return false;
}

bool YawInitialPlanner::search(
    const double start_yaw, const vector<double>& end_yaw_vec, const double& dt, vector<double>& path) {

  reset();

  // Step1: Prepare the parameters
  param_.dt_ = dt;
  // Tips: Adjust the coefficient to avoid search failures
  //  param_.max_diff_yaw_id_ = (2 * param_.max_yaw_rate_ * dt) / (2 * M_PI / param_.piece_num_);
  param_.max_diff_yaw_id_ = (1.5 * param_.max_yaw_rate_ * dt) / (2 * M_PI / param_.piece_num_);
  param_.max_diff_yaw_id_ = std::max(1, param_.max_diff_yaw_id_);

  size_t num_layers = pos_.size();

  // Step2: Set the start vertex
  start_vert_.reset(new YawVertex(start_yaw, 0));
  yaw2id(start_vert_);
  setVisbleFeatures(start_vert_);
  setVisbleFrontiers(start_vert_);
  start_vert_->frontiers_id_path_ = start_vert_->frontiers_id_;
  // Here we dont't care about the localizability of the start vertex

  vector<YawVertex::Ptr> layer, last_layer;

  // Step3: Generate nodes for each layer
  for (size_t i = 0; i < num_layers; ++i) {

    for (const auto& v : last_layer) addVertex(v);

    if (i == 0) {
      layer.push_back(start_vert_);
    }

    else if (i != num_layers - 1) {
      for (const auto& yaw : param_.yaw_samples_) {
        YawVertex::Ptr vert(new YawVertex(yaw, i));
        yaw2id(vert);
        setVisbleFeatures(vert);
        if (static_cast<int>(vert->features_id_.size()) <= param_.min_feature_num_) continue;
        setVisbleFrontiers(vert);
        checkIfVisGoal(vert);
        layer.push_back(vert);
      }
    }

    else {
      for (const auto& end_yaw : end_yaw_vec) {
        YawVertex::Ptr vert(new YawVertex(end_yaw, i));
        yaw2id(vert);
        setVisbleFeatures(vert);
        if (static_cast<int>(vert->features_id_.size()) <= param_.min_feature_num_) {
          ROS_WARN("[yaw initial planner]: End yaw %lf is not localizable!!! Try to adjust.", end_yaw);
          if (!refineEndYaw(vert)) {
            ROS_WARN("[yaw initial planner]: Fail to adjust end yaw!!!");
            continue;
          }

          else
            ROS_INFO("[yaw initial planner]: End yaw %lf is adjusted to %lf.", end_yaw, vert->yaw_);
        }

        setVisbleFrontiers(vert);
        checkIfVisGoal(vert);
        layer.push_back(vert);
      }
    }

    // Check if there is a connection with the previous layer
    if (i != 0) {
      int add_edge_num = 0;

      for (const auto& v1 : last_layer) {
        for (const auto& v2 : layer) {
          if (checkFeasibility(v1, v2) && getCoVisibleNum(v1, v2) > param_.min_covisible_feature_num_) {
            v1->edges_.push_back(v2);
            add_edge_num++;
            v2->candiate_parent_num_++;
          }
        }
      }

      layer.erase(remove_if(layer.begin(), layer.end(), [](const YawVertex::Ptr& v) { return v->candiate_parent_num_ == 0; }),
          layer.end());

      if (add_edge_num == 0) {
        if (!go2final_) {

          int diff_layer = num_layers - i;
          if (diff_layer < pos_.size() / 2) {

            ROS_WARN("[yaw initial planner]: Layer %zu/%zu is disconnected, treating it as the last layer.", i, num_layers);

            // We hope to maintain good explorability at the end of the trajectory
            sort(last_layer.begin(), last_layer.end(), [](const YawVertex::Ptr& a, const YawVertex::Ptr& b) {
              return a->frontiers_id_.size() > b->frontiers_id_.size();
            });

            break;
          }

          else {
            ROS_ERROR("[yaw initial planner]: Layer %zu/%zu is disconnected, give up this search.", i, num_layers);
            return false;
          }
        }

        else {
          ROS_ERROR(
              "[yaw initial planner]: Fail to add edge between layer %zu and layer %zu,total tayer: %zu", i, i + 1, num_layers);
          return false;
        }
      }
    }

    last_layer.clear();
    last_layer.swap(layer);
  }

  // Step4: Handle the last layer separately
  end_vert_vec_.swap(last_layer);
  if (!handleEndLayer()) {
    ROS_ERROR("[yaw initial planner]: Error in handle last layer.");
    return false;
  }

  // Step5(Debug): Check for duplicate graph_id in vertice_
  for (const auto& vertex : vertice_) {
    for (const auto& other_vertex : vertice_) {
      if (vertex != other_vertex && vertex->graph_id_ == other_vertex->graph_id_) {
        ROS_ERROR("Duplicate graph_id found: %d", vertex->graph_id_);
        ROS_BREAK();
      }
    }
  }

  // Step6: Using graph search algorithm to obatain the solution
  if (!graphSearch()) {
    ROS_BREAK();  // In theory, graph search will never fail
    // return false;
  }

  // Step7: Extract yaw waypoints stored in vert
  for (const auto& vert : vert_path_) {
    path.push_back(vert->yaw_);
  }

  return true;
}

void YawInitialPlanner::prepareOptData(const YawOptData::Ptr& data) {
  // Step1: Pos && Acc
  data->pos_vec_.clear();
  data->acc_vec_.clear();

  for (size_t layer = 1; layer < vert_path_.size() - 1; ++layer) {
    YawVertex::Ptr v = vert_path_[layer];

    Vector3d pos = pos_[v->layer_];
    data->pos_vec_.emplace_back(pos);

    Vector3d acc = acc_[v->layer_];
    data->acc_vec_.emplace_back(acc);
  }

  // Step2: Frontier status
  data->frontier_status_.resize(vert_path_.size());

  std::set<int> frontiers_id_path;

  for (size_t layer = 0; layer < vert_path_.size(); ++layer) {
    YawVertex::Ptr vertex = vert_path_[layer];

    data->frontier_status_[layer].resize(target_frontier_.size(), NOT_AVAILABLE);

    for (const auto& id : target_frontier_aft_preprocess_[layer]) {
      data->frontier_status_[layer][id] = AVAILABLE;
    }

    for (const auto& id : vertex->frontiers_id_) {
      data->frontier_status_[layer][id] = VISIBLE;
    }

    for (const auto& id : frontiers_id_path) {
      data->frontier_status_[layer][id] = HAS_BEEN_OBSERVED;
    }

    frontiers_id_path.insert(vertex->frontiers_id_.begin(), vertex->frontiers_id_.end());
  }

  // Remove the first and last elements from data->frontier_status_
  if (!data->frontier_status_.empty()) {
    data->frontier_status_.erase(data->frontier_status_.begin());
    if (!data->frontier_status_.empty()) {
      data->frontier_status_.pop_back();
    }
  }

  // Step3: Target frontier
  data->frontier_cells_ = target_frontier_;

  // Step4: Set the covisible features between two layers
  data->target_covis_features_.clear();
  for (size_t layer = 1; layer < vert_path_.size() - 1; ++layer) {
    YawVertex::Ptr v = vert_path_[layer];
    YawVertex::Ptr v_pre = vert_path_[layer - 1];
    YawVertex::Ptr v_next = vert_path_[layer + 1];

    // choose the covisible features(ID) of this layer with the previous and next layer
    set<int> co_feature_pre, co_feature_next;
    getCoVisibleSet(v, v_pre, co_feature_pre);
    getCoVisibleSet(v, v_next, co_feature_next);

    set<int> co_feature_pre_filtered;
    selectTargetCoVisibleID(v, v_pre, co_feature_pre, co_feature_pre_filtered);
    set<int> co_feature_next_filtered;
    selectTargetCoVisibleID(v, v_next, co_feature_next, co_feature_next_filtered);

    // Merge the two sets
    set<int> co_feature;
    co_feature.insert(co_feature_pre_filtered.begin(), co_feature_pre_filtered.end());
    co_feature.insert(co_feature_next_filtered.begin(), co_feature_next_filtered.end());

    data->target_covis_features_.emplace_back(co_feature);
  }
}

}  // namespace perception_aware_planner