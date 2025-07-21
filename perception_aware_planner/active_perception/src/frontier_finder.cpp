#include "active_perception/frontier_finder.h"

#include "plan_env/edt_environment.h"
#include "plan_env/feature_map.h"
#include "plan_env/sdf_map.h"

#include "traj_utils/planning_visualization.h"

#include <pcl/filters/voxel_grid.h>

#include <unordered_set>

using namespace std;
using namespace Eigen;

namespace perception_aware_planner {

FrontierFinder::FrontierFinder(const EDTEnvironment::Ptr& edt, const FeatureMap::Ptr& fea, ros::NodeHandle& nh) {
  edt_env_ = edt;
  feature_map_ = fea;

  int voxel_num = edt->sdf_map_->getVoxelNum();
  frontier_flag_ = vector<char>(voxel_num, 0);
  fill(frontier_flag_.begin(), frontier_flag_.end(), 0);

  nh.param("frontier/cluster_min", cluster_min_, -1);
  nh.param("frontier/cluster_size_xy", cluster_size_xy_, -1.0);
  nh.param("frontier/min_candidate_clearance", min_candidate_clearance_, -1.0);
  nh.param("frontier/down_sample", down_sample_, -1);
  nh.param("frontier/min_visib_num", min_visib_num_, -1);
  nh.param("frontier/min_view_finish_fraction", min_view_finish_fraction_, -1.0);
  nh.param("frontier/ceiling_dir", ceiling_dir_, -1.0);

  nh.param("frontier/candidate_dphi", frontier_candidate_params_.candidate_dphi_, -1.0);
  nh.param("frontier/candidate_rmax", frontier_candidate_params_.candidate_rmax_, -1.0);
  nh.param("frontier/candidate_rmin", frontier_candidate_params_.candidate_rmin_, -1.0);
  nh.param("frontier/candidate_rnum", frontier_candidate_params_.candidate_rnum_, -1);
  nh.param("frontier/feature_sample_dphi", frontier_candidate_params_.feature_sample_dphi_, -1.0);
  nh.param("frontier/z_sample_max_length", frontier_candidate_params_.z_sample_max_length_, -1.0);
  nh.param("frontier/z_sample_num", frontier_candidate_params_.z_sample_num_, -1);
  nh.param("frontier/cand_limit_per_cluster", frontier_candidate_params_.cand_limit_per_cluster_, -1);

  nh.param("frontier/feature/candidate_dphi", feature_candidate_params_.candidate_dphi_, -1.0);
  nh.param("frontier/feature/candidate_rmax", feature_candidate_params_.candidate_rmax_, -1.0);
  nh.param("frontier/feature/candidate_rmin", feature_candidate_params_.candidate_rmin_, -1.0);
  nh.param("frontier/feature/candidate_rnum", feature_candidate_params_.candidate_rnum_, -1);
  nh.param("frontier/feature/feature_sample_dphi", feature_candidate_params_.feature_sample_dphi_, -1.0);
  nh.param("frontier/feature/z_sample_max_length", feature_candidate_params_.z_sample_max_length_, -1.0);
  nh.param("frontier/feature/z_sample_num", feature_candidate_params_.z_sample_num_, -1);
  nh.param("frontier/feature/cand_limit_per_cluster", feature_candidate_params_.cand_limit_per_cluster_, -1);

  nh.param("frontier/final/candidate_dphi", final_goal_candidate_params_.candidate_dphi_, -1.0);
  nh.param("frontier/final/candidate_rmax", final_goal_candidate_params_.candidate_rmax_, -1.0);
  nh.param("frontier/final/candidate_rmin", final_goal_candidate_params_.candidate_rmin_, -1.0);
  nh.param("frontier/final/candidate_rnum", final_goal_candidate_params_.candidate_rnum_, -1);
  nh.param("frontier/final/feature_sample_dphi", final_goal_candidate_params_.feature_sample_dphi_, -1.0);
  nh.param("frontier/final/z_sample_max_length", final_goal_candidate_params_.z_sample_max_length_, -1.0);
  nh.param("frontier/final/z_sample_num", final_goal_candidate_params_.z_sample_num_, -1);
  nh.param("frontier/final/cand_limit_per_cluster", final_goal_candidate_params_.cand_limit_per_cluster_, -1);

  nh.param("frontier/we", sort_refer_.we, -1.0);
  nh.param("frontier/wg", sort_refer_.wg, -1.0);
  nh.param("frontier/wf", sort_refer_.wf, -1.0);
  nh.param("frontier/wc", sort_refer_.wc, -1.0);

  // For Visualization
  nh.param("frontier/visual_scores", visual_scores, false);
  nh.param("frontier/visual_all_frontier", visual_all_frontier, false);
  nh.param("frontier/visual_feature_cluster", visual_feature_cluster, false);
  nh.param("frontier/visual_frontier_viewpoint", visual_frontier_viewpoint, false);
  nh.param("frontier/visual_feature_viewpoint", visual_feature_viewpoint, false);
  nh.param("frontier/visual_final_viewpoint", visual_final_viewpoint, false);
  nh.param("frontier/visual_astar_path", visual_astar_path_, false);

  raycaster_.reset(new RayCaster);
  resolution_ = edt_env_->sdf_map_->getResolution();
  Eigen::Vector3d origin, size;
  edt_env_->sdf_map_->getRegion(origin, size);
  raycaster_->setParams(resolution_, origin);

  frontier_cam_ = Utils::getGlobalParam().frontier_cam_;
  feature_cam_ = Utils::getGlobalParam().feature_cam_;

  graph_search_.reset(new LocalizationAwareGraphSearch);
  graph_search_->init(nh);

  path_finder_.reset(new Astar);
  path_finder_->init(nh, edt_env_);

  shared_param_.vp_manager_.pos_thr =
      min(frontier_candidate_params_.z_sample_max_length_ / frontier_candidate_params_.z_sample_num_,
          (frontier_candidate_params_.candidate_rmax_ - frontier_candidate_params_.candidate_rmin_) /
              frontier_candidate_params_.candidate_rnum_) /
      2.0;
  shared_param_.vp_manager_.yaw_thr = frontier_candidate_params_.candidate_dphi_ / 2.0;

  running_ = true;
  worker_thread_ = std::thread(&FrontierFinder::frontierThread, this);

  shared_param_.get_final_goal = false;
}

FrontierFinder::~FrontierFinder() {
  running_ = false;  // Stop the thread
  if (worker_thread_.joinable()) {
    worker_thread_.join();  // Wait for the thread to finish
  }
}

void FrontierFinder::frontierThread() {
  while (running_) {

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    if (!begin_) continue;

    // Step1: Share the data with FSM
    updateShareFrontierParam();

    // Step2: Cluster the frontiers and compute their viewpoint
    computeFeatureViewpoint();

    // Step3: Build the graph
    buildFeatureGraph();

    // Step4: Perform the graph search
    searchFrontiers();

    // Step5: Computing the frontiers that need to be visited
    computeFrontiersToVisit();

    // Step6: Compute viewpoints that belong to final goal
    computeFinalViewpoint();

    // Step7: Generate viewpoints at the final goal
    computeViewpointinGoal();

    // Step8: Visualize
    visualFrontier();
  }
}

void FrontierFinder::getShareFrontierParam(const Vector3d& cur_pos, const Vector3d& cur_vel, const double& yaw_now,  // input
    vector<vector<Vector3d>>& active_frontiers, vector<vector<Vector3d>>& dead_frontiers, vector<Vector3d>& points,
    vector<double>& yaws, vector<Vector3d>& frontier_cells, vector<double>& score) {

  std::lock_guard<std::mutex> lock(data_mutex_share_);
  // input
  shared_param_.cur_pos = cur_pos;
  shared_param_.cur_vel = cur_vel;
  shared_param_.yaw_now = yaw_now;

  // output
  active_frontiers = shared_param_.active_frontiers;
  dead_frontiers = shared_param_.dead_frontiers;
  points = shared_param_.viewpoint_pos_vector;
  yaws = shared_param_.viewpoint_yaw_vector;
  frontier_cells = shared_param_.viewpoint_frontier_cell;
  score = shared_param_.score;

  begin_ = true;
}

void FrontierFinder::resetViewpointManager() {
  std::lock_guard<std::mutex> lock(data_mutex_share_);
  shared_param_.vp_manager_.clear();
}

void FrontierFinder::setFinalGoal(const Vector3d& final_goal) {
  std::lock_guard<std::mutex> lock(data_mutex_share_);

  shared_param_.final_goal = final_goal;
  shared_param_.get_final_goal = true;
}

void FrontierFinder::visualFrontier() {
  if (visualization_ == nullptr) return;

  vector<vector<Eigen::Vector3d>> active_frontiers;
  vector<vector<Eigen::Vector3d>> active_paths;
  vector<vector<double>> active_score;
  vector<Eigen::Vector3d> active_pos;
  vector<vector<Eigen::Vector3d>> dead_frontiers;
  vector<vector<Eigen::Vector3d>> dead_paths;
  vector<vector<double>> dead_score;
  vector<Eigen::Vector3d> dead_pos;
  Vector3d points;
  vector<double> yaws;
  vector<Vector3d> frontier_cells;
  vector<double> score;

  if (visual_scores) {
    getFrontiersScore(active_frontiers, active_score, active_pos, dead_frontiers, dead_score, dead_pos);
    visualization_->drawFrontiersScore(active_frontiers, active_score, active_pos, Vector4d(0.6, 0.6, 0.6, 0.25), false);
    visualization_->drawFrontiersScore(dead_frontiers, dead_score, dead_pos, Vector4d(0.0, 0.0, 0.0, 0.6), true);
  }

  else if (visual_all_frontier) {
    getFrontiers(active_frontiers);
    getDormantFrontiers(dead_frontiers);
    visualization_->drawFrontiers(active_frontiers, Vector4d(0.6, 0.6, 0.6, 0.3), false);
    visualization_->drawFrontiers(dead_frontiers, Vector4d(0.1, 0.1, 0.1, 0.4), true);
  }

  else
    getFrontiers(active_frontiers);

  if (active_frontiers.empty()) return;

  if (visual_astar_path_) visualization_->drawAtarPath(frontiers_.front().paths2goal, Vector4d(0.6, 0.0, 0.0, 0.6), 0);

  if (visual_frontier_viewpoint) {
    vector<std::pair<Eigen::Vector3d, double>> vps_visaul;
    for (const auto& frontier : frontiers_) {
      const auto& vp = frontier.viewpoints_.front();
      vps_visaul.push_back(make_pair(vp.pos_, vp.yaw_));
    }
    visualization_->drawViewpoints(vps_visaul, Vector4d(0.6, 0.6, 0.6, 0.6), 0);
  }

  if (visual_final_viewpoint) {
    vector<std::pair<Eigen::Vector3d, double>> vps_visaul;
    int visual_count = 0;
    for (const auto& vp : final_viewpoint_) {
      vps_visaul.push_back(make_pair(vp.pos_, vp.yaw_));
      if (++visual_count >= 3) break;
    }
    for (const auto& vp : goal_viewpoint_) {
      vps_visaul.push_back(make_pair(vp.pos_, vp.yaw_));
      if (++visual_count >= 1) break;
    }
    visualization_->drawViewpoints(vps_visaul, Vector4d(1.0, 0.0, 0.0, 0.6), 1);
  }

  if (visual_feature_cluster) {
    vector<Eigen::Vector3d> box_centers, box_scales;
    for (size_t i = 0; i < feature_cluster.size(); i++) {
      pcl::PointXYZ min_pt, max_pt;
      pcl::getMinMax3D(*feature_cluster[i].second, min_pt, max_pt);
      Eigen::Vector3d box_center((min_pt.x + max_pt.x) / 2.0, (min_pt.y + max_pt.y) / 2.0, (min_pt.z + max_pt.z) / 2.0);
      Eigen::Vector3d box_scale(
          std::abs(max_pt.x - min_pt.x) + 0.1, std::abs(max_pt.y - min_pt.y) + 0.1, std::abs(max_pt.z - min_pt.z) + 0.1);
      box_centers.push_back(box_center);
      box_scales.push_back(box_scale);
    }
    visualization_->drawClusters(box_centers, box_scales, Vector4d(0.6, 0.6, 0.0, 0.8));
  }

  if (visual_feature_viewpoint) {
    vector<pair<Vector3d, double>> vps_visaul;
    for (const auto& vp : feature_viewpoint) vps_visaul.emplace_back(vp.pos_, vp.yaw_);
    visualization_->drawViewpoints(vps_visaul, Vector4d(0.6, 0.6, 0.0, 0.6), 2);
  }
}

void FrontierFinder::updateShareFrontierParam() {
  std::lock_guard<std::mutex> lock(data_mutex_share_);
  // input
  sort_refer_.update(shared_param_.cur_pos, shared_param_.cur_vel, shared_param_.yaw_now);
  sort_refer_.get_final_goal = shared_param_.get_final_goal;
  if (sort_refer_.get_final_goal) sort_refer_.final_goal = shared_param_.final_goal;

  // output
  getFrontiers(shared_param_.active_frontiers);
  getDormantFrontiers(shared_param_.dead_frontiers);
  getSortedViewpointVector(shared_param_.viewpoints);
  getBestViewpointData(shared_param_.viewpoints, shared_param_.viewpoint_pos_vector, shared_param_.viewpoint_yaw_vector,
      shared_param_.viewpoint_frontier_cell, shared_param_.score);
}

void FrontierFinder::getBestViewpointData(const vector<Viewpoint>& available_viewpoints, vector<Vector3d>& points,
    vector<double>& yaws, vector<Vector3d>& frontier_cells, vector<double>& score) {
  yaws.clear();
  frontier_cells.clear();
  score.clear();

  for (const auto& vp : available_viewpoints) {
    points = vp.search_path;
    yaws = vp.search_yaw;
    frontier_cells = vp.filtered_cells_;
    score.push_back(vp.score_yaw_);
    score.push_back(vp.score_pos_);
    score.push_back(vp.final_score_);
    // isViewpointPathFeasible(vp);
    return;
  }

  ROS_WARN_THROTTLE(2.0, "[FrontierFinder::getBestViewpointData] No Best Viewpoint!!!!! %lu", available_viewpoints.size());
}

void FrontierFinder::getSortedViewpointVector(vector<Viewpoint>& viewpoints) {
  viewpoints.clear();

  for (const auto& vp : final_viewpoint_) {
    if (vp.search_path.size() >= 2) viewpoints.push_back(vp);
  }

  for (const auto& vp : goal_viewpoint_) {
    if (vp.search_path.size() >= 2) viewpoints.push_back(vp);
  }

  sort(viewpoints.begin(), viewpoints.end(),
      [](const Viewpoint& v1, const Viewpoint& v2) { return v1.final_score_ > v2.final_score_; });

  for (const auto& frontier : frontiers_) {
    for (const auto& vp : frontier.viewpoints_) {
      if (vp.search_path.size() >= 2) viewpoints.push_back(vp);
    }
  }
}

bool FrontierFinder::chooseNextViewpoint(vector<Vector3d>& points, vector<double>& yaws, vector<Vector3d>& frontier_cells) {

  std::lock_guard<std::mutex> lock(data_mutex_share_);
  for (const auto& vp : shared_param_.viewpoints) {
    auto& vpmgr = shared_param_.vp_manager_;
    if (!vpmgr.isViewpointExist(vp)) {
      points = vp.search_path;
      yaws = vp.search_yaw;
      frontier_cells = vp.filtered_cells_;
      vpmgr.addViewpoint(vp);
      return true;
    }
  }

  ROS_ERROR("[FrontierFinder::chooseNextViewpoint] ERROR !!! NO AVAILABLE FRONTIER!!");
  return false;
}

void FrontierFinder::searchFrontiers() {
  tmp_frontiers_.clear();

  // Bounding box of updated region
  Vector3d update_min, update_max;
  edt_env_->sdf_map_->getUpdatedBox(update_min, update_max, true);

  // Removed changed frontiers in updated map
  auto resetFlag = [&](list<Frontier>::iterator& iter, list<Frontier>& frontiers) {
    Eigen::Vector3i idx;
    for (auto cell : iter->cells_) {
      edt_env_->sdf_map_->posToIndex(cell, idx);
      frontier_flag_[toadr(idx)] = 0;
    }
    iter = frontiers.erase(iter);
  };

  // updateScorePos();
  int rmv_idx = 0;
  for (auto iter = frontiers_.begin(); iter != frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) && isFrontierChanged(*iter)) {
      resetFlag(iter, frontiers_);
    } else {
      ++rmv_idx;
      ++iter;
    }
  }

  for (auto iter = dormant_frontiers_.begin(); iter != dormant_frontiers_.end();) {
    if (haveOverlap(iter->box_min_, iter->box_max_, update_min, update_max) && isFrontierChanged(*iter))
      resetFlag(iter, dormant_frontiers_);
    else
      ++iter;
  }

  // Search new frontier within box slightly inflated from updated box
  Vector3d search_min = update_min - Vector3d(1, 1, 0.5);
  Vector3d search_max = update_max + Vector3d(1, 1, 0.5);
  Vector3d box_min, box_max;
  edt_env_->sdf_map_->getBox(box_min, box_max);
  for (int k = 0; k < 3; ++k) {
    search_min[k] = max(search_min[k], box_min[k]);
    search_max[k] = min(search_max[k], box_max[k]);
  }
  Eigen::Vector3i min_id, max_id;
  edt_env_->sdf_map_->posToIndex(search_min, min_id);
  edt_env_->sdf_map_->posToIndex(search_max, max_id);

  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z) {
        // Scanning the updated region to find seeds of frontiers
        Eigen::Vector3i cur(x, y, z);
        if (frontier_flag_[toadr(cur)] == 0 && knownfree(cur) && isNeighborUnknown(cur)) {
          // Expand from the seed cell to find a complete frontier cluster
          expandFrontier(cur);
        }
      }

  splitLargeFrontiers(tmp_frontiers_);

  // ROS_WARN_THROTTLE(5.0, "Frontier t: %lf", (ros::Time::now() - t1).toSec());
}

void FrontierFinder::expandFrontier(const Eigen::Vector3i& first) {

  // Data for clustering
  queue<Eigen::Vector3i> cell_queue;
  vector<Eigen::Vector3d> expanded;
  Vector3d pos;

  edt_env_->sdf_map_->indexToPos(first, pos);
  expanded.push_back(pos);
  cell_queue.push(first);
  frontier_flag_[toadr(first)] = 1;

  // Search frontier cluster based on region growing (distance clustering)
  while (!cell_queue.empty()) {
    auto cur = cell_queue.front();
    cell_queue.pop();
    auto nbrs = allNeighbors(cur);
    for (const auto& nbr : nbrs) {
      // Qualified cell should be inside bounding box and frontier cell not clustered
      int adr = toadr(nbr);
      if (frontier_flag_[adr] == 1 || !edt_env_->sdf_map_->isInBox(nbr) || !(knownfree(nbr) && isNeighborUnknown(nbr))) continue;

      edt_env_->sdf_map_->indexToPos(nbr, pos);
      if (pos[2] < 0.4) continue;  // Remove noise close to ground
      expanded.push_back(pos);
      cell_queue.push(nbr);
      frontier_flag_[adr] = 1;
    }
  }

  if (static_cast<int>(expanded.size()) > cluster_min_) {
    // Compute detailed info
    Frontier frontier;
    frontier.cells_ = expanded;
    computeFrontierInfo(frontier);
    tmp_frontiers_.push_back(frontier);
  }
}

void FrontierFinder::splitLargeFrontiers(list<Frontier>& frontiers) {
  list<Frontier> splits, tmps;
  for (auto it = frontiers.begin(); it != frontiers.end(); ++it) {
    // Check if each frontier needs to be split horizontally
    if (splitHorizontally(*it, splits)) {
      tmps.insert(tmps.end(), splits.begin(), splits.end());
      splits.clear();
    } else
      tmps.push_back(*it);
  }
  frontiers = tmps;
}

bool FrontierFinder::splitHorizontally(const Frontier& frontier, list<Frontier>& splits) {
  // Split a frontier into small piece if it is too large
  auto mean = frontier.average_.head<2>();
  bool need_split = false;
  for (const auto& cell : frontier.filtered_cells_) {
    if ((cell.head<2>() - mean).norm() > cluster_size_xy_) {
      need_split = true;
      break;
    }
  }
  if (!need_split) return false;

  // Compute principal component
  // Covariance matrix of cells
  Eigen::Matrix2d cov = Eigen::Matrix2d::Zero();
  for (const auto& cell : frontier.filtered_cells_) {
    Eigen::Vector2d diff = cell.head<2>() - mean;
    cov += diff * diff.transpose();
  }
  cov /= frontier.filtered_cells_.size();

  // Find eigenvector corresponds to maximal eigenvector
  Eigen::EigenSolver<Eigen::Matrix2d> es(cov);
  auto values = es.eigenvalues().real();
  auto vectors = es.eigenvectors().real();
  int max_idx;
  double max_eigenvalue = -1000000;
  for (int i = 0; i < values.rows(); ++i) {
    if (values[i] > max_eigenvalue) {
      max_idx = i;
      max_eigenvalue = values[i];
    }
  }
  Eigen::Vector2d first_pc = vectors.col(max_idx);

  // Split the frontier into two groups along the first PC
  Frontier ftr1, ftr2;
  for (const auto& cell : frontier.cells_) {
    if ((cell.head<2>() - mean).dot(first_pc) >= 0)
      ftr1.cells_.push_back(cell);
    else
      ftr2.cells_.push_back(cell);
  }
  computeFrontierInfo(ftr1);
  computeFrontierInfo(ftr2);

  // Recursive call to split frontier that is still too large
  list<Frontier> splits2;
  if (splitHorizontally(ftr1, splits2)) {
    splits.insert(splits.end(), splits2.begin(), splits2.end());
    splits2.clear();
  } else
    splits.push_back(ftr1);

  if (splitHorizontally(ftr2, splits2))
    splits.insert(splits.end(), splits2.begin(), splits2.end());
  else
    splits.push_back(ftr2);

  return true;
}

bool FrontierFinder::haveOverlap(const Vector3d& min1, const Vector3d& max1, const Vector3d& min2, const Vector3d& max2) {
  // Check if two box have overlap part
  Vector3d bmin, bmax;
  for (int i = 0; i < 3; ++i) {
    bmin[i] = max(min1[i], min2[i]);
    bmax[i] = min(max1[i], max2[i]);
    if (bmin[i] > bmax[i] + 1e-3) return false;
  }
  return true;
}

bool FrontierFinder::isFrontierChanged(const Frontier& ft) {
  for (auto cell : ft.cells_) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && isNeighborUnknown(idx))) return true;
  }
  return false;
}

bool FrontierFinder::isFrontierCeiling(const Frontier& ft) {
  return ft.box_max_.z() - ft.box_min_.z() < ceiling_dir_;
}

void FrontierFinder::computeFrontierInfo(Frontier& ftr) {
  // Compute average position and bounding box of cluster
  ftr.average_.setZero();
  ftr.box_max_ = ftr.box_min_ = ftr.cells_.front();
  for (const auto& cell : ftr.cells_) {
    ftr.average_ += cell;
    for (int i = 0; i < 3; ++i) {
      ftr.box_min_[i] = min(ftr.box_min_[i], cell[i]);
      ftr.box_max_[i] = max(ftr.box_max_[i], cell[i]);
    }
  }
  ftr.average_ /= ftr.cells_.size();

  // Compute downsampled cluster
  downsample(ftr.cells_, ftr.filtered_cells_);
}

void FrontierFinder::sortFrontiers() {

  const double& max_vel = Utils::getGlobalParam().max_vel_;
  for (auto& frontier : frontiers_) {
    frontier.paths2goal.clear();
    frontier.path_cost.clear();
    frontier.path_cost.resize(4);

    for (auto& vp : frontier.viewpoints_) {
      if (vp.search_path.size() < 2) {
        ROS_ERROR("[sortFrontiers] No Path To Best Viewpoint!!!!");
        continue;
      }

      path_finder_->reset();
      // auto start = std::chrono::high_resolution_clock::now();
      int res = path_finder_->search(vp.pos_, sort_refer_.final_goal);
      // cout << "res: " << res << endl;
      // auto end = std::chrono::high_resolution_clock::now();
      // std::chrono::duration<double> elapsed = end - start;
      // ROS_INFO("A* search took %f seconds", elapsed.count());
      if (res == Astar::REACH_END) {
        frontier.paths2goal = path_finder_->getPath();
        if (frontier.paths2goal.size() < 2) {
          frontier.path_cost.assign(4, 0.0);
          break;
        }
        Vector3d path_dir = frontier.paths2goal[1] - frontier.paths2goal[0];
        Vector3d odom_dir(std::cos(vp.yaw_), std::sin(vp.yaw_), 0);
        double v_compensate = odom_dir.dot(path_dir.normalized()) * max_vel;
        double t_compensate = (max_vel - v_compensate) / Utils::getGlobalParam().max_acc_;
        double t_final = frontier.paths2goal.size() * path_finder_->getResolution() / max_vel;

        double cost2final = sort_refer_.wg * (t_final + t_compensate);
        double cost2vp = -vp.score_pos_;

        frontier.path_cost[0] = cost2vp / sort_refer_.wc;
        frontier.path_cost[1] = t_compensate;
        frontier.path_cost[2] = t_final;
        frontier.path_cost[3] = cost2final + cost2vp;
        break;
      }

      else {
        frontier.path_cost[0] = -vp.score_pos_ / sort_refer_.wc;
        frontier.path_cost[1] = HUGE_COST_;
        frontier.path_cost[2] = HUGE_COST_;
        frontier.path_cost[3] = 3 * HUGE_COST_;
      }
    }
  }

  frontiers_.sort([](const Frontier& f1, const Frontier& f2) { return f1.path_cost[3] < f2.path_cost[3]; });
}

void FrontierFinder::computeFrontiersToVisit() {
  first_new_ftr_ = frontiers_.end();
  int new_num = 0;
  int new_dormant_num = 0;
  // update path for old frontier
  for (auto& old_ftr : frontiers_) {
    // Search viewpoints around frontier
    for (auto& vp : old_ftr.viewpoints_) {
      graph_search_->SearchViewpoint(
          vp.pos_, vp.yaw_, vp.visual_features_ids_, NodeType::FRONTIER_SAMPLE, vp.search_path, vp.search_yaw, vp.search_cost);
      computeConsistScore(vp);
      vp.final_score_ = vp.score_pos_ + vp.score_yaw_;
    }
    sort(old_ftr.viewpoints_.begin(), old_ftr.viewpoints_.end(),
        [](const Viewpoint& v1, const Viewpoint& v2) { return v1.final_score_ > v2.final_score_; });
  }

  // Try find viewpoints for each cluster and categorize them according to viewpoint number with feature number threshold
  for (auto& tmp_ftr : tmp_frontiers_) {
    // Search viewpoints around frontier
    if (isFrontierCeiling(tmp_ftr)) {
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
      continue;
    }

    computeFrontierViewpoint(tmp_ftr);
    if (!tmp_ftr.viewpoints_.empty()) {
      ++new_num;
      list<Frontier>::iterator inserted = frontiers_.insert(frontiers_.end(), tmp_ftr);
      if (first_new_ftr_ == frontiers_.end()) first_new_ftr_ = inserted;
    }

    else {
      // Find no viewpoint, move cluster to dormant list
      dormant_frontiers_.push_back(tmp_ftr);
      ++new_dormant_num;
    }
  }

  int idx = 0;
  for (auto& ft : frontiers_) ft.id_ = idx++;
  sortFrontiers();
}

void FrontierFinder::getFrontiers(vector<vector<Eigen::Vector3d>>& clusters) {
  clusters.clear();
  for (const auto& frontier : frontiers_) clusters.push_back(frontier.cells_);
}

void FrontierFinder::getFrontiersScore(vector<vector<Vector3d>>& active_clusters, vector<vector<double>>& active_score,
    vector<Vector3d>& active_pos, vector<vector<Vector3d>>& dead_clusters, vector<vector<double>>& dead_score,
    vector<Vector3d>& dead_pos) {

  active_clusters.clear();
  active_score.clear();
  active_pos.clear();
  dead_clusters.clear();
  dead_score.clear();
  dead_pos.clear();

  for (const auto& frontier : frontiers_) {
    active_clusters.push_back(frontier.cells_);

    vector<double> score;
    if (frontier.viewpoints_.empty())
      score.assign(4, 1000.0);
    else
      score = frontier.path_cost;

    active_score.push_back(score);
    active_pos.push_back(Utils::calculateTopMiddlePoint(frontier.filtered_cells_));
  }

  for (const auto& frontier : dormant_frontiers_) {
    dead_clusters.push_back(frontier.cells_);

    vector<double> score;
    if (frontier.viewpoints_.empty())
      score.assign(4, 1000.0);
    else
      score = frontier.path_cost;

    dead_score.push_back(score);
    dead_pos.push_back(Utils::calculateTopMiddlePoint(frontier.filtered_cells_));
  }
}

void FrontierFinder::getAstarPath(vector<vector<Vector3d>>& active_path, vector<vector<Vector3d>>& dead_path) {
  active_path.clear();
  dead_path.clear();
  for (const auto& frontier : frontiers_) active_path.push_back(frontier.paths2goal);
  for (const auto& frontier : dormant_frontiers_) dead_path.push_back(frontier.paths2goal);
}

void FrontierFinder::getDormantFrontiers(vector<vector<Vector3d>>& clusters) {
  clusters.clear();
  for (const auto& ft : dormant_frontiers_) clusters.push_back(ft.cells_);
}

// Sampling near a given center to find candidate viewpoints
void FrontierFinder::sampleCylindricalViewpoints(
    const CandidateParams& params, const Vector3d& center, vector<Viewpoint>& sampled_viewpoints) {

  sampled_viewpoints.clear();
  // Sampling to obtain the yaw vector corresponding to each Viewpoint
  vector<double> sample_yaw;
  for (double phi = -M_PI; phi < M_PI - 1e-5; phi += params.feature_sample_dphi_) {
    sample_yaw.push_back(phi);
  }

  for (double rc = params.candidate_rmin_, dr = (params.candidate_rmax_ - params.candidate_rmin_) / params.candidate_rnum_;
      rc <= params.candidate_rmax_ + 1e-3; rc += dr) {  // Radius sampling
    for (double z = center.z() - params.z_sample_max_length_; z <= center.z() + params.z_sample_max_length_ + 1e-3;
        z += 2 * params.z_sample_max_length_ / params.z_sample_num_) {       // Height Sampling
      for (double phi = -M_PI; phi < M_PI; phi += params.candidate_dphi_) {  // Angle Sampling
        Vector3d sample_pos = center + rc * Vector3d(cos(phi), sin(phi), 0);
        sample_pos.z() = z;

        double dist = edt_env_->evaluateCoarseEDT(sample_pos, -1.0);
        // Qualified viewpoint is in bounding box and in safe region
        if (!edt_env_->sdf_map_->isInBox(sample_pos) || edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1 ||
            isNearUnknown(sample_pos) || dist < 0.4)
          continue;

        vector<vector<int>> features_ids_per_yaw;
        feature_map_->getYawRangeUsingPos(sample_pos, sample_yaw, features_ids_per_yaw, raycaster_.get());

        // Determine whether the viewpoint meets the requirements
        for (size_t sample_id = 0; sample_id < sample_yaw.size(); ++sample_id) {
          int feature_num = features_ids_per_yaw[sample_id].size();
          if (feature_num <= Utils::getGlobalParam().min_feature_num_plan_) continue;

          Viewpoint vp;
          vp.pos_ = sample_pos;
          vp.yaw_ = sample_yaw[sample_id];
          vp.visual_features_ids_ = features_ids_per_yaw[sample_id];
          sampled_viewpoints.push_back(vp);
        }
      }
    }
  }
}

// Sample Viewpoints at the junction of two different clusters
void FrontierFinder::sampleJunctionVPs(const CandidateParams& params,
    const std::vector<std::pair<Vector3d, pcl::PointCloud<pcl::PointXYZ>::Ptr>>& clustered_results,
    vector<Viewpoint>& sampled_viewpoints) {

  sampled_viewpoints.clear();

  // Param
  const double& candidate_rmax = feature_cam_->visual_max;
  double vertical_rad_max = feature_cam_->fov_vertical * (M_PI / 180.0);
  double horizontal_rad_max = feature_cam_->fov_horizontal * (M_PI / 180.0);
  double vertical_dis_max = candidate_rmax * std::sqrt(2 * (1 - cos(vertical_rad_max)));
  double horizontal_dis_max = candidate_rmax * std::sqrt(2 * (1 - cos(horizontal_rad_max)));

  for (size_t cluster_i = 0; cluster_i < clustered_results.size(); cluster_i++) {
    for (size_t cluster_j = cluster_i + 1; cluster_j < clustered_results.size(); cluster_j++) {
      const Vector3d& center1 = clustered_results[cluster_i].first;
      const Vector3d& center2 = clustered_results[cluster_j].first;

      Vector3d center1_2 = center2 - center1;
      if (std::pow(center1_2.x(), 2) + std::pow(center1_2.y(), 2) >= std::pow(horizontal_dis_max, 2) ||
          std::abs(center1_2.z()) >= vertical_dis_max)
        continue;

      // Begin sample around center1
      for (double rc = params.candidate_rmin_, dr = (candidate_rmax - params.candidate_rmin_) / params.candidate_rnum_;
          rc <= candidate_rmax + 1e-3; rc += dr) {  // Radius sampling
        for (double z = center1.z() - params.z_sample_max_length_; z <= center1.z() + params.z_sample_max_length_ + 1e-3;
            z += 2 * params.z_sample_max_length_ / params.z_sample_num_) {       // Height Sampling
          for (double phi = -M_PI; phi < M_PI; phi += params.candidate_dphi_) {  // Angle Sampling
            Vector3d sample_pos = center1 + rc * Vector3d(cos(phi), sin(phi), 0);
            sample_pos.z() = z;
            Vector3d sample_feature_cam_pos;
            feature_cam_->fromOdom2Cam(sample_pos, sample_feature_cam_pos);

            double dist = edt_env_->evaluateCoarseEDT(sample_pos, -1.0);
            // Qualified viewpoint is in bounding box and in safe region
            if (!edt_env_->sdf_map_->isInBox(sample_pos) || edt_env_->sdf_map_->getInflateOccupancy(sample_pos) == 1 ||
                isNearUnknown(sample_pos) || dist < 0.4)
              continue;

            // Far away from center2
            Vector3d vec_sample_2 = center2 - sample_pos;
            if (std::pow(vec_sample_2.x(), 2) + std::pow(vec_sample_2.y(), 2) >= std::pow(candidate_rmax, 2) ||
                std::abs(vec_sample_2.z()) >= 2 * params.z_sample_max_length_)
              continue;

            // Occlusion by obstacles
            if (edt_env_->sdf_map_->checkObstacleBetweenPoints(sample_feature_cam_pos, center1, raycaster_.get())) continue;
            if (edt_env_->sdf_map_->checkObstacleBetweenPoints(sample_feature_cam_pos, center2, raycaster_.get())) continue;

            // is Angle Wrong
            Vector3d vec_sample_1 = center1 - sample_pos;
            // 1. compute horizontal angle
            Eigen::Vector3d vec1_xy(vec_sample_1.x(), vec_sample_1.y(), 0);
            Eigen::Vector3d vec2_xy(vec_sample_2.x(), vec_sample_2.y(), 0);
            double dot_product_xy = vec1_xy.dot(vec2_xy);
            double norm_xy_1 = vec1_xy.norm();
            double norm_xy_2 = vec2_xy.norm();
            double horizontal_angle = std::acos(dot_product_xy / (norm_xy_1 * norm_xy_2));
            // 2. compute 3D angle
            double dot_product_3d = vec_sample_1.dot(vec_sample_2);
            double norm_1 = vec_sample_1.norm();
            double norm_2 = vec_sample_2.norm();
            double full_3d_angle = std::acos(dot_product_3d / (norm_1 * norm_2));
            // 3. compute vertical angle
            double vertical_angle = full_3d_angle - horizontal_angle;
            // 4. compare
            if (horizontal_angle > horizontal_rad_max || vertical_angle > vertical_rad_max) continue;

            Vector3d yaw_dir = vec_sample_1.normalized() + vec_sample_2.normalized();
            Viewpoint vp;
            vp.pos_ = sample_pos;
            vp.yaw_ = atan2(yaw_dir.y(), yaw_dir.x());

            feature_map_->getFeatureIDUsingPosYaw(vp.pos_, vp.yaw_, vp.visual_features_ids_, raycaster_.get());
            int feature_num = vp.visual_features_ids_.size();
            if (feature_num > Utils::getGlobalParam().min_feature_num_plan_) sampled_viewpoints.push_back(vp);
          }
        }
      }
    }
  }
}

void FrontierFinder::computeConsistScore(Viewpoint& viewpoint) {
  /** Note：
   * Calculate consistency score for sorting viewpoints
   * Further verification is required
   **/

  if (viewpoint.search_cost.size() < 2) {  // Fail in graph search
    viewpoint.search_path = vector<Vector3d>{ sort_refer_.pos_now_, viewpoint.pos_ };
    viewpoint.search_yaw = vector<double>{ sort_refer_.yaw_now_, viewpoint.yaw_ };
    viewpoint.score_pos_ = 0;
  }

  else {
    viewpoint.score_pos_ = 0;
  }

  double t_move = 0.0;  // Estimation of the shortest time from current state to this viewpoint

  const double& max_vel = Utils::getGlobalParam().max_vel_;
  const double& max_acc = Utils::getGlobalParam().max_acc_;
  const double& max_yaw_rate = Utils::getGlobalParam().max_yaw_rate_;

  vector<Vector3d> vel_estimat_vec;
  vel_estimat_vec.resize(viewpoint.search_path.size());
  vel_estimat_vec[0] = sort_refer_.vel_now_;

  for (size_t i = 0; i < viewpoint.search_path.size() - 1; i++) {
    double t_pos, t_yaw;
    Vector3d diff_pos = viewpoint.search_path[i + 1] - viewpoint.search_path[i];
    double diff_pos_norm = diff_pos.norm();
    if (diff_pos_norm < 1e-6) continue;

    // Modeled as uniform acceleration motion
    double v1 = vel_estimat_vec[i].dot(diff_pos) / diff_pos_norm;
    double v2 = max_vel;

    double s1 = (v2 * v2 - v1 * v1) / (2 * max_acc);
    if (s1 < diff_pos_norm) {
      t_pos = (v2 - v1) / max_acc + (diff_pos_norm - s1) / max_vel;
      vel_estimat_vec[i + 1] = diff_pos.normalized() * max_vel;
    }

    else {
      t_pos = std::sqrt(v1 * v1 + 2 * max_acc * diff_pos_norm) / max_acc - v1 / max_acc;
      vel_estimat_vec[i + 1] = diff_pos.normalized() * (v1 + max_acc * t_pos);
    }

    double diff_yaw = std::abs(viewpoint.search_yaw[i + 1] - viewpoint.search_yaw[i]);
    diff_yaw = std::min(diff_yaw, 2 * M_PI - diff_yaw);
    t_yaw = diff_yaw / max_yaw_rate;

    t_move += std::max(t_pos, t_yaw);
  }

  // Some compensation
  Vector3d yaw_dir(std::cos(viewpoint.yaw_), std::sin(viewpoint.yaw_), 0);
  double v_start = vel_estimat_vec.back().dot(yaw_dir);
  t_move += (max_vel - v_start) / max_acc;

  // The shorter the time, the higher the score
  viewpoint.score_pos_ -= sort_refer_.wc * t_move;
}

void FrontierFinder::computeFrontierViewpoint(Frontier& frontier) {
  // Evaluate sample viewpoints on circles
  vector<Viewpoint> frontier_viewpoint_sample_;
  sampleCylindricalViewpoints(frontier_candidate_params_, frontier.average_, frontier_viewpoint_sample_);

  vector<Vector3d>& cells = frontier.filtered_cells_;

  // Compute score
  for (auto& vp : frontier_viewpoint_sample_) {
    vector<Vector3d> visib_cell;
    // Initial Path score
    // vp.score_pos_ = -HUGE_COST_;

    // Compute Yaw score
    if (countVisibleCells(vp.pos_, vp.yaw_, cells, visib_cell) > min_visib_num_) {
      double explorability = calExplorability(vp.pos_, vp.yaw_, visib_cell);
      vp.filtered_cells_ = visib_cell;
      vp.score_yaw_ = sort_refer_.we * explorability / frontier_cam_->visual_max;
      // Search Graph and Compute Path score
      graph_search_->SearchViewpoint(
          vp.pos_, vp.yaw_, vp.visual_features_ids_, NodeType::FRONTIER_SAMPLE, vp.search_path, vp.search_yaw, vp.search_cost);
      computeConsistScore(vp);  // Have safe path
    }

    else {
      vp.score_yaw_ = -HUGE_COST_;  // No effect
      vp.score_pos_ = -HUGE_COST_;  // No effect
    }

    vp.final_score_ = vp.score_pos_ + vp.score_yaw_;
  }

  // Sort frontier_viewpoint_sample_
  sort(frontier_viewpoint_sample_.begin(), frontier_viewpoint_sample_.end(),
      [](const Viewpoint& v1, const Viewpoint& v2) { return v1.final_score_ > v2.final_score_; });

  // Push frontier_viewpoint_sample to frontier.viewpoints_ in order
  int vp_num = 0;
  for (size_t i = 0; vp_num < frontier_candidate_params_.cand_limit_per_cluster_ && i < frontier_viewpoint_sample_.size(); ++i) {
    auto& vp = frontier_viewpoint_sample_[i];
    // The pos traj may be valid again in the subsequent movement, but once the yaw traj is not satisfied, it
    // means that this viewpoint is permanently not a suitable temporary target point.
    if (vp.score_yaw_ > -HUGE_COST_ / 2) {
      vp_num++;
      frontier.viewpoints_.push_back(vp);
    }
  }

  // Note that the path score will change as the starting point is updated, so the path score of the subsequent VIEWPIOINT
  // needs to be updated again
}

bool FrontierFinder::computeAstarPath(Frontier& frontier) {
  if (!sort_refer_.get_final_goal) return false;

  path_finder_->reset();
  frontier.paths2goal.clear();
  if (frontier.viewpoints_.empty()) return false;
  if (path_finder_->search(frontier.viewpoints_.front().pos_, sort_refer_.final_goal)) {
    frontier.paths2goal = path_finder_->getPath();
    return true;
  }

  return false;
}

void FrontierFinder::computeFinalViewpoint() {
  // Evaluate sample viewpoints on circles
  if (!sort_refer_.get_final_goal) return;

  const Vector3d& pos_final = sort_refer_.final_goal;
  vector<Viewpoint> final_viewpoint_sample_;
  sampleCylindricalViewpoints(final_goal_candidate_params_, pos_final, final_viewpoint_sample_);

  for (auto& vp : final_viewpoint_sample_) {
    // Check whether final goal is in FOV
    Eigen::Vector3d camera_pose;
    Eigen::Quaterniond camera_orient;
    frontier_cam_->fromOdom2Cam(
        vp.pos_, Eigen::Quaterniond(Eigen::AngleAxisd(vp.yaw_, Eigen::Vector3d::UnitZ())), camera_pose, camera_orient);

    vp.final_score_ = -HUGE_COST_;
    if (!frontier_cam_->inFOV(camera_pose, pos_final, camera_orient)) continue;
    if (edt_env_->sdf_map_->checkObstacleBetweenPoints(camera_pose, pos_final, raycaster_.get())) continue;

    // compute score(all of the final_viewpoint can see the final, choose the one which can easylier reach, use time as cost)
    Vector3d dir = pos_final - vp.pos_;
    double diff_yaw = std::abs(vp.yaw_ - atan2(dir.y(), dir.x()));
    diff_yaw = std::min(diff_yaw, 2 * M_PI - diff_yaw);
    vp.score_pos_ = dir.norm() / Utils::getGlobalParam().max_vel_;
    vp.score_yaw_ = diff_yaw / Utils::getGlobalParam().max_yaw_rate_;
    vp.final_score_ = HUGE_COST_ - max(vp.score_pos_, vp.score_yaw_);  // The shorter the time, the higher the score
  }

  // Push final_viewpoint_sample_ to final_viewpoint_ in order
  sort(final_viewpoint_sample_.begin(), final_viewpoint_sample_.end(),
      [](const Viewpoint& v1, const Viewpoint& v2) { return v1.final_score_ > v2.final_score_; });

  final_viewpoint_.clear();
  int vp_num = 0;
  for (size_t i = 0; vp_num < final_goal_candidate_params_.cand_limit_per_cluster_ && i < final_viewpoint_sample_.size(); ++i) {
    auto& vp = final_viewpoint_sample_[i];
    if (vp.final_score_ > -HUGE_COST_) {
      vp_num++;
      graph_search_->SearchViewpoint(
          vp.pos_, vp.yaw_, vp.visual_features_ids_, NodeType::FINAL_SAMPLE, vp.search_path, vp.search_yaw, vp.search_cost);
      final_viewpoint_.push_back(vp);
    }
  }
}

void FrontierFinder::computeViewpointinGoal() {
  if (!sort_refer_.get_final_goal) return;

  const Vector3d& pos_final = sort_refer_.final_goal;
  if (edt_env_->sdf_map_->getOccupancy(pos_final) != SDFMap::FREE) return;

  // sample in final goal
  vector<double> sample_yaw;
  for (double phi = -M_PI; phi < M_PI; phi += 5 * M_PI / 180) {
    sample_yaw.push_back(phi);
  }
  vector<Viewpoint> goal_viewpoint_sample_;

  vector<vector<int>> features_ids_per_yaw;
  feature_map_->getYawRangeUsingPos(pos_final, sample_yaw, features_ids_per_yaw, raycaster_.get());

  // Determine whether the viewpoint meets the requirements
  for (size_t sample_id = 0; sample_id < sample_yaw.size(); ++sample_id) {
    int feature_num = features_ids_per_yaw[sample_id].size();
    if (feature_num <= Utils::getGlobalParam().min_feature_num_plan_) continue;

    Viewpoint vp;
    vp.pos_ = pos_final;
    vp.yaw_ = sample_yaw[sample_id];
    vp.visual_features_ids_ = features_ids_per_yaw[sample_id];
    vp.score_pos_ = HUGE_COST_;
    double diff_yaw = std::abs(vp.yaw_ - sort_refer_.yaw_now_);
    diff_yaw = std::min(diff_yaw, 2 * M_PI - diff_yaw);
    vp.score_yaw_ = HUGE_COST_ - diff_yaw / Utils::getGlobalParam().max_yaw_rate_;
    vp.final_score_ = vp.score_pos_ + vp.score_yaw_;
    goal_viewpoint_sample_.push_back(vp);
  }

  sort(goal_viewpoint_sample_.begin(), goal_viewpoint_sample_.end(),
      [](const Viewpoint& v1, const Viewpoint& v2) { return v1.final_score_ > v2.final_score_; });

  goal_viewpoint_.clear();
  int vp_num = 0;
  for (size_t i = 0; vp_num < 10 && i < goal_viewpoint_sample_.size(); ++i) {
    vp_num++;
    auto& vp = goal_viewpoint_sample_[i];
    graph_search_->SearchViewpoint(
        vp.pos_, vp.yaw_, vp.visual_features_ids_, NodeType::FINAL, vp.search_path, vp.search_yaw, vp.search_cost);
    goal_viewpoint_.push_back(vp);
  }
}

void FrontierFinder::computeFeatureViewpoint() {
  // Cluster all current feature points
  feature_map_->clusterFeatures(sort_refer_.pos_now_, 1.0, 10, 100, feature_cluster);
  feature_viewpoint.clear();

  for (size_t i = 0; i < feature_cluster.size(); i++) {
    const auto& cluster = feature_cluster[i];
    const Vector3d& cluster_center = cluster.first;

    // Sampling around each cluster
    vector<Viewpoint> feature_viewpoint_sample_;
    sampleCylindricalViewpoints(feature_candidate_params_, cluster_center, feature_viewpoint_sample_);

    // The more cells that can be seen at the same time, the easier it is to transfer this state.
    for (auto& vp : feature_viewpoint_sample_) {
      // score_pos_ is determined by whether the cluster center can be seen.
      if (!feature_cam_->inFOVOdom(vp.pos_, vp.yaw_, cluster_center))
        vp.score_pos_ = -HUGE_COST_;
      else
        vp.score_pos_ = 0;
      vp.score_yaw_ = sort_refer_.wf * vp.visual_features_ids_.size();
      vp.final_score_ = vp.score_pos_ + vp.score_yaw_;
    }

    // Sort frontier_viewpoint_sample_
    sort(feature_viewpoint_sample_.begin(), feature_viewpoint_sample_.end(),
        [](const Viewpoint& v1, const Viewpoint& v2) { return v1.final_score_ > v2.final_score_; });

    int vp_num = 0;
    for (size_t i = 0; vp_num < feature_candidate_params_.cand_limit_per_cluster_ && i < feature_viewpoint_sample_.size(); ++i) {
      vp_num++;
      feature_viewpoint.push_back(feature_viewpoint_sample_[i]);
    }
  }

  vector<Viewpoint> junction_vps;
  sampleJunctionVPs(feature_candidate_params_, feature_cluster, junction_vps);
  feature_viewpoint.insert(feature_viewpoint.end(), junction_vps.begin(), junction_vps.end());
}

bool FrontierFinder::isinterstFrontierCovered(vector<Vector3d>& frontier_cells) {
  std::lock_guard<std::mutex> lock(data_mutex_share_);
  const int change_thresh = min_view_finish_fraction_ * frontier_cells.size();
  int change_num = 0;
  for (const auto& cell : frontier_cells) {
    Eigen::Vector3i idx;
    edt_env_->sdf_map_->posToIndex(cell, idx);
    if (!(knownfree(idx) && isNeighborUnknown(idx)) && ++change_num >= change_thresh) return true;
  }
  return false;
}

bool FrontierFinder::isNearUnknown(const Vector3d& pos) {
  const int vox_num = floor(min_candidate_clearance_ / resolution_);
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -1; z <= 1; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) == SDFMap::UNKNOWN) return true;
      }
  return false;
}

bool FrontierFinder::isNearKnown(const Vector3d& pos) {
  const int vox_num = floor(min_candidate_clearance_ / resolution_);
  for (int x = -vox_num; x <= vox_num; ++x)
    for (int y = -vox_num; y <= vox_num; ++y)
      for (int z = -1; z <= 1; ++z) {
        Eigen::Vector3d vox;
        vox << pos[0] + x * resolution_, pos[1] + y * resolution_, pos[2] + z * resolution_;
        if (edt_env_->sdf_map_->getOccupancy(vox) != SDFMap::UNKNOWN) return true;
      }
  return false;
}

int FrontierFinder::countVisibleCells(
    const Vector3d& pos, const double yaw, const vector<Vector3d>& cluster, vector<Vector3d>& cluster_visual) {

  cluster_visual.reserve(cluster.size());

  Eigen::Vector3i idx;
  Eigen::AngleAxisd angle_axis(yaw, Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond odom_orient(angle_axis);
  Eigen::Vector3d camera_pos;
  Eigen::Quaterniond camera_orient;
  frontier_cam_->fromOdom2Cam(pos, odom_orient, camera_pos, camera_orient);
  int visib_num = 0;

  for (size_t i = 0; i < cluster.size(); ++i) {
    const auto& cell = cluster[i];
    // Check if frontier cell is inside FOV
    if (!frontier_cam_->inFOV(camera_pos, cell, camera_orient)) continue;

    // Check if frontier cell is visible (not occulded by obstacles)
    raycaster_->input(cell, camera_pos);
    bool visib = true;
    while (raycaster_->nextId(idx)) {
      if (edt_env_->sdf_map_->getInflateOccupancy(idx) == 1 || edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::UNKNOWN) {
        visib = false;
        break;
      }
    }
    if (visib) {
      cluster_visual.push_back(cell);
      visib_num++;
    }
  }

  return visib_num;
}

double FrontierFinder::calExplorability(const Vector3d& pos, const double& yaw, const vector<Vector3d>& cluster) {
  Eigen::AngleAxisd angle_axis(yaw, Eigen::Vector3d::UnitZ());
  Eigen::Quaterniond odom_orient(angle_axis);
  Eigen::Vector3d camera_pos;
  Eigen::Quaterniond camera_orient;
  frontier_cam_->fromOdom2Cam(pos, odom_orient, camera_pos, camera_orient);

  double explorability = 0.0;
  const double& visual_max = frontier_cam_->visual_max;

  for (size_t i = 0; i < cluster.size(); ++i) {
    const auto& cell = cluster[i];

    Eigen::Vector3d dir_vector = cell - camera_pos;
    double d = dir_vector.norm();
    if (d < 1e-6) continue;

    double proj_len = visual_max - d;
    if (proj_len < 0) continue;

    Eigen::Vector3d scaled_vector = (visual_max - d) * (dir_vector / d);

    if (sort_refer_.get_final_goal) {
      Vector3d cell2goal = sort_refer_.final_goal - cell;
      double cell2goal_norm = cell2goal.norm();
      if (cell2goal_norm > 1e-6) {
        proj_len = scaled_vector.dot(cell2goal) / cell2goal_norm;
      }
    }

    if (proj_len < 0) continue;
    explorability += proj_len;
  }
  return explorability;
}

void FrontierFinder::downsample(const vector<Vector3d>& cluster_in, vector<Vector3d>& cluster_out) {
  // downsamping cluster
  if (cluster_in.empty()) return;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloudf(new pcl::PointCloud<pcl::PointXYZ>);
  for (const auto& cell : cluster_in) cloud->points.emplace_back(cell[0], cell[1], cell[2]);

  const double leaf_size = edt_env_->sdf_map_->getResolution() * down_sample_;
  pcl::VoxelGrid<pcl::PointXYZ> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf_size, leaf_size, leaf_size);
  sor.filter(*cloudf);

  cluster_out.clear();
  for (const auto& pt : cloudf->points) cluster_out.emplace_back(pt.x, pt.y, pt.z);
}

inline vector<Eigen::Vector3i> FrontierFinder::sixNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(6);
  Eigen::Vector3i tmp;

  tmp = voxel - Eigen::Vector3i(1, 0, 0);
  neighbors[0] = tmp;
  tmp = voxel + Eigen::Vector3i(1, 0, 0);
  neighbors[1] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 1, 0);
  neighbors[2] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 1, 0);
  neighbors[3] = tmp;
  tmp = voxel - Eigen::Vector3i(0, 0, 1);
  neighbors[4] = tmp;
  tmp = voxel + Eigen::Vector3i(0, 0, 1);
  neighbors[5] = tmp;

  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::tenNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(10);
  Eigen::Vector3i tmp;
  int count = 0;

  for (int x = -1; x <= 1; ++x) {
    for (int y = -1; y <= 1; ++y) {
      if (x == 0 && y == 0) continue;
      tmp = voxel + Eigen::Vector3i(x, y, 0);
      neighbors[count++] = tmp;
    }
  }
  neighbors[count++] = tmp - Eigen::Vector3i(0, 0, 1);
  neighbors[count++] = tmp + Eigen::Vector3i(0, 0, 1);
  return neighbors;
}

inline vector<Eigen::Vector3i> FrontierFinder::allNeighbors(const Eigen::Vector3i& voxel) {
  vector<Eigen::Vector3i> neighbors(26);
  Eigen::Vector3i tmp;
  int count = 0;
  for (int x = -1; x <= 1; ++x)
    for (int y = -1; y <= 1; ++y)
      for (int z = -1; z <= 1; ++z) {
        if (x == 0 && y == 0 && z == 0) continue;
        tmp = voxel + Eigen::Vector3i(x, y, z);
        neighbors[count++] = tmp;
      }
  return neighbors;
}

inline bool FrontierFinder::isNeighborUnknown(const Eigen::Vector3i& voxel) {
  // At least one neighbor is unknown
  auto nbrs = sixNeighbors(voxel);
  for (const auto& nbr : nbrs) {
    if (edt_env_->sdf_map_->getOccupancy(nbr) == SDFMap::UNKNOWN) return true;
  }

  return false;
}

inline int FrontierFinder::toadr(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->toAddress(idx);
}

inline bool FrontierFinder::knownfree(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->getOccupancy(idx) == SDFMap::FREE;
}

inline bool FrontierFinder::inmap(const Eigen::Vector3i& idx) {
  return edt_env_->sdf_map_->isInMap(idx);
}

void FrontierFinder::buildFeatureGraph() {
  graph_search_->clear();

  // add start node
  vector<int> feature_id;
  feature_map_->getFeatureIDUsingPosYaw(sort_refer_.pos_now_, sort_refer_.yaw_now_, feature_id, raycaster_.get());
  graph_search_->addNode(sort_refer_.pos_now_, sort_refer_.yaw_now_, feature_id, NodeType::START);

  // add intermediate nodes sampled by features
  for (const auto& vp : feature_viewpoint)
    graph_search_->addNode(vp.pos_, vp.yaw_, vp.visual_features_ids_, NodeType::FEATURE_SAMPLE);

  // add edges
  graph_search_->addFeatureEdges();
}

void FrontierFinder::isViewpointPathFeasible(const Viewpoint& vp) {
  if (vp.search_path.size() != vp.search_yaw.size() || vp.search_path.size() < 2) {
    ROS_WARN("[isViewpointPathFeasible] This vp is unavailble");
    return;
  }

  vector<vector<int>> temp_feature_ids;
  temp_feature_ids.resize(vp.search_path.size());
  for (size_t i = 0; i < vp.search_path.size(); ++i)
    feature_map_->getFeatureIDUsingPosYaw(vp.search_path[i], vp.search_yaw[i], temp_feature_ids[i], raycaster_.get());
  std::cout << "....................Node Path...................." << std::endl;
  for (size_t i = 0; i < temp_feature_ids.size() - 1; ++i) {
    vector<int>& v1 = temp_feature_ids[i];
    vector<int>& v2 = temp_feature_ids[i + 1];
    std::unordered_set<int> set(v1.begin(), v1.end());
    int count = 0;
    for (const int& elem : v2) {
      if (set.find(elem) != set.end()) ++count;
    }
    std::cout << "Node " << i << ": "
              << "Position (" << vp.search_path[i].transpose() << "), Yaw: " << vp.search_yaw[i]
              << " Visible num: " << temp_feature_ids[i].size() << " Covisible num: " << count << endl;
  }

  size_t i = vp.search_path.size() - 1;
  std::cout << "Node " << i << ": "
            << "Position (" << vp.search_path[i].transpose() << "), Yaw: " << vp.search_yaw[i]
            << " Visible num: " << temp_feature_ids[i].size() << endl;

  const vector<int>& v_ori = vp.visual_features_ids_;
  const vector<int>& v0 = temp_feature_ids.back();
  std::unordered_set<int> set_ori(v_ori.begin(), v_ori.end());
  int count_ori = 0;
  for (const int& elem : v0) {
    if (set_ori.find(elem) != set_ori.end()) ++count_ori;
  }
  std::cout << "Node  view: "
            << "Position (" << vp.pos_.transpose() << "), Yaw: " << vp.yaw_ << " Visible num: " << v_ori.size()
            << " Covisible num: " << count_ori << endl;
  std::cout << "..............................................." << std::endl;
}

}  // namespace perception_aware_planner