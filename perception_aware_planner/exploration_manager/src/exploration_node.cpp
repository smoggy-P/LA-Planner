#include "exploration_manager/perception_aware_exploration_fsm.h"

#include "local_plan_manager/backward.hpp"

#include <ros/ros.h>

namespace backward {
backward::SignalHandling sh;
}

using namespace perception_aware_planner;

int main(int argc, char** argv) {
  ros::init(argc, argv, "exploration_node");
  ros::NodeHandle nh("~");

  auto expl_fsm = std::make_shared<PAExplorationFSM>();
  expl_fsm->init(nh);

  ros::Duration(1.0).sleep();
  ros::spin();

  return 0;
}
