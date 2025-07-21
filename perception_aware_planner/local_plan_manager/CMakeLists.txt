cmake_minimum_required(VERSION 2.8.3)
project(local_plan_manager)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  quadrotor_msgs
  plan_env
  path_searching
  traj_utils
  traj_opt
  active_perception
)


catkin_package(
 INCLUDE_DIRS include
 LIBRARIES local_plan_manager
 CATKIN_DEPENDS plan_env path_searching active_perception traj_utils traj_opt)

include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_library(local_plan_manager 
  src/perception_aware_planner_manager.cpp
  src/yaw_initial_planner.cpp
  )
target_link_libraries(local_plan_manager ${catkin_LIBRARIES})

add_executable(traj_server src/traj_server.cpp)
target_link_libraries(traj_server ${catkin_LIBRARIES} -ldw)