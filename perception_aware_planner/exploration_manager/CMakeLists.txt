cmake_minimum_required(VERSION 2.8.3)
project(exploration_manager)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
set(CMAKE_BUILD_TYPE Debug)


find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  geometry_msgs
  quadrotor_msgs
  active_perception
  local_plan_manager
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES exploration_manager
 CATKIN_DEPENDS active_perception local_plan_manager
#  DEPENDS system_lib
)

include_directories(
  include
  SYSTEM
  ${catkin_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(exploration_node
  src/exploration_node.cpp 
  src/perception_aware_exploration_fsm.cpp
  src/perception_aware_exploration_manager.cpp
  src/failure_detector.cpp
  )
target_link_libraries(exploration_node 
  ${catkin_LIBRARIES}
  # ${PCL_LIBRARIES}
  # -ldw
  )