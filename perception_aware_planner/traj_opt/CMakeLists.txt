cmake_minimum_required(VERSION 2.8.3)
project(traj_opt)

set(CMAKE_CXX_STANDARD 14)
find_package(NLopt REQUIRED)
set(NLopt_INCLUDE_DIRS ${NLOPT_INCLUDE_DIR})
 
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
 
find_package(Eigen3 REQUIRED)
find_package(PCL 1.7 REQUIRED)
 
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  visualization_msgs
  plan_env
  traj_utils
  utils
)
 
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES traj_opt
 CATKIN_DEPENDS plan_env traj_utils utils
#  DEPENDS system_lib
)
 
include_directories( 
    SYSTEM 
    include 
    ${catkin_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS} 
    ${PCL_INCLUDE_DIRS}
    ${NLOPT_INCLUDE_DIR}
)
 
add_library( traj_opt 
    src/bspline_optimizer.cpp 
    src/yaw_traj_opt.cpp 
    )
target_link_libraries( traj_opt
    ${catkin_LIBRARIES} 
    ${NLOPT_LIBRARIES}
    )  