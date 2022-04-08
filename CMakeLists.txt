cmake_minimum_required(VERSION 2.8.3)
project(pose2movement)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs)

catkin_python_setup()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
