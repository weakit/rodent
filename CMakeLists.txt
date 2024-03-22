cmake_minimum_required(VERSION 3.0.2)
project(rodent_odometry)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs 
  sensor_msgs
)

catkin_python_setup()

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs sensor_msgs message_runtime 
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
