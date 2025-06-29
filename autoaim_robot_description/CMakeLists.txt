cmake_minimum_required(VERSION 3.8)
project(autoaim_robot_description)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/robot_description_node.cpp
)

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN autoaim_robot_description::RobotDescriptionNode
  EXECUTABLE ${PROJECT_NAME}_node
)

ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)