cmake_minimum_required(VERSION 3.8)
project(autoaim_buff)

# 基础配置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 显式查找Ceres和Eigen3（Ceres依赖Eigen3）
find_package(Eigen3 REQUIRED)  # 必须添加
find_package(Ceres REQUIRED COMPONENTS SuiteSparse)  # 添加COMPONENTS参数6,11
find_package(cv_bridge REQUIRED)

# ROS2自动依赖管理
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# 添加目标库并链接依赖
ament_auto_add_library(${PROJECT_NAME} SHARED src/buff_node.cpp)

# 显式链接Ceres和Eigen3
target_link_libraries(
  ${PROJECT_NAME}
  ${CERES_LIBRARIES}  # 确保链接Ceres核心库
  Eigen3::Eigen       # 显式链接Eigen3
  cv_bridge::cv_bridge  # 添加cv_bridge链接
)

# 设置包含目录（替代旧的include_directories）
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  /opt/ros/humble/include/cv_bridge
)

# ROS2节点注册
rclcpp_components_register_node(
  ${PROJECT_NAME}
  PLUGIN "autoaim_buff::BuffNode"
  EXECUTABLE ${PROJECT_NAME}_node
)

# 测试配置
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

# 打包配置
ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
)