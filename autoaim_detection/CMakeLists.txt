cmake_minimum_required(VERSION 3.18)
project(autoaim_detection
  LANGUAGES CXX CUDA  # 添加CUDA语言支持
  VERSION 1.0.0
)

# 基础编译设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")
add_compile_options(-Ofast -Wall -Wextra)

### 推理后端配置 ###
option(INFER_BACKEND_OPENVINO "Enable OpenVINO backend" OFF)
option(INFER_BACKEND_CUDA "Enable CUDA backend" ON)

# ROS2包配置
find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

ament_auto_find_build_dependencies()

# 源码构建配置
ament_auto_add_library(${PROJECT_NAME} SHARED
  src/detection_node.cpp
)

# 公共头文件目录（所有用户可见的头文件）
target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${OpenCV_INCLUDE_DIRS}
)

### 后端特定配置 ###
if(INFER_BACKEND_OPENVINO)
  # OpenVINO配置
  add_definitions(-DINFER_BACKEND_OPENVINO)
  find_package(OpenVINO COMPONENTS Runtime REQUIRED)

  # OpenVINO专用头文件（实现细节）
  target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/openvino_infer
  )

  target_link_libraries(${PROJECT_NAME} 
    openvino::runtime
    ${OpenMP_CXX_LIBRARIES}
  )

elseif(INFER_BACKEND_CUDA)
  # CUDA配置
  add_definitions(-DINFER_BACKEND_CUDA)

  target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_infer
    ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_infer/impl
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  )

  # CUDA运行时配置
  find_package(CUDA REQUIRED)
  find_package(CUDAToolkit REQUIRED)

  # 设置CUDA编译参数
  set(CMAKE_CUDA_ARCHITECTURES "native")  
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall")

  cuda_add_library(preprocess SHARED ${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_infer/impl/preprocess.cu)
  target_link_libraries(preprocess ${OpenCV_LIBS} ${CUDA_LIBS})
  set_target_properties(preprocess PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # 链接CUDA相关库
  target_link_libraries(${PROJECT_NAME} 
    ${CUDART_LIBRARY}
    nvinfer
    nvonnxparser
    cudart
    preprocess
  )

  # 确保CUDA编译目标可见
  set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  ament_target_dependencies(${PROJECT_NAME} rclcpp sensor_msgs CUDA)

else()
  message(FATAL_ERROR "Must select at least one inference backend")
endif()

# 公共链接库
target_link_libraries(${PROJECT_NAME} 
  ${OpenCV_LIBS}
  OpenMP::OpenMP_CXX
)

# ROS2组件注册
# rclcpp_components_register_node(${PROJECT_NAME}
#   PLUGIN autoaim_detection::YoloDetectNode
#   EXECUTABLE ${PROJECT_NAME}_node
# )

# 测试配置
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

rclcpp_components_register_node(${PROJECT_NAME}  #生成node
  PLUGIN autoaim_detection::YoloDetectNode
  EXECUTABLE ${PROJECT_NAME}_node
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 安装配置
ament_auto_package(
  INSTALL_TO_SHARE
  config
  launch
  model
  include  # 安装公共头文件
  # $<$<BOOL:${INFER_BACKEND_CUDA}>:include/cuda_infer>  # 条件安装CUDA头文件
  # $<$<BOOL:${INFER_BACKEND_OPENVINO}>:include/openvino_infer>  # 条件安装OpenVINO头文件
)

# 条件安装CUDA头文件
if(INFER_BACKEND_CUDA)
  install(DIRECTORY include/cuda_infer
          DESTINATION include)
  install(TARGETS preprocess
          DESTINATION lib  # 将库安装到 lib/autoaim_detection 目录
  )
endif()

# 条件安装OpenVINO头文件
if(INFER_BACKEND_OPENVINO)
  install(DIRECTORY include/openvino_infer
          DESTINATION include)
endif()
