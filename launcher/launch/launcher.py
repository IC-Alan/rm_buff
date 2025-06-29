import launch, time, os
# from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes, SetRemap
from launch_ros.descriptions import ComposableNode
from launch.event_handlers import (OnExecutionComplete, OnProcessExit, OnProcessIO, OnProcessStart, OnShutdown)
from launch.actions import RegisterEventHandler, LogInfo, OpaqueFunction, SetEnvironmentVariable
from ament_index_python.packages import get_package_share_directory

use_intra_process_comms = True

def launch_func(context, *args, **kwargs):
    camera_params_yaml = os.path.join(get_package_share_directory('autoaim_camera'), 'config', 'params.yaml')
    detection_params_yaml = os.path.join(get_package_share_directory('autoaim_detection'), 'config', 'params.yaml')
    prediction_params_yaml = os.path.join(get_package_share_directory('autoaim_prediction'), 'config', 'params.yaml')
    serial_params_yaml = os.path.join(get_package_share_directory('autoaim_serial_driver'), 'config', 'params.yaml')
    robot_description_params_yaml = os.path.join(get_package_share_directory('autoaim_robot_description'), 'config', 'params.yaml')
    recorder_params_yaml = os.path.join(get_package_share_directory('autoaim_recorder'), 'config', 'params.yaml')
    buff_params_yaml = os.path.join(get_package_share_directory('autoaim_buff'), 'config', 'params.yaml')


    composable_node_descriptions = [
        ComposableNode(
            package='autoaim_serial_driver',
            plugin='autoaim_serial_driver::InfantrySerialDriverNode',
            name='autoaim_serial_driver',
            parameters=[serial_params_yaml],
            extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
        ),
        
        ComposableNode(
            package='autoaim_robot_description',
            plugin='autoaim_robot_description::RobotDescriptionNode', 
            name='autoaim_robot_description',
            parameters=[robot_description_params_yaml],
            extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
        ),
        
        ComposableNode(
            package='autoaim_prediction',
            plugin='autoaim_prediction::PredictionNode',
            name='autoaim_prediction',
            parameters=[prediction_params_yaml],
            extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
        ),
        
        ComposableNode(
            package='autoaim_camera',
            plugin='autoaim_camera::CameraNode',
            name='autoaim_camera',
            parameters=[camera_params_yaml],
            extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
        ),
        
        ComposableNode(
            package='autoaim_detection',
            plugin='autoaim_detection::YoloDetectNode',
            name='autoaim_detection',
            parameters=[detection_params_yaml],
            extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
        ),
        ComposableNode(
            package='autoaim_recorder',
            plugin='autoaim_recorder::RecorderNode',
            name='autoaim_recorder',
            parameters=[recorder_params_yaml],
            extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
        ),

        ComposableNode(
            package='autoaim_buff',
            plugin='autoaim_buff::BuffNode',
            name='autoaim_buff',
            parameters=[buff_params_yaml],
            extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
        )
    ]
    time.sleep(1)
    return [
        LoadComposableNodes(
            composable_node_descriptions=composable_node_descriptions, 
            target_container='autoaim_container'
        )
    ]

def generate_launch_description():
    set_env_log_format = SetEnvironmentVariable(
        name='RCUTILS_CONSOLE_OUTPUT_FORMAT',
        value='[{time}] [{severity}] [{name}]: {message}'
    )
    
    container = ComposableNodeContainer(
        name='autoaim_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container_mt',
        output='both',
        emulate_tty=True,
        respawn=True,
        respawn_delay=3
    )
    
    event = RegisterEventHandler(
        OnProcessStart(
            target_action=container,
            on_start=[LogInfo(msg='autoaim container started'), OpaqueFunction(function=launch_func)]
        )
    )

    return launch.LaunchDescription([
        set_env_log_format,
        event,
        container
    ])


# import os
# from launch import LaunchDescription
# from launch_ros.actions import ComposableNodeContainer
# from launch_ros.descriptions import ComposableNode
# from launch_ros.substitutions import FindPackageShare
# from launch.actions import Shutdown, SetEnvironmentVariable, IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from ament_index_python.packages import get_package_share_directory

# use_intra_process_comms = True

# def generate_launch_description():
#     set_env_log_format = SetEnvironmentVariable(
#         name='RCUTILS_CONSOLE_OUTPUT_FORMAT',
#         value='[{severity}] [{name}]: {message}'
#     )

#     camera_params_yaml = os.path.join(get_package_share_directory('autoaim_camera'), 'config', 'params.yaml')
#     detection_params_yaml = os.path.join(get_package_share_directory('autoaim_detection'), 'config', 'params.yaml')
#     prediction_params_yaml = os.path.join(get_package_share_directory('autoaim_prediction'), 'config', 'params.yaml')
#     serial_params_yaml = os.path.join(get_package_share_directory('autoaim_serial_driver'), 'config', 'params.yaml')
#     robot_description_params_yaml = os.path.join(get_package_share_directory('autoaim_robot_description'), 'config', 'params.yaml')
#     recorder_params_yaml = os.path.join(get_package_share_directory('autoaim_recorder'), 'config', 'params.yaml')
#     buff_params_yaml = os.path.join(get_package_share_directory('autoaim_buff'), 'config', 'params.yaml')
#     #watchdog_params_yaml = os.path.join(get_package_share_directory('autoaim_watchdog'), 'config', 'params.yaml')

#     container = ComposableNodeContainer(
#         name='autoaim',
#         namespace='',
#         package='rclcpp_components',
#         executable='component_container_mt',
#         composable_node_descriptions=[
#             ComposableNode(
#                 package='autoaim_serial_driver',
#                 plugin='autoaim_serial_driver::InfantrySerialDriverNode',
#                 name='autoaim_serial_driver',
#                 parameters=[serial_params_yaml],
#                 extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
#             ),
            
#             ComposableNode(
#                 package='autoaim_robot_description',
#                 plugin='autoaim_robot_description::RobotDescriptionNode', 
#                 name='autoaim_robot_description',
#                 parameters=[robot_description_params_yaml],
#                 extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
#             ),
            
#             ComposableNode(
#                 package='autoaim_prediction',
#                 plugin='autoaim_prediction::PredictionNode',
#                 name='autoaim_prediction',
#                 parameters=[prediction_params_yaml],
#                 extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
#             ),
            
#             ComposableNode(
#                 package='autoaim_camera',
#                 plugin='autoaim_camera::CameraNode',
#                 name='autoaim_camera',
#                 parameters=[camera_params_yaml],
#                 extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
#             ),
            
#             ComposableNode(
#                 package='autoaim_detection',
#                 plugin='autoaim_detection::YoloDetectNode',
#                 name='autoaim_detection',
#                 parameters=[detection_params_yaml],
#                 extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
#             ),
#             ComposableNode(
#                 package='autoaim_recorder',
#                 plugin='autoaim_recorder::RecorderNode',
#                 name='autoaim_recorder',
#                 parameters=[recorder_params_yaml],
#                 extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
#             ),

#             ComposableNode(
#                 package='autoaim_buff',
#                 plugin='autoaim_buff::BuffNode',
#                 name='autoaim_buff',
#                 parameters=[buff_params_yaml],
#                 extra_arguments=[{'use_intra_process_comms': use_intra_process_comms}]
#             )

#         ],
        
#         output='both',
#         emulate_tty=True,
#         on_exit=Shutdown(),
#     )

#     return LaunchDescription([set_env_log_format, container])