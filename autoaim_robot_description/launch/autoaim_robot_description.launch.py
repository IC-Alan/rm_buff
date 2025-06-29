import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    config_yaml = os.path.join( get_package_share_directory('autoaim_robot_description'), 'config', 'params.yaml')
    
    return LaunchDescription([
        Node(
            package='autoaim_robot_description',
            executable='autoaim_robot_description_node',
            output='screen',
            parameters=[config_yaml]
        )
    ])