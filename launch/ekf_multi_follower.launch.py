#!/usr/bin/env python3
"""
Launch file for acoustic EKF with configurable follower namespace.
Usage: ros2 launch acoustic_ekf_pkg ekf_multi_follower.launch.py follower_ns:=f1
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import yaml


def generate_launch_description():
    # Declare launch argument for follower namespace
    follower_ns_arg = DeclareLaunchArgument(
        'follower_ns',
        default_value='f1',
        description='Namespace for the follower (e.g., f1, f2, f3)'
    )
    
    # Path to config file
    config_file = os.path.join(
        get_package_share_directory('acoustic_ekf_pkg'),
        'config',
        'ekf_config_multi_follower.yaml'
    )
    
    def launch_setup(context, *args, **kwargs):
        # Get the follower namespace from launch argument
        follower_ns = LaunchConfiguration('follower_ns').perform(context)
        
        # Load config file and substitute namespace
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        # Replace placeholder with actual namespace
        config_content = config_content.replace('${follower_ns}', follower_ns)
        
        # Parse the substituted config
        config_dict = yaml.safe_load(config_content)
        
        # Create the node with substituted parameters
        ekf_node = Node(
            package='acoustic_ekf_pkg',
            executable='ekf_node',
            name=f'acoustic_ekf_{follower_ns}',
            output='screen',
            parameters=[config_dict],
            remappings=[
                # Explicit remappings for clarity (already in config, but good practice)
                ('/follower/ekf/state', config_dict['publish_topic']),
                ('/follower/ekf/geopoint', config_dict['geopoint_topic']),
                ('/follower/ekf/delta_pos', config_dict['delta_pos_topic']),
            ]
        )
        
        return [ekf_node]
    
    return LaunchDescription([
        follower_ns_arg,
        OpaqueFunction(function=launch_setup)
    ])
