"""Launch acoustic UKF for an OWTT stick follower (live, wall clock).

Example:
  ros2 launch acoustic_ekf_pkg ukf_stick.launch.py robot_name:=stick_3
  ros2 launch acoustic_ekf_pkg ukf_stick.launch.py robot_name:=stick_3 setpoint_modem_ids:=001,002
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    robot = LaunchConfiguration('robot_name').perform(context)
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() in (
        '1', 'true', 'yes')
    setpoint_modem_ids = LaunchConfiguration('setpoint_modem_ids').perform(context)
    config_file = f'ukf_{robot}.yaml'
    return [
        Node(
            package='acoustic_ekf_pkg',
            executable='ukf_node',
            name=f'{robot}_acoustic_ukf',
            output='screen',
            parameters=[{
                'config_file': config_file,
                'use_sim_time': use_sim_time,
                'setpoint_modem_ids': setpoint_modem_ids,
            }],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_name', default_value='stick_3',
            description='Follower stick whose namespaced OWTT topics to fuse '
                        '(stick_3 or stick_4; needs matching ukf_<name>.yaml)'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='false',
            description='false for live pier bringup; true only for bag replay'),
        DeclareLaunchArgument(
            'setpoint_modem_ids', default_value='',
            description='Two modem ids for the setpoint diameter, e.g. 001,002. '
                        'Empty uses the two fastest leaders.'),
        OpaqueFunction(function=_launch_setup),
    ])
