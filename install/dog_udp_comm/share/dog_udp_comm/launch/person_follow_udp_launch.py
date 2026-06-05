from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    dog_share = get_package_share_directory("dog_udp_comm")
    lidar_share = get_package_share_directory("cspc_lidar")

    follow_default_params = f"{dog_share}/params/person_follow_params.yaml"
    lidar_default_params = f"{lidar_share}/params/cspc_lidar.yaml"
    lidar_launch = f"{lidar_share}/launch/lidar_launch.py"

    follow_params_file = LaunchConfiguration("follow_params_file")

    return LaunchDescription([
        DeclareLaunchArgument("follow_params_file", default_value=follow_default_params),
        DeclareLaunchArgument("enable_lidar", default_value="true"),
        DeclareLaunchArgument("enable_latency_monitor", default_value="true"),
        DeclareLaunchArgument("lidar_params_file", default_value=lidar_default_params),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(lidar_launch),
            condition=IfCondition(LaunchConfiguration("enable_lidar")),
            launch_arguments={
                "params_file": LaunchConfiguration("lidar_params_file"),
            }.items(),
        ),
        Node(
            package="dog_udp_comm",
            executable="tracking_string_bridge.py",
            name="tracking_string_bridge",
            output="screen",
            parameters=[follow_params_file],
        ),
        Node(
            package="dog_udp_comm",
            executable="pixel_to_scan_polar.py",
            name="pixel_to_scan_polar",
            output="screen",
            parameters=[follow_params_file],
        ),
        Node(
            package="dog_udp_comm",
            executable="host_mpc_controller.py",
            name="host_mpc_controller",
            output="screen",
            parameters=[follow_params_file],
        ),
        Node(
            package="dog_udp_comm",
            executable="sender_node",
            name="udp_cmd_vel_server",
            output="screen",
            parameters=[follow_params_file],
        ),
        Node(
            package="dog_udp_comm",
            executable="latency_monitor.py",
            name="person_follow_latency_monitor",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_latency_monitor")),
            parameters=[follow_params_file],
        ),
    ])
