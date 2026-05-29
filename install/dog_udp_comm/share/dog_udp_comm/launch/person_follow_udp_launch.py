from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    dog_share = get_package_share_directory("dog_udp_comm")
    follow_default_params = f"{dog_share}/params/person_follow_params.yaml"
    lidar_share = get_package_share_directory("cspc_lidar")
    lidar_launch = f"{lidar_share}/launch/lidar_launch.py"
    lidar_default_params = f"{lidar_share}/params/cspc_lidar.yaml"

    args = [
        DeclareLaunchArgument("follow_params_file", default_value=follow_default_params),
        DeclareLaunchArgument("enable_lidar", default_value="true"),
        DeclareLaunchArgument("lidar_params_file", default_value=lidar_default_params),
        DeclareLaunchArgument("tracking_topic", default_value="/tracking"),
        DeclareLaunchArgument("state_topic", default_value="/tracking_state"),
        DeclareLaunchArgument("pixel_topic", default_value="/tracking_pixel"),
        DeclareLaunchArgument("person_topic", default_value="/person_polar"),
        DeclareLaunchArgument("cmd_topic", default_value="/track_cmd_vel"),
        DeclareLaunchArgument("scan_topic", default_value="/scan"),
        DeclareLaunchArgument("image_width", default_value="640.0"),
        DeclareLaunchArgument("image_height", default_value="384.0"),
        DeclareLaunchArgument("fx", default_value="600.0"),
        DeclareLaunchArgument("cx", default_value="320.0"),
        DeclareLaunchArgument("yaw_cam_to_lidar", default_value="0.0"),
        DeclareLaunchArgument("local_ip", default_value="0.0.0.0"),
        DeclareLaunchArgument("local_port", default_value="8888"),
        DeclareLaunchArgument("use_fixed_receiver", default_value="false"),
        DeclareLaunchArgument("remote_ip", default_value="192.168.19.90"),
        DeclareLaunchArgument("remote_port", default_value="8890"),
        DeclareLaunchArgument("desired_distance", default_value="1.2"),
        DeclareLaunchArgument("desired_angle", default_value="0.0"),
        DeclareLaunchArgument("max_v", default_value="0.6"),
        DeclareLaunchArgument("max_w", default_value="2.2"),
        DeclareLaunchArgument("force_zero_linear_velocity", default_value="false"),
        DeclareLaunchArgument("reverse_angular_output", default_value="true"),
    ]

    lidar_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(lidar_launch),
        condition=IfCondition(LaunchConfiguration("enable_lidar")),
        launch_arguments={
            "params_file": LaunchConfiguration("lidar_params_file"),
        }.items(),
    )

    tracking_bridge_node = Node(
        package="dog_udp_comm",
        executable="tracking_string_bridge.py",
        name="tracking_string_bridge",
        output="screen",
        parameters=[
            LaunchConfiguration("follow_params_file"),
            {"tracking_topic": LaunchConfiguration("tracking_topic")},
            {"state_topic": LaunchConfiguration("state_topic")},
            {"pixel_topic": LaunchConfiguration("pixel_topic")},
            {"image_width": ParameterValue(LaunchConfiguration("image_width"), value_type=float)},
            {"image_height": ParameterValue(LaunchConfiguration("image_height"), value_type=float)},
            {"max_allowed_age_ms": 1000.0},
            {"max_consecutive_drop": 3},
            {"warn_interval_sec": 1.0},
            {"use_source_timestamp_for_header": True},
            {"relative_timestamp_threshold_ms": 1000000000.0},
            {"topic_check_period_sec": 0.5},
            {"tracking_topic_lost_timeout_sec": 1.0},
        ],
    )

    pixel_to_scan_node = Node(
        package="dog_udp_comm",
        executable="pixel_to_scan_polar.py",
        name="pixel_to_scan_polar",
        output="screen",
        parameters=[
            LaunchConfiguration("follow_params_file"),
            {"scan_topic": LaunchConfiguration("scan_topic")},
            {"pixel_topic": LaunchConfiguration("pixel_topic")},
            {"out_topic": LaunchConfiguration("person_topic")},
            {"fx": ParameterValue(LaunchConfiguration("fx"), value_type=float)},
            {"cx": ParameterValue(LaunchConfiguration("cx"), value_type=float)},
            {"yaw_cam_to_lidar": ParameterValue(LaunchConfiguration("yaw_cam_to_lidar"), value_type=float)},
            {"search_half_window": 6},
            {"range_min_valid": 0.10},
            {"range_max_valid": 10.0},
            {"prefer_scan_angle_output": False},
            {"publish_debug": False},
        ],
    )

    sender_node = Node(
        package="dog_udp_comm",
        executable="sender_node",
        name="udp_cmd_vel_server",
        output="screen",
        parameters=[
            LaunchConfiguration("follow_params_file"),
            {"local_ip": LaunchConfiguration("local_ip")},
            {"local_port": ParameterValue(LaunchConfiguration("local_port"), value_type=int)},
            {"use_fixed_receiver": ParameterValue(LaunchConfiguration("use_fixed_receiver"), value_type=bool)},
            {"remote_ip": LaunchConfiguration("remote_ip")},
            {"remote_port": ParameterValue(LaunchConfiguration("remote_port"), value_type=int)},
        ],
    )

    mpc_controller_node = Node(
        package="dog_udp_comm",
        executable="host_mpc_controller.py",
        name="host_mpc_controller",
        output="screen",
        parameters=[
            LaunchConfiguration("follow_params_file"),
            {"person_topic": LaunchConfiguration("person_topic")},
            {"tracking_state_topic": LaunchConfiguration("state_topic")},
            {"cmd_topic": LaunchConfiguration("cmd_topic")},
            {"control_hz": 50.0},
            {"dt": 0.0},
            {"horizon": 12},
            {"msg_timeout": 1.0},
            {"tracking_state_timeout_sec": 1.0},
            {"require_first_tracking_frame": True},
            {"desired_distance": ParameterValue(LaunchConfiguration("desired_distance"), value_type=float)},
            {"desired_angle": ParameterValue(LaunchConfiguration("desired_angle"), value_type=float)},
            {"distance_tolerance": 0.08},
            {"angle_tolerance": 0.08},
            {"stop_when_aligned": True},
            {"stop_on_lost_target": True},
            {"allow_reverse": True},
            {"target_lost_duration_sec": 1.0},
            {"max_v": ParameterValue(LaunchConfiguration("max_v"), value_type=float)},
            {"max_w": ParameterValue(LaunchConfiguration("max_w"), value_type=float)},
            {"max_reverse_v": 0.15},
            {"force_zero_linear_velocity": ParameterValue(LaunchConfiguration("force_zero_linear_velocity"), value_type=bool)},
            {"reverse_angular_output": ParameterValue(LaunchConfiguration("reverse_angular_output"), value_type=bool)},
            {"q_dist": 16.0},
            {"q_angle": 15.0},
            {"r_v": 0.75},
            {"r_w": 0.10},
            {"qf_scale": 4.0},
            {"kff_dist": 0.45},
            {"kff_angle": 2.0},
            {"use_tracking_x_for_angle": True},
            {"tracking_x_target": 0.5},
            {"tracking_x_deadband": 0.003},
            {"tracking_x_timeout_sec": 1.0},
            {"enable_angle_prediction": True},
            {"max_prediction_sec": 0.35},
            {"angle_rate_filter_alpha": 0.6},
            {"max_angle_rate": 3.5},
            {"enable_tracking_only_fallback": True},
            {"tracking_only_linear_speed": 0.1},
            {"tracking_only_kp_w": 3.0},
            {"tracking_only_kd_w": 0.4},
            {"image_width": ParameterValue(LaunchConfiguration("image_width"), value_type=float)},
            {"fx": ParameterValue(LaunchConfiguration("fx"), value_type=float)},
            {"cx": ParameterValue(LaunchConfiguration("cx"), value_type=float)},
            {"yaw_cam_to_control": ParameterValue(LaunchConfiguration("yaw_cam_to_lidar"), value_type=float)},
            {"tracking_angle_gain": 2.5},
            {"distance_filter_alpha": 0.65},
            {"angle_filter_alpha": 0.65},
            {"min_valid_distance": 0.15},
            {"max_valid_distance": 8.0},
            {"enable_distance_speed_profile": True},
            {"reverse_distance_threshold": 0.5},
            {"stop_reverse_distance": 0.8},
            {"speed_profile_near_distance": 1.3},
            {"speed_profile_far_distance": 3.2},
            {"speed_profile_min_v": 0.04},
            {"speed_profile_max_v": ParameterValue(LaunchConfiguration("max_v"), value_type=float)},
            {"angle_priority_threshold": 0.12},
            {"min_heading_speed_scale": 0.05},
            {"max_accel": 1.0},
            {"max_w_accel": 10.0},
        ],
    )

    return LaunchDescription(
        args
        + [
            lidar_node,
            tracking_bridge_node,
            pixel_to_scan_node,
            mpc_controller_node,
            sender_node,
        ]
    )
