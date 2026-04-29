from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dog_udp_comm',
            executable='sender_node',
            name='udp_cmd_vel_server',
            output='screen',
            parameters=[
                {'local_ip': '0.0.0.0'},
                {'local_port': 8888},
            ],
        ),
        Node(
            package='dog_udp_comm',
            executable='host_mpc_controller.py',
            name='host_mpc_controller',
            output='screen',
            parameters=[
                {'person_topic': '/person_polar'},
                {'cmd_topic': '/track_cmd_vel'},
                {'control_hz': 30.0},
                {'dt': 0.0},
                {'horizon': 12},
                {'msg_timeout': 0.4},
                {'desired_distance': 1.2},
                {'distance_tolerance': 0.08},
                {'angle_tolerance': 0.08},
                {'stop_when_aligned': True},
                {'stop_on_lost_target': True},
                {'allow_reverse': False},
                {'max_v': 0.8},
                {'max_w': 1.5},
                {'max_reverse_v': 0.2},
                {'q_dist': 16.0},
                {'q_angle': 10.0},
                {'r_v': 0.35},
                {'r_w': 0.25},
                {'qf_scale': 4.0},
                {'kff_dist': 0.9},
                {'kff_angle': 1.2},
                {'distance_filter_alpha': 0.65},
                {'angle_filter_alpha': 0.65},
                {'min_valid_distance': 0.15},
                {'max_valid_distance': 8.0},
            ],
        ),
    ])
