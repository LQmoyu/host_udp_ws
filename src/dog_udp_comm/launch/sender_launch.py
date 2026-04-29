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
                {'local_port': 8888}
            ]
        )
    ])
