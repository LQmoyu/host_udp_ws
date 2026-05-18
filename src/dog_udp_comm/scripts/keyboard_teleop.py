#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped
import sys, select, termios, tty

msg = """
==============================
ROS 2 Python Keyboard Teleop
Hold 'W' to move Forward
Hold 'S' to move Backward
Hold 'A' to turn Left
Hold 'D' to turn Right
Release keys to STOP (Safety Mode)
Press 'Q' or Ctrl+C to Quit
==============================
"""

def getKey(settings, timeout):
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

class KeyboardTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_teleop')
        self.declare_parameter("cmd_topic", "/track_cmd_vel")
        self.declare_parameter("use_stamped_msg", False)
        self.declare_parameter("linear_speed", 0.3)
        self.declare_parameter("angular_speed", 0.5)

        self.cmd_topic = self.get_parameter("cmd_topic").value
        self.use_stamped_msg = self.get_parameter("use_stamped_msg").value
        self.linear_speed = self.get_parameter("linear_speed").value
        self.angular_speed = self.get_parameter("angular_speed").value

        self.pub_twist = None
        self.pub_twist_stamped = None
        if self.use_stamped_msg:
            self.pub_twist_stamped = self.create_publisher(TwistStamped, self.cmd_topic, 10)
        else:
            self.pub_twist = self.create_publisher(Twist, self.cmd_topic, 10)

        self.cmd_vx = 0.0
        self.cmd_wz = 0.0
        self.last_key_time = self.get_clock().now().nanoseconds / 1e9
        
        # ROS 2 使用 timer 替代 ROS 1 的 Rate 循环
        self.timer = self.create_timer(0.05, self.timer_callback) # 20Hz
        self.settings = termios.tcgetattr(sys.stdin)
        self.get_logger().info(msg)
        self.get_logger().info(
            f"Publish topic: {self.cmd_topic}, "
            f"message type: {'geometry_msgs/msg/TwistStamped' if self.use_stamped_msg else 'geometry_msgs/msg/Twist'}"
        )

    def timer_callback(self):
        key = getKey(self.settings, 0.02)
        current_time = self.get_clock().now().nanoseconds / 1e9

        if key:
            self.last_key_time = current_time
            if key in ['w', 'W']:
                self.cmd_vx = self.linear_speed; self.cmd_wz = 0.0
            elif key in ['s', 'S']:
                self.cmd_vx = -self.linear_speed; self.cmd_wz = 0.0
            elif key in ['a', 'A']:
                self.cmd_vx = 0.0; self.cmd_wz = self.angular_speed
            elif key in ['d', 'D']:
                self.cmd_vx = 0.0; self.cmd_wz = -self.angular_speed
            elif key in ['q', 'Q', '\x03']:
                self.stop_robot()
                sys.exit(0)
        
        # 看门狗：0.3 秒无按键则归零
        if (current_time - self.last_key_time) > 0.3:
            self.cmd_vx = 0.0
            self.cmd_wz = 0.0

        self.publish_cmd(float(self.cmd_vx), float(self.cmd_wz))

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

    def publish_cmd(self, linear_x, angular_z):
        if self.use_stamped_msg:
            msg_stamped = TwistStamped()
            msg_stamped.header.stamp = self.get_clock().now().to_msg()
            msg_stamped.twist.linear.x = linear_x
            msg_stamped.twist.angular.z = angular_z
            self.pub_twist_stamped.publish(msg_stamped)
        else:
            msg_twist = Twist()
            msg_twist.linear.x = linear_x
            msg_twist.angular.z = angular_z
            self.pub_twist.publish(msg_twist)

def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
