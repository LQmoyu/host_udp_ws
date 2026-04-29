#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
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
        self.pub = self.create_publisher(Twist, '/track_cmd_vel', 10)
        
        self.declare_parameter("linear_speed", 0.3)
        self.declare_parameter("angular_speed", 0.5)
        self.linear_speed = self.get_parameter("linear_speed").value
        self.angular_speed = self.get_parameter("angular_speed").value

        self.cmd_vx = 0.0
        self.cmd_wz = 0.0
        self.last_key_time = self.get_clock().now().nanoseconds / 1e9
        
        # ROS 2 使用 timer 替代 ROS 1 的 Rate 循环
        self.timer = self.create_timer(0.05, self.timer_callback) # 20Hz
        self.settings = termios.tcgetattr(sys.stdin)
        self.get_logger().info(msg)

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

        twist = Twist()
        twist.linear.x = float(self.cmd_vx)
        twist.angular.z = float(self.cmd_wz)
        self.pub.publish(twist)

    def stop_robot(self):
        twist = Twist()
        self.pub.publish(twist)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)

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
