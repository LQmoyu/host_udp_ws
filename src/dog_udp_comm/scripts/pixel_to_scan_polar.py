#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Convert image pixel x coordinate to lidar polar measurement using single-line LaserScan.

Input:
- sensor_msgs/LaserScan on `scan_topic` (default: /scan)
- geometry_msgs/PointStamped on `pixel_topic` (default: /pixel_xy)
  - point.x: pixel u coordinate
  - point.y: pixel v coordinate (ignored for single-line lidar geometry)

Output:
- geometry_msgs/Vector3Stamped on `out_topic` (default: /person_polar)
  - vector.x: distance in meters
  - vector.y: angle in radians in lidar frame (left positive in ROS frame convention)
"""

import math
from threading import Lock

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


def wrap_to_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def wrap_to_2pi(angle):
    while angle >= 2.0 * math.pi:
        angle -= 2.0 * math.pi
    while angle < 0.0:
        angle += 2.0 * math.pi
    return angle


class PixelToScanPolarNode(Node):
    def __init__(self):
        super().__init__("pixel_to_scan_polar")

        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("pixel_topic", "/pixel_xy")
        self.declare_parameter("out_topic", "/person_polar")

        self.declare_parameter("fx", 600.0)
        self.declare_parameter("cx", 320.0)
        self.declare_parameter("yaw_cam_to_lidar", 0.0)

        self.declare_parameter("search_half_window", 6)
        self.declare_parameter("range_min_valid", 0.10)
        self.declare_parameter("range_max_valid", 10.0)
        self.declare_parameter("publish_debug", False)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.pixel_topic = str(self.get_parameter("pixel_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)

        self.fx = float(self.get_parameter("fx").value)
        self.cx = float(self.get_parameter("cx").value)
        self.yaw_cam_to_lidar = float(self.get_parameter("yaw_cam_to_lidar").value)

        self.search_half_window = int(self.get_parameter("search_half_window").value)
        self.range_min_valid = float(self.get_parameter("range_min_valid").value)
        self.range_max_valid = float(self.get_parameter("range_max_valid").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)

        self.scan_lock = Lock()
        self.latest_scan = None

        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_subscription(PointStamped, self.pixel_topic, self.pixel_cb, 10)
        self.polar_pub = self.create_publisher(Vector3Stamped, self.out_topic, 10)

        self.get_logger().info(
            "Pixel->Scan polar bridge ready. "
            f"scan_topic={self.scan_topic} pixel_topic={self.pixel_topic} out_topic={self.out_topic} "
            f"fx={self.fx:.3f} cx={self.cx:.3f} yaw_cam_to_lidar={self.yaw_cam_to_lidar:.3f}"
        )

    def scan_cb(self, msg: LaserScan):
        with self.scan_lock:
            self.latest_scan = msg

    def _is_valid_range(self, r: float) -> bool:
        if not np.isfinite(r):
            return False
        if r <= self.range_min_valid:
            return False
        if r >= self.range_max_valid:
            return False
        return True

    def _find_nearest_valid(self, ranges, center_idx: int):
        n = len(ranges)
        best_idx = -1
        best_range = 0.0
        best_dist = 1e9

        left = max(0, center_idx - self.search_half_window)
        right = min(n - 1, center_idx + self.search_half_window)
        for i in range(left, right + 1):
            r = float(ranges[i])
            if not self._is_valid_range(r):
                continue
            d = abs(i - center_idx)
            if d < best_dist:
                best_dist = d
                best_idx = i
                best_range = r

        return best_idx, best_range

    def pixel_cb(self, msg: PointStamped):
        with self.scan_lock:
            scan = self.latest_scan

        if scan is None or len(scan.ranges) == 0:
            return

        u = float(msg.point.x)

        # Horizontal pinhole projection: pixel x -> camera yaw angle.
        theta_cam = math.atan2((u - self.cx), max(self.fx, 1e-6))

        # Convert to lidar frame angle.
        theta_lidar = theta_cam + self.yaw_cam_to_lidar

        # Driver currently provides scan in [0, 2pi].
        theta_lidar = wrap_to_2pi(theta_lidar)

        if scan.angle_increment <= 0.0:
            return

        idx = int(round((theta_lidar - scan.angle_min) / scan.angle_increment))
        idx = max(0, min(len(scan.ranges) - 1, idx))

        best_idx, best_range = self._find_nearest_valid(scan.ranges, idx)
        if best_idx < 0:
            return

        angle = scan.angle_min + best_idx * scan.angle_increment

        out = Vector3Stamped()
        out.header = msg.header
        out.vector.x = float(best_range)
        out.vector.y = float(wrap_to_pi(angle))
        out.vector.z = 0.0
        self.polar_pub.publish(out)

        if self.publish_debug:
            self.get_logger().info(
                f"u={u:.1f}, idx={idx}, use_idx={best_idx}, "
                f"range={best_range:.3f}m, angle={out.vector.y:.3f}rad"
            )


def main():
    rclpy.init()
    node = PixelToScanPolarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
