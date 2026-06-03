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
  - vector.y: angle in radians used by controller (left positive)
"""

import math
from collections import deque
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
        self.declare_parameter("prefer_scan_angle_output", False)
        self.declare_parameter("publish_debug", False)
        self.declare_parameter("sync_policy", "latest")
        self.declare_parameter("max_sync_diff_sec", 0.08)
        self.declare_parameter("warn_sync_diff_sec", 0.05)
        self.declare_parameter("scan_buffer_sec", 1.0)
        self.declare_parameter("stamp_mismatch_fallback_sec", 10.0)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.pixel_topic = str(self.get_parameter("pixel_topic").value)
        self.out_topic = str(self.get_parameter("out_topic").value)

        self.fx = float(self.get_parameter("fx").value)
        self.cx = float(self.get_parameter("cx").value)
        self.yaw_cam_to_lidar = float(self.get_parameter("yaw_cam_to_lidar").value)

        self.search_half_window = int(self.get_parameter("search_half_window").value)
        self.range_min_valid = float(self.get_parameter("range_min_valid").value)
        self.range_max_valid = float(self.get_parameter("range_max_valid").value)
        self.prefer_scan_angle_output = bool(self.get_parameter("prefer_scan_angle_output").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.sync_policy = str(self.get_parameter("sync_policy").value).strip().lower()
        self.max_sync_diff_sec = max(0.0, float(self.get_parameter("max_sync_diff_sec").value))
        self.warn_sync_diff_sec = max(0.0, float(self.get_parameter("warn_sync_diff_sec").value))
        self.scan_buffer_sec = max(0.05, float(self.get_parameter("scan_buffer_sec").value))
        self.stamp_mismatch_fallback_sec = max(
            0.0, float(self.get_parameter("stamp_mismatch_fallback_sec").value)
        )
        if self.sync_policy not in ("latest", "strict"):
            self.get_logger().warn(
                f"Unknown sync_policy='{self.sync_policy}', fallback to 'latest'."
            )
            self.sync_policy = "latest"

        self.scan_lock = Lock()
        self.scan_buffer = deque()
        self.last_warn_ns = 0

        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_subscription(PointStamped, self.pixel_topic, self.pixel_cb, 10)
        self.polar_pub = self.create_publisher(Vector3Stamped, self.out_topic, 10)

        self.get_logger().info(
            "Pixel->Scan polar bridge ready. "
            f"scan_topic={self.scan_topic} pixel_topic={self.pixel_topic} out_topic={self.out_topic} "
            f"fx={self.fx:.3f} cx={self.cx:.3f} yaw_cam_to_lidar={self.yaw_cam_to_lidar:.3f} "
            f"sync_policy={self.sync_policy} max_sync_diff={self.max_sync_diff_sec:.3f}s "
            f"scan_buffer={self.scan_buffer_sec:.3f}s"
        )

    def scan_cb(self, msg: LaserScan):
        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = self._message_stamp_ns(msg.header.stamp, now_ns)
        keep_after_ns = now_ns - int(self.scan_buffer_sec * 1e9)
        with self.scan_lock:
            self.scan_buffer.append((stamp_ns, now_ns, msg))
            while self.scan_buffer and self.scan_buffer[0][1] < keep_after_ns:
                self.scan_buffer.popleft()

    def warn_throttle(self, text: str, period_sec: float = 1.0):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_warn_ns >= int(period_sec * 1e9):
            self.get_logger().warn(text)
            self.last_warn_ns = now_ns

    def _stamp_to_ns(self, stamp):
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _message_stamp_ns(self, stamp, receive_ns: int) -> int:
        stamp_ns = self._stamp_to_ns(stamp)
        if stamp_ns <= 0:
            return receive_ns

        if self.stamp_mismatch_fallback_sec <= 0.0:
            return stamp_ns

        now_ns = self.get_clock().now().nanoseconds
        mismatch_sec = abs(stamp_ns - now_ns) * 1e-9
        if mismatch_sec > self.stamp_mismatch_fallback_sec:
            return receive_ns
        return stamp_ns

    def _select_scan(self, pixel_stamp_ns: int):
        with self.scan_lock:
            if not self.scan_buffer:
                return None, 0.0
            stamp_ns, _, scan = min(
                self.scan_buffer,
                key=lambda item: abs(item[0] - pixel_stamp_ns),
            )

        sync_error_sec = abs(stamp_ns - pixel_stamp_ns) * 1e-9
        return scan, sync_error_sec

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
        now_ns = self.get_clock().now().nanoseconds
        pixel_stamp_ns = self._message_stamp_ns(msg.header.stamp, now_ns)
        scan, sync_error_sec = self._select_scan(pixel_stamp_ns)

        if scan is None:
            self.warn_throttle("Waiting for first /scan frame before pixel-to-scan matching.")
            return
        if len(scan.ranges) == 0:
            return
        if sync_error_sec > self.max_sync_diff_sec:
            text = (
                f"Camera/Lidar sync diff {sync_error_sec:.3f}s exceeds "
                f"max_sync_diff_sec={self.max_sync_diff_sec:.3f}s."
            )
            if self.sync_policy == "strict":
                self.warn_throttle(text + " Drop pixel frame because sync_policy=strict.")
                return
            self.warn_throttle(text + " Keep nearest scan because sync_policy=latest.")
        elif sync_error_sec > self.warn_sync_diff_sec:
            self.warn_throttle(
                f"Camera/Lidar sync diff {sync_error_sec:.3f}s exceeds "
                f"warn_sync_diff_sec={self.warn_sync_diff_sec:.3f}s."
            )

        u = float(msg.point.x)

        # Horizontal pinhole projection: pixel x -> camera yaw angle.
        theta_cam = math.atan2((u - self.cx), max(self.fx, 1e-6))

        # Convert to lidar frame angle. This directly represents the image-centering
        # direction (x=0.5 -> angle~=0 with correct camera intrinsics/extrinsics).
        theta_lidar = theta_cam + self.yaw_cam_to_lidar
        theta_control = wrap_to_pi(theta_lidar)

        # Driver currently provides scan in [0, 2pi].
        theta_lidar = wrap_to_2pi(theta_lidar)

        if scan.angle_increment <= 0.0:
            return

        idx = int(round((theta_lidar - scan.angle_min) / scan.angle_increment))
        idx = max(0, min(len(scan.ranges) - 1, idx))

        best_idx, best_range = self._find_nearest_valid(scan.ranges, idx)
        if best_idx < 0:
            return

        angle_scan = scan.angle_min + best_idx * scan.angle_increment
        if self.prefer_scan_angle_output:
            angle_out = wrap_to_pi(angle_scan)
        else:
            angle_out = theta_control

        out = Vector3Stamped()
        out.header = msg.header
        out.vector.x = float(best_range)
        out.vector.y = float(angle_out)
        out.vector.z = float(sync_error_sec)
        self.polar_pub.publish(out)

        if self.publish_debug:
            self.get_logger().info(
                f"u={u:.1f}, idx={idx}, use_idx={best_idx}, "
                f"range={best_range:.3f}m, angle_ctrl={theta_control:.3f}rad, "
                f"angle_scan={wrap_to_pi(angle_scan):.3f}rad, out={out.vector.y:.3f}rad, "
                f"sync_diff={sync_error_sec:.3f}s"
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
