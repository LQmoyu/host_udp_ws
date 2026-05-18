#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Simple center-alignment tester using `/tracking` (std_msgs/String).

Input:
- /tracking payload format: [x y detected timestamp_ms]

Output:
- geometry_msgs/Twist on /track_cmd_vel

Control target:
- Keep detected target centered at x = 0.5.
"""

import math
import re
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String


def parse_tracking_text(text: str) -> Optional[Tuple[float, float, int, float]]:
    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(values) < 4:
        return None

    x = float(values[0])
    y = float(values[1])
    detected_raw = float(values[2])
    ts_raw = float(values[3])

    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(detected_raw) and math.isfinite(ts_raw)):
        return None
    if detected_raw not in (-1.0, 1.0):
        return None
    if ts_raw < 0.0:
        return None

    return x, y, int(detected_raw), ts_raw


def normalize_timestamp_to_ms(ts_raw: float) -> float:
    # Supports common epoch scales: ns/us/ms/s.
    if ts_raw >= 1.0e17:
        return ts_raw / 1.0e6
    if ts_raw >= 1.0e14:
        return ts_raw / 1.0e3
    if ts_raw >= 1.0e11:
        return ts_raw
    if ts_raw >= 1.0e9:
        return ts_raw * 1.0e3
    return ts_raw


class TrackingCenterTester(Node):
    def __init__(self):
        super().__init__("tracking_center_tester")

        self.declare_parameter("tracking_topic", "/tracking")
        self.declare_parameter("cmd_topic", "/track_cmd_vel")

        self.declare_parameter("target_x", 0.5)
        self.declare_parameter("x_tolerance", 0.03)
        self.declare_parameter("k_angular", 2.0)
        self.declare_parameter("max_angular", 1.2)
        self.declare_parameter("reverse_angular", False)
        self.declare_parameter("linear_speed_when_centered", 0.0)

        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("frame_timeout_sec", 0.3)
        self.declare_parameter("max_allowed_age_ms", 500.0)
        self.declare_parameter("stop_on_target_lost", True)
        self.declare_parameter("warn_interval_sec", 1.0)

        self.tracking_topic = str(self.get_parameter("tracking_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.target_x = float(self.get_parameter("target_x").value)
        self.x_tolerance = max(0.0, float(self.get_parameter("x_tolerance").value))
        self.k_angular = float(self.get_parameter("k_angular").value)
        self.max_angular = max(0.0, float(self.get_parameter("max_angular").value))
        self.reverse_angular = bool(self.get_parameter("reverse_angular").value)
        self.linear_speed_when_centered = float(self.get_parameter("linear_speed_when_centered").value)

        self.control_hz = max(1.0, float(self.get_parameter("control_hz").value))
        self.frame_timeout_sec = max(0.05, float(self.get_parameter("frame_timeout_sec").value))
        self.max_allowed_age_ms = max(1.0, float(self.get_parameter("max_allowed_age_ms").value))
        self.stop_on_target_lost = bool(self.get_parameter("stop_on_target_lost").value)
        self.warn_interval_sec = max(0.1, float(self.get_parameter("warn_interval_sec").value))

        self.last_warn_ns = 0
        self.last_rx_ns = 0
        self.have_frame = False

        self.latest_x = 0.5
        self.latest_detected = -1
        self.latest_age_ms = 0.0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.create_subscription(String, self.tracking_topic, self.tracking_cb, 10)
        self.timer = self.create_timer(1.0 / self.control_hz, self.control_tick)

        self.get_logger().info(
            "Tracking center tester started. "
            f"tracking_topic={self.tracking_topic}, cmd_topic={self.cmd_topic}, "
            f"target_x={self.target_x:.3f}, tol={self.x_tolerance:.3f}, "
            f"k_angular={self.k_angular:.3f}, max_angular={self.max_angular:.3f}, "
            f"hz={self.control_hz:.1f}"
        )

    def warn_throttle(self, text: str):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_warn_ns >= int(self.warn_interval_sec * 1e9):
            self.get_logger().warn(text)
            self.last_warn_ns = now_ns

    def publish_cmd(self, linear_x: float, angular_z: float):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self.cmd_pub.publish(msg)

    def publish_zero(self):
        self.publish_cmd(0.0, 0.0)

    def tracking_cb(self, msg: String):
        parsed = parse_tracking_text(msg.data)
        if parsed is None:
            self.warn_throttle("Invalid /tracking payload, ignore frame.")
            return

        x, _y, detected, ts_raw = parsed
        ts_ms = normalize_timestamp_to_ms(ts_raw)
        now_ms = self.get_clock().now().nanoseconds / 1.0e6
        age_ms = now_ms - ts_ms

        if age_ms > self.max_allowed_age_ms:
            self.warn_throttle(
                f"Drop stale tracking frame: age_ms={age_ms:.1f} > {self.max_allowed_age_ms:.1f}"
            )
            return
        if age_ms < -200.0:
            self.warn_throttle(f"Drop future tracking frame: age_ms={age_ms:.1f}")
            return

        self.latest_x = float(x)
        self.latest_detected = int(detected)
        self.latest_age_ms = float(age_ms)
        self.last_rx_ns = self.get_clock().now().nanoseconds
        self.have_frame = True

    def control_tick(self):
        if not self.have_frame:
            self.warn_throttle("Waiting for /tracking frames...")
            if self.stop_on_target_lost:
                self.publish_zero()
            return

        now_ns = self.get_clock().now().nanoseconds
        silent_sec = (now_ns - self.last_rx_ns) * 1e-9
        if silent_sec > self.frame_timeout_sec:
            self.warn_throttle(
                f"/tracking timeout: no fresh frame for {silent_sec:.3f}s > {self.frame_timeout_sec:.3f}s"
            )
            if self.stop_on_target_lost:
                self.publish_zero()
            return

        if self.latest_detected != 1:
            self.warn_throttle("Target not detected (detected=-1), publish zero.")
            if self.stop_on_target_lost:
                self.publish_zero()
            return

        if not (0.0 <= self.latest_x <= 1.0):
            self.warn_throttle(f"Invalid x={self.latest_x:.3f}, publish zero.")
            self.publish_zero()
            return

        error_x = self.target_x - self.latest_x
        if abs(error_x) <= self.x_tolerance:
            linear = self.linear_speed_when_centered
            angular = 0.0
        else:
            sign = -1.0 if self.reverse_angular else 1.0
            angular = sign * self.k_angular * error_x
            angular = max(-self.max_angular, min(self.max_angular, angular))
            linear = 0.0

        self.publish_cmd(linear, angular)


def main():
    rclpy.init()
    node = TrackingCenterTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_zero()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
