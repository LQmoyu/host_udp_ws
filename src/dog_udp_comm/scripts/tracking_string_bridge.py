#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge `/tracking` (std_msgs/String) to structured ROS topics.

Input:
- std_msgs/String on `tracking_topic`, expected payload:
  "[x y detected timestamp]"
  x,y in [0,1], detected in {-1, 1}, timestamp is wall-clock time.

Output:
- geometry_msgs/Vector3Stamped on `state_topic`:
  vector.x = normalized x, vector.y = normalized y, vector.z = detected flag.
- geometry_msgs/PointStamped on `pixel_topic` (only when detected == 1):
  point.x = pixel u, point.y = pixel v.
"""

import re
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
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

    if detected_raw not in (-1.0, 1.0):
        return None
    if ts_raw < 0.0:
        return None

    return x, y, int(detected_raw), ts_raw


def normalize_timestamp_to_ms(ts_raw: float) -> float:
    """
    Normalize common timestamp scales to milliseconds.
    - ns epoch: ~1e18
    - us epoch: ~1e15
    - ms epoch: ~1e12
    - sec epoch: ~1e9
    """
    if ts_raw >= 1.0e17:
        return ts_raw / 1.0e6
    if ts_raw >= 1.0e14:
        return ts_raw / 1.0e3
    if ts_raw >= 1.0e11:
        return ts_raw
    if ts_raw >= 1.0e9:
        return ts_raw * 1.0e3
    return ts_raw


class TrackingStringBridgeNode(Node):
    def __init__(self):
        super().__init__("tracking_string_bridge")

        self.declare_parameter("tracking_topic", "/tracking")
        self.declare_parameter("state_topic", "/tracking_state")
        self.declare_parameter("pixel_topic", "/tracking_pixel")
        self.declare_parameter("image_width", 640.0)
        self.declare_parameter("image_height", 480.0)
        self.declare_parameter("max_allowed_age_ms", 500.0)
        self.declare_parameter("max_consecutive_drop", 3)
        self.declare_parameter("warn_interval_sec", 1.0)

        self.tracking_topic = str(self.get_parameter("tracking_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.pixel_topic = str(self.get_parameter("pixel_topic").value)
        self.image_width = float(self.get_parameter("image_width").value)
        self.image_height = float(self.get_parameter("image_height").value)
        self.max_allowed_age_ms = float(self.get_parameter("max_allowed_age_ms").value)
        self.max_consecutive_drop = int(self.get_parameter("max_consecutive_drop").value)
        self.warn_interval_sec = float(self.get_parameter("warn_interval_sec").value)

        self.consecutive_drop = 0
        self.last_warn_ns = 0

        self.state_pub = self.create_publisher(Vector3Stamped, self.state_topic, 10)
        self.pixel_pub = self.create_publisher(PointStamped, self.pixel_topic, 10)
        self.create_subscription(String, self.tracking_topic, self.tracking_cb, 10)

        self.get_logger().info(
            "Tracking bridge ready. "
            f"tracking_topic={self.tracking_topic}, state_topic={self.state_topic}, pixel_topic={self.pixel_topic}"
        )

    def warn_throttle(self, text: str):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_warn_ns >= int(self.warn_interval_sec * 1e9):
            self.get_logger().warn(text)
            self.last_warn_ns = now_ns

    def bump_drop(self, reason: str):
        self.consecutive_drop += 1
        if self.consecutive_drop > self.max_consecutive_drop:
            self.warn_throttle(
                f"Consecutive dropped tracking frames={self.consecutive_drop}, reason={reason}"
            )

    def tracking_cb(self, msg: String):
        parsed = parse_tracking_text(msg.data)
        if parsed is None:
            self.bump_drop("parse_or_detected_invalid")
            return

        x, y, detected, ts_raw = parsed
        ts_ms = normalize_timestamp_to_ms(ts_raw)
        now_ms = self.get_clock().now().nanoseconds / 1.0e6
        age_ms = now_ms - ts_ms

        if age_ms > self.max_allowed_age_ms:
            self.bump_drop(f"stale_age_ms={age_ms:.1f}")
            return

        if age_ms < -200.0:
            self.bump_drop(f"future_timestamp_age_ms={age_ms:.1f}")
            return

        stamp = self.get_clock().now().to_msg()
        state_msg = Vector3Stamped()
        state_msg.header.stamp = stamp
        state_msg.header.frame_id = "tracking"
        state_msg.vector.x = float(x)
        state_msg.vector.y = float(y)
        state_msg.vector.z = float(detected)
        self.state_pub.publish(state_msg)

        if detected == 1:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                self.bump_drop("x_or_y_out_of_range_when_detected")
                return

            pixel_msg = PointStamped()
            pixel_msg.header = state_msg.header
            pixel_msg.point.x = float(x * max(self.image_width - 1.0, 1.0))
            pixel_msg.point.y = float(y * max(self.image_height - 1.0, 1.0))
            pixel_msg.point.z = 0.0
            self.pixel_pub.publish(pixel_msg)

        self.consecutive_drop = 0


def main():
    rclpy.init()
    node = TrackingStringBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
