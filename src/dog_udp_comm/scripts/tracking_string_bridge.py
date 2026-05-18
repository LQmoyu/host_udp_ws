#!/usr/bin/python3
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
import math
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import String

INT32_TIME_MIN = -2147483648
INT32_TIME_MAX = 2147483647


def parse_tracking_text(text: str) -> Tuple[Optional[Tuple[float, float, int, float]], str]:
    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(values) < 4:
        return None, "parse_failed_not_enough_fields"

    x = float(values[0])
    y = float(values[1])
    detected_raw = float(values[2])
    ts_raw = float(values[3])

    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(detected_raw) and math.isfinite(ts_raw)):
        return None, "parse_failed_non_finite_number"

    if detected_raw not in (-1.0, 1.0):
        return None, f"detected_invalid_{detected_raw}"
    if ts_raw < 0.0:
        return None, "timestamp_negative"

    return (x, y, int(detected_raw), ts_raw), ""


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


def split_stamp_from_ms(ts_ms: float) -> Optional[Tuple[int, int]]:
    if not math.isfinite(ts_ms):
        return None

    total_ns = int(round(ts_ms * 1.0e6))
    sec = total_ns // 1_000_000_000
    nanosec = total_ns - sec * 1_000_000_000
    if sec < INT32_TIME_MIN or sec > INT32_TIME_MAX:
        return None
    if nanosec < 0 or nanosec >= 1_000_000_000:
        return None
    return sec, nanosec


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
        self.declare_parameter("use_source_timestamp_for_header", True)
        self.declare_parameter("topic_check_period_sec", 0.5)
        self.declare_parameter("tracking_topic_lost_timeout_sec", 1.0)

        self.tracking_topic = str(self.get_parameter("tracking_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.pixel_topic = str(self.get_parameter("pixel_topic").value)
        self.image_width = float(self.get_parameter("image_width").value)
        self.image_height = float(self.get_parameter("image_height").value)
        self.max_allowed_age_ms = float(self.get_parameter("max_allowed_age_ms").value)
        self.max_consecutive_drop = int(self.get_parameter("max_consecutive_drop").value)
        self.warn_interval_sec = float(self.get_parameter("warn_interval_sec").value)
        self.use_source_timestamp_for_header = bool(
            self.get_parameter("use_source_timestamp_for_header").value
        )
        self.topic_check_period_sec = max(0.1, float(self.get_parameter("topic_check_period_sec").value))
        self.tracking_topic_lost_timeout_sec = max(
            0.1, float(self.get_parameter("tracking_topic_lost_timeout_sec").value)
        )

        self.consecutive_drop = 0
        self.last_warn_ns = 0
        self.last_any_tracking_rx_ns = 0
        self.last_valid_tracking_rx_ns = 0
        self.last_has_publisher_ns = 0

        self.state_pub = self.create_publisher(Vector3Stamped, self.state_topic, 10)
        self.pixel_pub = self.create_publisher(PointStamped, self.pixel_topic, 10)
        self.create_subscription(String, self.tracking_topic, self.tracking_cb, 10)
        self.watchdog_timer = self.create_timer(self.topic_check_period_sec, self.topic_watchdog_tick)

        self.get_logger().info(
            "Tracking bridge ready. "
            f"tracking_topic={self.tracking_topic}, state_topic={self.state_topic}, pixel_topic={self.pixel_topic}, "
            f"use_source_timestamp_for_header={self.use_source_timestamp_for_header}, "
            f"topic_lost_timeout={self.tracking_topic_lost_timeout_sec:.2f}s"
        )

    def warn_throttle(self, text: str):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_warn_ns >= int(self.warn_interval_sec * 1e9):
            self.get_logger().warn(text)
            self.last_warn_ns = now_ns

    def bump_drop(self, reason: str, raw_text: Optional[str] = None):
        self.consecutive_drop += 1
        if self.consecutive_drop > self.max_consecutive_drop:
            self.warn_throttle(
                f"Consecutive dropped tracking frames={self.consecutive_drop}, reason={reason}"
            )
        if reason.startswith("detected_invalid_") or reason.startswith("parse_failed_"):
            raw_short = ""
            if raw_text is not None:
                raw_short = raw_text.strip().replace("\n", " ")[:120]
            self.warn_throttle(f"Drop /tracking frame: reason={reason}, raw='{raw_short}'")

    def topic_watchdog_tick(self):
        now_ns = self.get_clock().now().nanoseconds
        pub_count = self.count_publishers(self.tracking_topic)

        if pub_count > 0:
            self.last_has_publisher_ns = now_ns
        else:
            if self.last_has_publisher_ns == 0:
                self.last_has_publisher_ns = now_ns
            no_pub_sec = (now_ns - self.last_has_publisher_ns) * 1e-9
            if no_pub_sec > self.tracking_topic_lost_timeout_sec:
                self.warn_throttle(
                    f"Tracking topic liveliness lost: no publisher on {self.tracking_topic} "
                    f"for {no_pub_sec:.2f}s."
                )

        if self.last_any_tracking_rx_ns == 0:
            return

        silent_sec = (now_ns - self.last_any_tracking_rx_ns) * 1e-9
        if silent_sec > self.tracking_topic_lost_timeout_sec:
            self.warn_throttle(
                f"Tracking frame timeout: no new frame on {self.tracking_topic} "
                f"for {silent_sec:.2f}s."
            )

    def tracking_cb(self, msg: String):
        now_ns = self.get_clock().now().nanoseconds
        self.last_any_tracking_rx_ns = now_ns

        parsed, reason = parse_tracking_text(msg.data)
        if parsed is None:
            self.bump_drop(reason, msg.data)
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

        if self.use_source_timestamp_for_header:
            stamp_pair = split_stamp_from_ms(ts_ms)
            if stamp_pair is None:
                self.bump_drop("timestamp_out_of_ros_time_range")
                return
            stamp_sec, stamp_nanosec = stamp_pair
            stamp = self.get_clock().now().to_msg()
            stamp.sec = int(stamp_sec)
            stamp.nanosec = int(stamp_nanosec)
        else:
            stamp = self.get_clock().now().to_msg()

        if detected == 1:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                self.bump_drop("x_or_y_out_of_range_when_detected")
                return

        state_msg = Vector3Stamped()
        state_msg.header.stamp = stamp
        state_msg.header.frame_id = "tracking"
        if detected == 1:
            state_msg.vector.x = float(x)
            state_msg.vector.y = float(y)
        else:
            # Document rule: x/y are invalid when detected != 1.
            state_msg.vector.x = 0.0
            state_msg.vector.y = 0.0
        state_msg.vector.z = float(detected)
        self.state_pub.publish(state_msg)

        if detected == 1:
            pixel_msg = PointStamped()
            pixel_msg.header = state_msg.header
            pixel_msg.point.x = float(x * max(self.image_width - 1.0, 1.0))
            pixel_msg.point.y = float(y * max(self.image_height - 1.0, 1.0))
            pixel_msg.point.z = 0.0
            self.pixel_pub.publish(pixel_msg)

        self.last_valid_tracking_rx_ns = now_ns
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
