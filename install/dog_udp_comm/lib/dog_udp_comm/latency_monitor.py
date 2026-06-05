#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Online latency monitor for the person-following pipeline.

It only subscribes and prints diagnostics. It does not publish control commands.
"""

import math
import re
import statistics
from collections import deque
from typing import Optional

import rclpy
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import String


def normalize_timestamp_to_ms(ts_raw: float) -> float:
    if ts_raw >= 1.0e17:
        return ts_raw / 1.0e6
    if ts_raw >= 1.0e14:
        return ts_raw / 1.0e3
    if ts_raw >= 1.0e11:
        return ts_raw
    if ts_raw >= 1.0e9:
        return ts_raw * 1.0e3
    return ts_raw


def parse_tracking_timestamp_ms(text: str) -> Optional[float]:
    values = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if len(values) < 4:
        return None
    ts_raw = float(values[3])
    if not math.isfinite(ts_raw) or ts_raw < 0.0:
        return None
    return normalize_timestamp_to_ms(ts_raw)


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def ms_from_ns(ns: int) -> float:
    return ns / 1.0e6


def fmt_ms(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.1f}ms"


def fmt_hz(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.1f}Hz"


class TopicStats:
    def __init__(self, window_size: int):
        self.times_ns = deque(maxlen=max(2, window_size))
        self.header_ages_ms = deque(maxlen=max(2, window_size))
        self.last_rx_ns = 0
        self.last_header_stamp_ns = 0

    def push_rx(self, rx_ns: int):
        self.times_ns.append(rx_ns)
        self.last_rx_ns = rx_ns

    def push_header_age(self, rx_ns: int, stamp_ns: int):
        if stamp_ns <= 0:
            return
        self.last_header_stamp_ns = stamp_ns
        self.header_ages_ms.append(ms_from_ns(rx_ns - stamp_ns))

    def hz(self) -> Optional[float]:
        if len(self.times_ns) < 2:
            return None
        duration_sec = (self.times_ns[-1] - self.times_ns[0]) * 1.0e-9
        if duration_sec <= 1e-6:
            return None
        return (len(self.times_ns) - 1) / duration_sec

    def median_header_age_ms(self) -> Optional[float]:
        if not self.header_ages_ms:
            return None
        return statistics.median(self.header_ages_ms)

    def p95_header_age_ms(self) -> Optional[float]:
        if not self.header_ages_ms:
            return None
        values = sorted(self.header_ages_ms)
        idx = int(round((len(values) - 1) * 0.95))
        return values[idx]


class LatencyMonitorNode(Node):
    def __init__(self):
        super().__init__("person_follow_latency_monitor")

        self.declare_parameter("tracking_topic", "/tracking")
        self.declare_parameter("tracking_state_topic", "/tracking_state")
        self.declare_parameter("pixel_topic", "/tracking_pixel")
        self.declare_parameter("person_topic", "/person_polar")
        self.declare_parameter("person_debug_topic", "/person_polar_debug")
        self.declare_parameter("cmd_topic", "/track_cmd_vel")
        self.declare_parameter("report_period_sec", 1.0)
        self.declare_parameter("window_size", 120)
        self.declare_parameter("relative_timestamp_threshold_ms", 1.0e9)

        self.tracking_topic = str(self.get_parameter("tracking_topic").value)
        self.tracking_state_topic = str(self.get_parameter("tracking_state_topic").value)
        self.pixel_topic = str(self.get_parameter("pixel_topic").value)
        self.person_topic = str(self.get_parameter("person_topic").value)
        self.person_debug_topic = str(self.get_parameter("person_debug_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.report_period_sec = max(0.2, float(self.get_parameter("report_period_sec").value))
        self.window_size = max(10, int(self.get_parameter("window_size").value))
        self.relative_timestamp_threshold_ms = float(
            self.get_parameter("relative_timestamp_threshold_ms").value
        )

        self.tracking = TopicStats(self.window_size)
        self.state = TopicStats(self.window_size)
        self.pixel = TopicStats(self.window_size)
        self.person = TopicStats(self.window_size)
        self.cmd = TopicStats(self.window_size)

        self.tracking_source_ages_ms = deque(maxlen=self.window_size)
        self.sync_diffs_ms = deque(maxlen=self.window_size)
        self.cmd_from_state_ms = deque(maxlen=self.window_size)
        self.cmd_from_person_ms = deque(maxlen=self.window_size)
        self.latest_cmd_v = 0.0
        self.latest_cmd_w = 0.0
        self.latest_person_debug = "n/a"

        self.create_subscription(String, self.tracking_topic, self.tracking_cb, 10)
        self.create_subscription(Vector3Stamped, self.tracking_state_topic, self.state_cb, 10)
        self.create_subscription(PointStamped, self.pixel_topic, self.pixel_cb, 10)
        self.create_subscription(Vector3Stamped, self.person_topic, self.person_cb, 10)
        self.create_subscription(String, self.person_debug_topic, self.person_debug_cb, 10)
        self.create_subscription(Twist, self.cmd_topic, self.cmd_cb, 10)
        self.create_timer(self.report_period_sec, self.report)

        self.get_logger().info(
            "Latency monitor ready. "
            f"tracking={self.tracking_topic}, state={self.tracking_state_topic}, "
            f"pixel={self.pixel_topic}, person={self.person_topic}, "
            f"person_debug={self.person_debug_topic}, cmd={self.cmd_topic}"
        )

    def now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def tracking_cb(self, msg: String):
        rx_ns = self.now_ns()
        self.tracking.push_rx(rx_ns)
        ts_ms = parse_tracking_timestamp_ms(msg.data)
        if ts_ms is None:
            return
        if ts_ms < self.relative_timestamp_threshold_ms:
            return
        self.tracking_source_ages_ms.append(ms_from_ns(rx_ns) - ts_ms)

    def state_cb(self, msg: Vector3Stamped):
        rx_ns = self.now_ns()
        self.state.push_rx(rx_ns)
        self.state.push_header_age(rx_ns, stamp_to_ns(msg.header.stamp))

    def pixel_cb(self, msg: PointStamped):
        rx_ns = self.now_ns()
        self.pixel.push_rx(rx_ns)
        self.pixel.push_header_age(rx_ns, stamp_to_ns(msg.header.stamp))

    def person_cb(self, msg: Vector3Stamped):
        rx_ns = self.now_ns()
        self.person.push_rx(rx_ns)
        self.person.push_header_age(rx_ns, stamp_to_ns(msg.header.stamp))
        if math.isfinite(msg.vector.z) and msg.vector.z >= 0.0:
            self.sync_diffs_ms.append(msg.vector.z * 1000.0)

    def cmd_cb(self, msg: Twist):
        rx_ns = self.now_ns()
        self.cmd.push_rx(rx_ns)
        self.latest_cmd_v = float(msg.linear.x)
        self.latest_cmd_w = float(msg.angular.z)
        if self.state.last_header_stamp_ns > 0:
            self.cmd_from_state_ms.append(ms_from_ns(rx_ns - self.state.last_header_stamp_ns))
        if self.person.last_header_stamp_ns > 0:
            self.cmd_from_person_ms.append(ms_from_ns(rx_ns - self.person.last_header_stamp_ns))

    def person_debug_cb(self, msg: String):
        self.latest_person_debug = msg.data

    def median(self, values) -> Optional[float]:
        if not values:
            return None
        return statistics.median(values)

    def p95(self, values) -> Optional[float]:
        if not values:
            return None
        sorted_values = sorted(values)
        idx = int(round((len(sorted_values) - 1) * 0.95))
        return sorted_values[idx]

    def report(self):
        lines = [
            "latency summary:",
            f"  hz: tracking={fmt_hz(self.tracking.hz())}, "
            f"state={fmt_hz(self.state.hz())}, pixel={fmt_hz(self.pixel.hz())}, "
            f"person={fmt_hz(self.person.hz())}, cmd={fmt_hz(self.cmd.hz())}",
            f"  header_age_p50: state={fmt_ms(self.state.median_header_age_ms())}, "
            f"pixel={fmt_ms(self.pixel.median_header_age_ms())}, "
            f"person={fmt_ms(self.person.median_header_age_ms())}",
            f"  header_age_p95: state={fmt_ms(self.state.p95_header_age_ms())}, "
            f"pixel={fmt_ms(self.pixel.p95_header_age_ms())}, "
            f"person={fmt_ms(self.person.p95_header_age_ms())}",
            f"  camera_lidar_sync: p50={fmt_ms(self.median(self.sync_diffs_ms))}, "
            f"p95={fmt_ms(self.p95(self.sync_diffs_ms))}",
            f"  cmd_age: from_tracking_state={fmt_ms(self.median(self.cmd_from_state_ms))}, "
            f"from_person_polar={fmt_ms(self.median(self.cmd_from_person_ms))}, "
            f"last_cmd_v={self.latest_cmd_v:+.3f}, last_cmd_w={self.latest_cmd_w:+.3f}",
            f"  lidar_match: {self.latest_person_debug}",
        ]
        source_age = self.median(self.tracking_source_ages_ms)
        if source_age is not None:
            lines.append(f"  tracking_source_age_p50: {fmt_ms(source_age)}")
        self.get_logger().info("\n".join(lines))


def main():
    rclpy.init()
    node = LatencyMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
