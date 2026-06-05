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
from std_msgs.msg import String


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
        self.declare_parameter("search_mode", "sector")
        self.declare_parameter("sector_half_angle_deg", 10.0)
        self.declare_parameter("sector_min_points", 1)
        self.declare_parameter("sector_select_mode", "percentile")
        self.declare_parameter("sector_range_percentile", 35.0)
        self.declare_parameter("ignore_close_range_below", 0.0)
        self.declare_parameter("ignore_close_if_farther_exists", 0.0)
        self.declare_parameter("enable_output_filter", True)
        self.declare_parameter("range_filter_alpha_near", 0.35)
        self.declare_parameter("range_filter_alpha_far", 0.70)
        self.declare_parameter("angle_filter_alpha", 0.50)
        self.declare_parameter("max_range_step_up", 0.0)
        self.declare_parameter("max_range_step_down", 0.0)
        self.declare_parameter("range_min_valid", 0.10)
        self.declare_parameter("range_max_valid", 10.0)
        self.declare_parameter("prefer_scan_angle_output", False)
        self.declare_parameter("publish_debug", False)
        self.declare_parameter("debug_topic", "/person_polar_debug")
        self.declare_parameter("debug_log_period_sec", 1.0)
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
        self.search_mode = str(self.get_parameter("search_mode").value).strip().lower()
        self.sector_half_angle_rad = math.radians(
            max(0.0, float(self.get_parameter("sector_half_angle_deg").value))
        )
        self.sector_min_points = max(1, int(self.get_parameter("sector_min_points").value))
        self.sector_select_mode = str(self.get_parameter("sector_select_mode").value).strip().lower()
        self.sector_range_percentile = float(
            np.clip(float(self.get_parameter("sector_range_percentile").value), 0.0, 100.0)
        )
        self.ignore_close_range_below = max(
            0.0,
            float(self.get_parameter("ignore_close_range_below").value),
        )
        self.ignore_close_if_farther_exists = max(
            0.0,
            float(self.get_parameter("ignore_close_if_farther_exists").value),
        )
        self.enable_output_filter = bool(self.get_parameter("enable_output_filter").value)
        self.range_filter_alpha_near = float(
            np.clip(float(self.get_parameter("range_filter_alpha_near").value), 0.0, 0.98)
        )
        self.range_filter_alpha_far = float(
            np.clip(float(self.get_parameter("range_filter_alpha_far").value), 0.0, 0.98)
        )
        self.angle_filter_alpha = float(
            np.clip(float(self.get_parameter("angle_filter_alpha").value), 0.0, 0.98)
        )
        self.max_range_step_up = max(0.0, float(self.get_parameter("max_range_step_up").value))
        self.max_range_step_down = max(0.0, float(self.get_parameter("max_range_step_down").value))
        self.range_min_valid = float(self.get_parameter("range_min_valid").value)
        self.range_max_valid = float(self.get_parameter("range_max_valid").value)
        self.prefer_scan_angle_output = bool(self.get_parameter("prefer_scan_angle_output").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)
        self.debug_log_period_sec = max(0.1, float(self.get_parameter("debug_log_period_sec").value))
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
        if self.search_mode not in ("window", "sector"):
            self.get_logger().warn(
                f"Unknown search_mode='{self.search_mode}', fallback to 'sector'."
            )
            self.search_mode = "sector"
        if self.sector_select_mode not in ("nearest", "percentile", "median"):
            self.get_logger().warn(
                f"Unknown sector_select_mode='{self.sector_select_mode}', fallback to 'percentile'."
            )
            self.sector_select_mode = "percentile"

        self.scan_lock = Lock()
        self.scan_buffer = deque()
        self.last_warn_ns = 0
        self.last_debug_log_ns = 0
        self.pixel_count = 0
        self.hit_count = 0
        self.drop_count = 0
        self.last_diag = ""
        self.filtered_range = None
        self.filtered_angle = None

        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, 10)
        self.create_subscription(PointStamped, self.pixel_topic, self.pixel_cb, 10)
        self.polar_pub = self.create_publisher(Vector3Stamped, self.out_topic, 10)
        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        self.get_logger().info(
            "Pixel->Scan polar bridge ready. "
            f"scan_topic={self.scan_topic} pixel_topic={self.pixel_topic} out_topic={self.out_topic} "
            f"fx={self.fx:.3f} cx={self.cx:.3f} yaw_cam_to_lidar={self.yaw_cam_to_lidar:.3f} "
            f"search_mode={self.search_mode} sector_half_angle={math.degrees(self.sector_half_angle_rad):.1f}deg "
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

    def publish_diag(self, text: str, force_log: bool = False):
        self.last_diag = text
        msg = String()
        msg.data = text
        self.debug_pub.publish(msg)
        if self.publish_debug:
            now_ns = self.get_clock().now().nanoseconds
            if force_log or now_ns - self.last_debug_log_ns >= int(self.debug_log_period_sec * 1e9):
                self.get_logger().info(text)
                self.last_debug_log_ns = now_ns

    def filter_output(self, raw_range: float, raw_angle: float):
        if not self.enable_output_filter:
            self.filtered_range = raw_range
            self.filtered_angle = raw_angle
            return raw_range, raw_angle

        if self.filtered_range is None or self.filtered_angle is None:
            self.filtered_range = raw_range
            self.filtered_angle = raw_angle
            return raw_range, raw_angle

        range_in = raw_range
        if range_in > self.filtered_range and self.max_range_step_up > 0.0:
            range_in = min(range_in, self.filtered_range + self.max_range_step_up)
        elif range_in < self.filtered_range and self.max_range_step_down > 0.0:
            range_in = max(range_in, self.filtered_range - self.max_range_step_down)

        # Trust closer measurements faster for safety, smooth farther jumps more strongly.
        alpha_r = self.range_filter_alpha_far if range_in > self.filtered_range else self.range_filter_alpha_near
        self.filtered_range = alpha_r * self.filtered_range + (1.0 - alpha_r) * range_in

        alpha_a = self.angle_filter_alpha
        ca = alpha_a * math.cos(self.filtered_angle) + (1.0 - alpha_a) * math.cos(raw_angle)
        sa = alpha_a * math.sin(self.filtered_angle) + (1.0 - alpha_a) * math.sin(raw_angle)
        self.filtered_angle = math.atan2(sa, ca)
        return self.filtered_range, self.filtered_angle

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

    def _scan_index_angle(self, scan: LaserScan, idx: int) -> float:
        return scan.angle_min + idx * scan.angle_increment

    def _angle_to_scan_index(self, scan: LaserScan, angle_2pi: float) -> int:
        idx = int(round((angle_2pi - scan.angle_min) / scan.angle_increment))
        return max(0, min(len(scan.ranges) - 1, idx))

    def _sector_indices(self, scan: LaserScan, center_angle_2pi: float):
        if self.sector_half_angle_rad <= 0.0:
            center_idx = self._angle_to_scan_index(scan, center_angle_2pi)
            return [center_idx]

        n = len(scan.ranges)
        indices = []
        for i in range(n):
            a = self._scan_index_angle(scan, i)
            diff = abs(wrap_to_pi(a - center_angle_2pi))
            if diff <= self.sector_half_angle_rad:
                indices.append(i)
        return indices

    def _find_sector_valid(self, scan: LaserScan, center_angle_2pi: float):
        candidates = []
        for i in self._sector_indices(scan, center_angle_2pi):
            r = float(scan.ranges[i])
            if not self._is_valid_range(r):
                continue
            angle = self._scan_index_angle(scan, i)
            angle_err = abs(wrap_to_pi(angle - center_angle_2pi))
            candidates.append((i, r, angle, angle_err))

        if len(candidates) < self.sector_min_points:
            return -1, 0.0, 0, 0.0, 0.0, 0

        raw_valid_count = len(candidates)
        raw_ranges = [item[1] for item in candidates]
        raw_min = min(raw_ranges)
        raw_max = max(raw_ranges)
        filtered_close_count = 0
        if (
            self.ignore_close_range_below > 0.0
            and self.ignore_close_if_farther_exists > self.ignore_close_range_below
            and any(item[1] >= self.ignore_close_if_farther_exists for item in candidates)
        ):
            kept = [item for item in candidates if item[1] >= self.ignore_close_range_below]
            if len(kept) >= self.sector_min_points:
                filtered_close_count = len(candidates) - len(kept)
                candidates = kept

        if self.sector_select_mode == "nearest":
            selected = min(candidates, key=lambda item: (item[1], item[3]))
        elif self.sector_select_mode == "median":
            candidates_sorted = sorted(candidates, key=lambda item: item[1])
            selected = candidates_sorted[len(candidates_sorted) // 2]
        else:
            ranges = np.array([item[1] for item in candidates], dtype=float)
            target_range = float(np.percentile(ranges, self.sector_range_percentile))
            selected = min(candidates, key=lambda item: (abs(item[1] - target_range), item[3]))

        return selected[0], selected[1], raw_valid_count, raw_min, raw_max, filtered_close_count

    def pixel_cb(self, msg: PointStamped):
        now_ns = self.get_clock().now().nanoseconds
        self.pixel_count += 1
        pixel_stamp_ns = self._message_stamp_ns(msg.header.stamp, now_ns)
        pixel_age_sec = max(0.0, (now_ns - pixel_stamp_ns) * 1e-9)
        scan, sync_error_sec = self._select_scan(pixel_stamp_ns)

        if scan is None:
            self.drop_count += 1
            self.publish_diag(
                f"hit=0 reason=no_scan pixel_age={pixel_age_sec*1000.0:.1f}ms "
                f"pixels={self.pixel_count} hits={self.hit_count} drops={self.drop_count}"
            )
            self.warn_throttle("Waiting for first /scan frame before pixel-to-scan matching.")
            return
        if len(scan.ranges) == 0:
            self.drop_count += 1
            self.publish_diag(
                f"hit=0 reason=empty_scan pixel_age={pixel_age_sec*1000.0:.1f}ms "
                f"sync_diff={sync_error_sec*1000.0:.1f}ms"
            )
            return
        if sync_error_sec > self.max_sync_diff_sec:
            text = (
                f"Camera/Lidar sync diff {sync_error_sec:.3f}s exceeds "
                f"max_sync_diff_sec={self.max_sync_diff_sec:.3f}s."
            )
            if self.sync_policy == "strict":
                self.drop_count += 1
                self.publish_diag(
                    f"hit=0 reason=sync_drop pixel_age={pixel_age_sec*1000.0:.1f}ms "
                    f"sync_diff={sync_error_sec*1000.0:.1f}ms"
                )
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

        idx = self._angle_to_scan_index(scan, theta_lidar)

        valid_count = 0
        sector_min = 0.0
        sector_max = 0.0
        filtered_close_count = 0
        if self.search_mode == "sector":
            (
                best_idx,
                best_range,
                valid_count,
                sector_min,
                sector_max,
                filtered_close_count,
            ) = self._find_sector_valid(
                scan,
                theta_lidar,
            )
        else:
            best_idx, best_range = self._find_nearest_valid(scan.ranges, idx)
            valid_count = 1 if best_idx >= 0 else 0
        if best_idx < 0:
            self.drop_count += 1
            hit_rate = 100.0 * self.hit_count / max(1, self.pixel_count)
            self.publish_diag(
                f"hit=0 reason=no_valid_range mode={self.search_mode} u={u:.1f} "
                f"idx={idx} valid_points={valid_count} pixel_age={pixel_age_sec*1000.0:.1f}ms "
                f"sync_diff={sync_error_sec*1000.0:.1f}ms hit_rate={hit_rate:.1f}% "
                f"pixels={self.pixel_count} hits={self.hit_count} drops={self.drop_count}"
            )
            return

        angle_scan = scan.angle_min + best_idx * scan.angle_increment
        if self.prefer_scan_angle_output:
            angle_out = wrap_to_pi(angle_scan)
        else:
            angle_out = theta_control

        raw_range = float(best_range)
        raw_angle_out = float(angle_out)
        filtered_range, filtered_angle = self.filter_output(raw_range, raw_angle_out)

        out = Vector3Stamped()
        out.header = msg.header
        out.vector.x = float(filtered_range)
        out.vector.y = float(filtered_angle)
        out.vector.z = float(sync_error_sec)
        self.polar_pub.publish(out)

        self.hit_count += 1
        hit_rate = 100.0 * self.hit_count / max(1, self.pixel_count)
        self.publish_diag(
            f"hit=1 mode={self.search_mode} u={u:.1f} center_idx={idx} use_idx={best_idx} "
            f"range={filtered_range:.3f}m raw_range={raw_range:.3f}m valid_points={valid_count} "
            f"sector_min={sector_min:.3f}m sector_max={sector_max:.3f}m "
            f"filtered_close={filtered_close_count} "
            f"angle_ctrl={theta_control:.3f}rad angle_scan={wrap_to_pi(angle_scan):.3f}rad "
            f"out_angle={out.vector.y:.3f}rad raw_angle={raw_angle_out:.3f}rad "
            f"pixel_age={pixel_age_sec*1000.0:.1f}ms "
            f"sync_diff={sync_error_sec*1000.0:.1f}ms hit_rate={hit_rate:.1f}% "
            f"pixels={self.pixel_count} hits={self.hit_count} drops={self.drop_count}"
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
