#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class FrontDistanceTester(Node):
    def __init__(self):
        super().__init__('front_distance_tester')

        self.scan_topic = str(self.declare_parameter('scan_topic', '/scan').value)
        self.front_angle_deg = float(self.declare_parameter('front_angle_deg', 0.0).value)
        self.angle_window_deg = float(self.declare_parameter('angle_window_deg', 3.0).value)
        self.print_hz = float(self.declare_parameter('print_hz', 10.0).value)

        if self.angle_window_deg < 0.0:
            self.angle_window_deg = 0.0
        if self.print_hz <= 0.0:
            self.print_hz = 1.0

        self.latest_scan = None
        self.sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_cb,
            10,
        )
        self.timer = self.create_timer(1.0 / self.print_hz, self.on_timer)

        self.get_logger().info(
            'front_distance_tester (python) started. '
            f'scan_topic={self.scan_topic}, '
            f'front_angle_deg={self.front_angle_deg:.2f}, '
            f'angle_window_deg={self.angle_window_deg:.2f}, '
            f'print_hz={self.print_hz:.2f}'
        )

    @staticmethod
    def wrap_to_pi(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def is_valid_range(r):
        return math.isfinite(r) and r > 0.0

    def angle_to_index(self, scan: LaserScan, target_angle_rad: float):
        if len(scan.ranges) == 0:
            return -1
        if scan.angle_increment <= 0.0:
            return -1

        target = target_angle_rad
        min_a = scan.angle_min
        max_a = scan.angle_max

        if target < min_a:
            target += 2.0 * math.pi
        if target > max_a:
            target -= 2.0 * math.pi

        idx = int(round((target - min_a) / scan.angle_increment))
        idx = max(0, min(len(scan.ranges) - 1, idx))
        return idx

    def scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def on_timer(self):
        scan = self.latest_scan
        if scan is None:
            self.get_logger().warn('Waiting for /scan ...', throttle_duration_sec=2.0)
            return

        center_angle_rad = math.radians(self.front_angle_deg)
        center_idx = self.angle_to_index(scan, center_angle_rad)
        if center_idx < 0:
            self.get_logger().warn('Invalid scan metadata.', throttle_duration_sec=2.0)
            return

        half_win = int(round(math.radians(self.angle_window_deg) / max(scan.angle_increment, 1e-6)))
        left = max(0, center_idx - half_win)
        right = min(len(scan.ranges) - 1, center_idx + half_win)

        center_range = scan.ranges[center_idx]
        center_valid = (
            self.is_valid_range(center_range)
            and center_range >= scan.range_min
            and center_range <= scan.range_max
        )

        valid = []
        for i in range(left, right + 1):
            r = float(scan.ranges[i])
            if self.is_valid_range(r) and scan.range_min <= r <= scan.range_max:
                valid.append(r)

        if not valid:
            self.get_logger().warn(
                f'Front invalid. idx={center_idx}, angle={self.front_angle_deg:.2f} deg, '
                f'window=[{left},{right}], no valid range.'
            )
            return

        valid.sort()
        min_r = valid[0]
        max_r = valid[-1]
        median_r = valid[len(valid) // 2]
        mean_r = sum(valid) / len(valid)

        if center_valid:
            self.get_logger().info(
                f'Front range: center={center_range:.3f} m, '
                f'median={median_r:.3f} m, mean={mean_r:.3f} m, '
                f'min={min_r:.3f} m, max={max_r:.3f} m, valid={len(valid)}'
            )
        else:
            self.get_logger().info(
                f'Front range: center=invalid, '
                f'median={median_r:.3f} m, mean={mean_r:.3f} m, '
                f'min={min_r:.3f} m, max={max_r:.3f} m, valid={len(valid)}'
            )


def main(args=None):
    rclpy.init(args=args)
    node = FrontDistanceTester()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
