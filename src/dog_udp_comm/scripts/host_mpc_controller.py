#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Host-side person-following MPC controller (ROS2).

Input:
- geometry_msgs/Vector3Stamped on `person_topic`
  - vector.x: person distance in meters
  - vector.y: person angle in radians (left positive)

Output:
- geometry_msgs/Twist on `cmd_topic` (default: /track_cmd_vel)
  - forwarded to robot via existing UDP sender_node.
"""

import math
from threading import Lock

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from rclpy.node import Node


def wrap_to_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class HostMPCControllerNode(Node):
    def __init__(self):
        super().__init__("host_mpc_controller")

        self.declare_parameter("person_topic", "/person_polar")
        self.declare_parameter("cmd_topic", "/track_cmd_vel")

        self.declare_parameter("control_hz", 30.0)
        self.declare_parameter("dt", 0.0)  # <=0 means use 1/control_hz
        self.declare_parameter("horizon", 12)
        self.declare_parameter("msg_timeout", 0.4)

        self.declare_parameter("desired_distance", 1.2)
        self.declare_parameter("distance_tolerance", 0.08)
        self.declare_parameter("angle_tolerance", 0.08)
        self.declare_parameter("stop_when_aligned", True)
        self.declare_parameter("stop_on_lost_target", True)
        self.declare_parameter("allow_reverse", False)

        self.declare_parameter("max_v", 0.8)
        self.declare_parameter("max_w", 1.5)
        self.declare_parameter("max_reverse_v", 0.2)

        self.declare_parameter("q_dist", 16.0)
        self.declare_parameter("q_angle", 10.0)
        self.declare_parameter("r_v", 0.35)
        self.declare_parameter("r_w", 0.25)
        self.declare_parameter("qf_scale", 4.0)

        self.declare_parameter("kff_dist", 0.9)
        self.declare_parameter("kff_angle", 1.2)
        self.declare_parameter("distance_filter_alpha", 0.65)
        self.declare_parameter("angle_filter_alpha", 0.65)
        self.declare_parameter("min_valid_distance", 0.15)
        self.declare_parameter("max_valid_distance", 8.0)

        self.person_topic = str(self.get_parameter("person_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.control_hz = float(self.get_parameter("control_hz").value)
        dt_param = float(self.get_parameter("dt").value)
        self.dt = dt_param if dt_param > 0.0 else (1.0 / max(self.control_hz, 1e-3))
        self.horizon = int(self.get_parameter("horizon").value)
        self.msg_timeout = float(self.get_parameter("msg_timeout").value)

        self.desired_distance = float(self.get_parameter("desired_distance").value)
        self.distance_tolerance = float(self.get_parameter("distance_tolerance").value)
        self.angle_tolerance = float(self.get_parameter("angle_tolerance").value)
        self.stop_when_aligned = bool(self.get_parameter("stop_when_aligned").value)
        self.stop_on_lost_target = bool(self.get_parameter("stop_on_lost_target").value)
        self.allow_reverse = bool(self.get_parameter("allow_reverse").value)

        self.max_v = float(self.get_parameter("max_v").value)
        self.max_w = float(self.get_parameter("max_w").value)
        self.max_reverse_v = float(self.get_parameter("max_reverse_v").value)

        self.q_dist = float(self.get_parameter("q_dist").value)
        self.q_angle = float(self.get_parameter("q_angle").value)
        self.r_v = float(self.get_parameter("r_v").value)
        self.r_w = float(self.get_parameter("r_w").value)
        self.qf_scale = float(self.get_parameter("qf_scale").value)

        self.kff_dist = float(self.get_parameter("kff_dist").value)
        self.kff_angle = float(self.get_parameter("kff_angle").value)
        self.distance_filter_alpha = float(self.get_parameter("distance_filter_alpha").value)
        self.angle_filter_alpha = float(self.get_parameter("angle_filter_alpha").value)
        self.min_valid_distance = float(self.get_parameter("min_valid_distance").value)
        self.max_valid_distance = float(self.get_parameter("max_valid_distance").value)

        self.state_lock = Lock()
        self.person_distance = None
        self.person_angle = None
        self.last_person_stamp_ns = 0
        self.last_warn_ns = 0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 1)
        self.create_subscription(Vector3Stamped, self.person_topic, self.person_cb, 10)
        self.timer = self.create_timer(1.0 / max(self.control_hz, 1e-3), self.control_tick)

        self.get_logger().info(
            "Host person-follow MPC ready. "
            f"person_topic={self.person_topic} cmd_topic={self.cmd_topic} "
            f"hz={self.control_hz:.1f} dt={self.dt:.3f} N={self.horizon} "
            f"d_ref={self.desired_distance:.2f}"
        )

    def warn_throttle(self, period_sec, text):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_warn_ns >= int(period_sec * 1e9):
            self.get_logger().warn(text)
            self.last_warn_ns = now_ns

    def publish_zero(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def clip_v(self, v):
        vmax = self.max_v
        if self.allow_reverse:
            vmin = -abs(self.max_reverse_v)
        else:
            vmin = 0.0
        return float(np.clip(v, vmin, vmax))

    def clip_w(self, w):
        return float(np.clip(w, -self.max_w, self.max_w))

    def person_cb(self, msg):
        d_raw = float(msg.vector.x)
        a_raw = wrap_to_pi(float(msg.vector.y))

        if d_raw < self.min_valid_distance or d_raw > self.max_valid_distance:
            return

        now_ns = self.get_clock().now().nanoseconds
        with self.state_lock:
            if self.person_distance is None:
                d = d_raw
                a = a_raw
            else:
                ad = np.clip(self.distance_filter_alpha, 0.0, 1.0)
                aa = np.clip(self.angle_filter_alpha, 0.0, 1.0)
                d = ad * self.person_distance + (1.0 - ad) * d_raw
                # Keep angle filter robust near +/-pi boundary.
                ca = aa * math.cos(self.person_angle) + (1.0 - aa) * math.cos(a_raw)
                sa = aa * math.sin(self.person_angle) + (1.0 - aa) * math.sin(a_raw)
                a = math.atan2(sa, ca)

            self.person_distance = float(d)
            self.person_angle = float(a)
            self.last_person_stamp_ns = now_ns

    def step_model(self, x, u):
        """
        Relative polar dynamics for a stationary target:
          d_dot = -v cos(a)
          a_dot = -w + (v / d) sin(a)
        where:
          d: target distance, a: target bearing in robot frame.
        """
        d = max(x[0], self.min_valid_distance)
        a = x[1]
        v = u[0]
        w = u[1]

        d_next = d + self.dt * (-v * math.cos(a))
        a_next = wrap_to_pi(a + self.dt * (-w + (v / max(d, 1e-3)) * math.sin(a)))

        d_next = max(d_next, self.min_valid_distance * 0.5)
        return np.array([d_next, a_next], dtype=float)

    def build_nominal_horizon(self, x0):
        n = self.horizon
        xr = np.zeros((n + 1, 2), dtype=float)
        ur = np.zeros((n, 2), dtype=float)
        x_nom = np.zeros((n + 1, 2), dtype=float)

        xr[:, 0] = self.desired_distance
        xr[:, 1] = 0.0
        x_nom[0, :] = x0

        for k in range(n):
            d = x_nom[k, 0]
            a = x_nom[k, 1]
            e_d = d - self.desired_distance

            v_ff = self.kff_dist * e_d * max(0.1, math.cos(a))
            if not self.allow_reverse:
                v_ff = max(v_ff, 0.0)
            w_ff = self.kff_angle * a

            ur[k, 0] = self.clip_v(v_ff)
            ur[k, 1] = self.clip_w(w_ff)
            x_nom[k + 1, :] = self.step_model(x_nom[k, :], ur[k, :])

        return xr, ur, x_nom

    def solve_mpc(self, x0):
        """
        Finite-horizon LTV-LQR on linearized relative polar model:
          x_{k+1} ≈ A_k x_k + B_k u_k
        tracking reference xr/ur generated from nominal rollout.
        """
        xr, ur, x_nom = self.build_nominal_horizon(x0)
        n = self.horizon

        Q = np.diag([self.q_dist, self.q_angle])
        R = np.diag([self.r_v, self.r_w])
        P = self.qf_scale * Q

        A_list = []
        B_list = []
        for k in range(n):
            d = max(x_nom[k, 0], self.min_valid_distance)
            a = x_nom[k, 1]
            v = ur[k, 0]

            sa = math.sin(a)
            ca = math.cos(a)

            A = np.array([
                [1.0, self.dt * (v * sa)],
                [self.dt * (-v * sa / max(d * d, 1e-6)), 1.0 + self.dt * (v * ca / max(d, 1e-3))]
            ], dtype=float)
            B = np.array([
                [-self.dt * ca, 0.0],
                [self.dt * (sa / max(d, 1e-3)), -self.dt]
            ], dtype=float)

            A_list.append(A)
            B_list.append(B)

        K_list = [None] * n
        for k in reversed(range(n)):
            A = A_list[k]
            B = B_list[k]
            G = R + B.T @ P @ B
            K = np.linalg.solve(G, B.T @ P @ A)
            K_list[k] = K
            P = Q + A.T @ P @ (A - B @ K)

        e0 = np.array([
            x0[0] - xr[0, 0],
            wrap_to_pi(x0[1] - xr[0, 1])
        ], dtype=float)

        du0 = -K_list[0] @ e0
        v_cmd = ur[0, 0] + du0[0]
        w_cmd = ur[0, 1] + du0[1]

        return self.clip_v(v_cmd), self.clip_w(w_cmd)

    def control_tick(self):
        now_ns = self.get_clock().now().nanoseconds
        with self.state_lock:
            d = self.person_distance
            a = self.person_angle
            stamp_ns = self.last_person_stamp_ns

        if d is None or a is None:
            if self.stop_on_lost_target:
                self.publish_zero()
            return

        age = (now_ns - stamp_ns) * 1e-9
        if age > self.msg_timeout:
            self.warn_throttle(
                1.0,
                f"Person measurement timeout: age={age:.3f}s > {self.msg_timeout:.3f}s, publish zero."
            )
            if self.stop_on_lost_target:
                self.publish_zero()
            return

        if self.stop_when_aligned:
            if abs(d - self.desired_distance) < self.distance_tolerance and abs(a) < self.angle_tolerance:
                self.publish_zero()
                return

        x0 = np.array([d, a], dtype=float)
        try:
            v_cmd, w_cmd = self.solve_mpc(x0)
        except np.linalg.LinAlgError:
            self.warn_throttle(1.0, "MPC solve failed (singular matrix), fallback zero cmd.")
            self.publish_zero()
            return

        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = HostMPCControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
