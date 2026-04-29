#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Host-side MPC tracker for TITA/D1 external control.

Design target:
- Runs on external IPC/main-controller (ROS2).
- Does NOT change robot-side environment.
- Publishes /track_cmd_vel for existing sender_node.cpp.

Controller:
- Finite-horizon linearized MPC (LTV-LQR form, unconstrained QP closed-form).
- Receding horizon: solve every control tick, apply first control only.
- Input saturation applied after solve.
"""

import math
from threading import Lock

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path


def wrap_to_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quat_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class HostMPCControllerNode(Node):
    def __init__(self):
        super().__init__("host_mpc_controller")

        self.declare_parameter("odom_topic", "/localization/odom")
        self.declare_parameter("path_topic", "/reference_path")
        self.declare_parameter("cmd_topic", "/track_cmd_vel")
        self.declare_parameter("control_hz", 30.0)
        # dt <= 0 means auto-select dt from control_hz
        self.declare_parameter("dt", 0.0)
        self.declare_parameter("horizon", 12)
        self.declare_parameter("path_step", 3)
        self.declare_parameter("v_ref", 0.5)
        self.declare_parameter("max_v", 1.0)
        self.declare_parameter("max_w", 1.5)
        self.declare_parameter("q_x", 8.0)
        self.declare_parameter("q_y", 12.0)
        self.declare_parameter("q_yaw", 5.0)
        self.declare_parameter("r_v", 0.4)
        self.declare_parameter("r_w", 0.2)
        self.declare_parameter("qf_scale", 4.0)
        self.declare_parameter("goal_tolerance", 0.15)
        self.declare_parameter("stop_when_goal_reached", True)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.path_topic = str(self.get_parameter("path_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.control_hz = float(self.get_parameter("control_hz").value)
        dt_param = float(self.get_parameter("dt").value)
        self.dt = dt_param if dt_param > 0.0 else (1.0 / max(self.control_hz, 1e-3))
        self.horizon = int(self.get_parameter("horizon").value)
        self.path_step = int(self.get_parameter("path_step").value)

        self.v_ref = float(self.get_parameter("v_ref").value)
        self.max_v = float(self.get_parameter("max_v").value)
        self.max_w = float(self.get_parameter("max_w").value)

        self.q_x = float(self.get_parameter("q_x").value)
        self.q_y = float(self.get_parameter("q_y").value)
        self.q_yaw = float(self.get_parameter("q_yaw").value)
        self.r_v = float(self.get_parameter("r_v").value)
        self.r_w = float(self.get_parameter("r_w").value)
        self.qf_scale = float(self.get_parameter("qf_scale").value)

        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.stop_when_goal_reached = bool(self.get_parameter("stop_when_goal_reached").value)

        self.pose = None  # (x, y, yaw)
        self.path = []    # list[(x, y, yaw)]
        self.last_nearest = 0
        self.state_lock = Lock()
        self.last_warn_ns = 0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 1)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 10)
        self.create_subscription(Path, self.path_topic, self.path_cb, 3)

        self.timer = self.create_timer(1.0 / max(self.control_hz, 1e-3), self.control_tick)

        self.get_logger().info(
            f"Host MPC ready. odom={self.odom_topic} path={self.path_topic} "
            f"cmd={self.cmd_topic} hz={self.control_hz:.1f} N={self.horizon} dt={self.dt:.3f}"
        )

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        yaw = quat_to_yaw(msg.pose.pose.orientation)
        with self.state_lock:
            self.pose = (p.x, p.y, yaw)

    def path_cb(self, msg):
        poses = msg.poses
        if len(poses) == 0:
            with self.state_lock:
                self.path = []
                self.last_nearest = 0
            return

        out = []
        for i, ps in enumerate(poses):
            pose = ps.pose
            x = pose.position.x
            y = pose.position.y
            # Path orientation may be missing; derive from next point when needed.
            q = pose.orientation
            qnorm = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
            if qnorm > 1e-6:
                yaw = quat_to_yaw(q)
            elif i + 1 < len(poses):
                nx = poses[i + 1].pose.position.x
                ny = poses[i + 1].pose.position.y
                yaw = math.atan2(ny - y, nx - x)
            elif i > 0:
                px = poses[i - 1].pose.position.x
                py = poses[i - 1].pose.position.y
                yaw = math.atan2(y - py, x - px)
            else:
                yaw = 0.0
            out.append((x, y, yaw))

        with self.state_lock:
            self.path = out
            self.last_nearest = 0
        self.get_logger().info(f"New reference path received. points={len(out)}")

    def publish_zero(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_pub.publish(msg)

    def warn_throttle(self, period_sec, text):
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self.last_warn_ns >= int(period_sec * 1e9):
            self.get_logger().warn(text)
            self.last_warn_ns = now_ns

    def find_nearest_index(self, x, y, path):
        if not path:
            return 0
        start = max(0, self.last_nearest - 20)
        end = min(len(path), self.last_nearest + 80)
        if end <= start:
            start = 0
            end = len(path)

        best_i = start
        best_d2 = 1e18
        for i in range(start, end):
            dx = x - path[i][0]
            dy = y - path[i][1]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        self.last_nearest = best_i
        return best_i

    def build_reference_horizon(self, nearest_i, path, pose):
        n = self.horizon
        xr = np.zeros((n + 1, 3), dtype=float)
        ur = np.zeros((n, 2), dtype=float)  # [v, w]

        for k in range(n + 1):
            idx = min(nearest_i + k * self.path_step, len(path) - 1)
            xr[k, 0] = path[idx][0]
            xr[k, 1] = path[idx][1]
            xr[k, 2] = path[idx][2]

        for k in range(n):
            dyaw = wrap_to_pi(xr[k + 1, 2] - xr[k, 2])
            ur[k, 0] = self.v_ref
            ur[k, 1] = dyaw / max(self.dt, 1e-3)

            ur[k, 0] = float(np.clip(ur[k, 0], -self.max_v, self.max_v))
            ur[k, 1] = float(np.clip(ur[k, 1], -self.max_w, self.max_w))

        # Near goal, smoothly reduce reference speed.
        goal_x, goal_y, _ = path[-1]
        if pose is not None:
            dxg = goal_x - pose[0]
            dyg = goal_y - pose[1]
            dg = math.hypot(dxg, dyg)
            if dg < 1.0:
                scale = max(0.0, min(1.0, dg / 1.0))
                ur[:, 0] *= scale

        return xr, ur

    def solve_mpc(self, x0, xr, ur):
        """
        Solve linearized finite-horizon tracking MPC:
          e_{k+1} = A_k e_k + B_k du_k
          du_k = -K_k e_k (Riccati recursion)
          u_k = u_ref_k + du_k
        """
        n = self.horizon
        Q = np.diag([self.q_x, self.q_y, self.q_yaw])
        R = np.diag([self.r_v, self.r_w])
        P = self.qf_scale * Q

        A_list = []
        B_list = []
        for k in range(n):
            yaw_r = xr[k, 2]
            v_r = ur[k, 0]
            c = math.cos(yaw_r)
            s = math.sin(yaw_r)
            A = np.array([
                [1.0, 0.0, -v_r * s * self.dt],
                [0.0, 1.0,  v_r * c * self.dt],
                [0.0, 0.0, 1.0]
            ], dtype=float)
            B = np.array([
                [c * self.dt, 0.0],
                [s * self.dt, 0.0],
                [0.0, self.dt]
            ], dtype=float)
            A_list.append(A)
            B_list.append(B)

        K_list = [None] * n
        for k in reversed(range(n)):
            A = A_list[k]
            B = B_list[k]
            BtPB = B.T @ P @ B
            G = R + BtPB
            K = np.linalg.solve(G, B.T @ P @ A)
            K_list[k] = K
            P = Q + A.T @ P @ (A - B @ K)

        e = np.array([
            x0[0] - xr[0, 0],
            x0[1] - xr[0, 1],
            wrap_to_pi(x0[2] - xr[0, 2])
        ], dtype=float)

        du0 = -K_list[0] @ e
        v_cmd = ur[0, 0] + du0[0]
        w_cmd = ur[0, 1] + du0[1]

        v_cmd = float(np.clip(v_cmd, -self.max_v, self.max_v))
        w_cmd = float(np.clip(w_cmd, -self.max_w, self.max_w))
        return v_cmd, w_cmd

    def control_tick(self):
        with self.state_lock:
            pose = self.pose
            path = list(self.path)

        if pose is None or len(path) < 2:
            self.publish_zero()
            return

        x, y, yaw = pose
        nearest = self.find_nearest_index(x, y, path)

        goal_x, goal_y, goal_yaw = path[-1]
        dist_goal = math.hypot(goal_x - x, goal_y - y)
        yaw_goal_err = abs(wrap_to_pi(goal_yaw - yaw))
        if self.stop_when_goal_reached and dist_goal < self.goal_tolerance and yaw_goal_err < 0.35:
            self.publish_zero()
            return

        xr, ur = self.build_reference_horizon(nearest, path, pose)
        try:
            v_cmd, w_cmd = self.solve_mpc((x, y, yaw), xr, ur)
        except np.linalg.LinAlgError:
            self.warn_throttle(1.0, "MPC solver numerically unstable, fallback zero cmd.")
            self.publish_zero()
            return

        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)


if __name__ == "__main__":
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
