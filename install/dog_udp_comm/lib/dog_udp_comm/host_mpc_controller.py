#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Host-side person-following MPC controller (ROS2).

Input:
- geometry_msgs/Vector3Stamped on `person_topic`
  - vector.x: person distance in meters
  - vector.y: person angle in radians (left positive)
- geometry_msgs/Vector3Stamped on optional `tracking_state_topic`
  - vector.x: normalized tracking x in [0,1] when detected==1
  - vector.z: detected flag (1 or -1), used for loss-state handling

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
        self.declare_parameter("tracking_state_topic", "/tracking_state")
        self.declare_parameter("cmd_topic", "/track_cmd_vel")

        self.declare_parameter("control_hz", 30.0)
        self.declare_parameter("dt", 0.0)  # <=0 means use 1/control_hz
        self.declare_parameter("horizon", 12)
        self.declare_parameter("msg_timeout", 1.0)
        self.declare_parameter("tracking_state_timeout_sec", 1.0)
        self.declare_parameter("require_first_tracking_frame", True)

        self.declare_parameter("desired_distance", 1.2)
        self.declare_parameter("desired_angle", 0.0)
        self.declare_parameter("distance_tolerance", 0.08)
        self.declare_parameter("angle_tolerance", 0.08)
        self.declare_parameter("stop_when_aligned", True)
        self.declare_parameter("stop_on_lost_target", True)
        self.declare_parameter("allow_reverse", True)
        self.declare_parameter("target_lost_duration_sec", 1.0)

        self.declare_parameter("max_v", 0.35)
        self.declare_parameter("max_w", 5.0)
        self.declare_parameter("max_reverse_v", 0.15)
        self.declare_parameter("force_zero_linear_velocity", False)
        self.declare_parameter("reverse_angular_output", False)

        self.declare_parameter("q_dist", 16.0)
        self.declare_parameter("q_angle", 35.0)
        self.declare_parameter("r_v", 0.75)
        self.declare_parameter("r_w", 0.10)
        self.declare_parameter("qf_scale", 4.0)

        self.declare_parameter("kff_dist", 0.45)
        self.declare_parameter("kff_angle", 3.0)
        self.declare_parameter("use_tracking_x_for_angle", True)
        self.declare_parameter("tracking_x_target", 0.5)
        self.declare_parameter("tracking_x_deadband", 0.003)
        self.declare_parameter("tracking_x_timeout_sec", 1.0)
        self.declare_parameter("enable_angle_prediction", True)
        self.declare_parameter("max_prediction_sec", 0.35)
        self.declare_parameter("angle_rate_filter_alpha", 0.6)
        self.declare_parameter("max_angle_rate", 3.5)
        self.declare_parameter("enable_tracking_only_fallback", True)
        self.declare_parameter("tracking_only_linear_speed", 0.08)
        self.declare_parameter("tracking_only_kp_w", 3.0)
        self.declare_parameter("tracking_only_kd_w", 0.4)
        self.declare_parameter("image_width", 640.0)
        self.declare_parameter("fx", 600.0)
        self.declare_parameter("cx", 320.0)
        self.declare_parameter("yaw_cam_to_control", 0.0)
        self.declare_parameter("tracking_angle_gain", 2.0)
        self.declare_parameter("distance_filter_alpha", 0.65)
        self.declare_parameter("angle_filter_alpha", 0.65)
        self.declare_parameter("min_valid_distance", 0.15)
        self.declare_parameter("max_valid_distance", 8.0)
        self.declare_parameter("enable_distance_speed_profile", True)
        self.declare_parameter("reverse_distance_threshold", 0.9)
        self.declare_parameter("stop_reverse_distance", 1.05)
        self.declare_parameter("speed_profile_near_distance", 1.3)
        self.declare_parameter("speed_profile_far_distance", 3.2)
        self.declare_parameter("speed_profile_min_v", 0.04)
        self.declare_parameter("speed_profile_max_v", 0.35)
        self.declare_parameter("angle_priority_threshold", 0.12)
        self.declare_parameter("min_heading_speed_scale", 0.05)
        self.declare_parameter("max_accel", 0.45)
        self.declare_parameter("max_w_accel", 8.0)

        self.person_topic = str(self.get_parameter("person_topic").value)
        self.tracking_state_topic = str(self.get_parameter("tracking_state_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)

        self.control_hz = float(self.get_parameter("control_hz").value)
        dt_param = float(self.get_parameter("dt").value)
        self.dt = dt_param if dt_param > 0.0 else (1.0 / max(self.control_hz, 1e-3))
        self.horizon = int(self.get_parameter("horizon").value)
        self.msg_timeout = float(self.get_parameter("msg_timeout").value)
        self.tracking_state_timeout_sec = float(self.get_parameter("tracking_state_timeout_sec").value)
        self.require_first_tracking_frame = bool(
            self.get_parameter("require_first_tracking_frame").value
        )

        self.desired_distance = float(self.get_parameter("desired_distance").value)
        self.desired_angle = float(self.get_parameter("desired_angle").value)
        self.distance_tolerance = float(self.get_parameter("distance_tolerance").value)
        self.angle_tolerance = float(self.get_parameter("angle_tolerance").value)
        self.stop_when_aligned = bool(self.get_parameter("stop_when_aligned").value)
        self.stop_on_lost_target = bool(self.get_parameter("stop_on_lost_target").value)
        self.allow_reverse = bool(self.get_parameter("allow_reverse").value)
        self.target_lost_duration_sec = float(self.get_parameter("target_lost_duration_sec").value)

        self.max_v = float(self.get_parameter("max_v").value)
        self.max_w = float(self.get_parameter("max_w").value)
        self.max_reverse_v = float(self.get_parameter("max_reverse_v").value)
        self.force_zero_linear_velocity = bool(
            self.get_parameter("force_zero_linear_velocity").value
        )
        self.reverse_angular_output = bool(
            self.get_parameter("reverse_angular_output").value
        )

        self.q_dist = float(self.get_parameter("q_dist").value)
        self.q_angle = float(self.get_parameter("q_angle").value)
        self.r_v = float(self.get_parameter("r_v").value)
        self.r_w = float(self.get_parameter("r_w").value)
        self.qf_scale = float(self.get_parameter("qf_scale").value)

        self.kff_dist = float(self.get_parameter("kff_dist").value)
        self.kff_angle = float(self.get_parameter("kff_angle").value)
        self.use_tracking_x_for_angle = bool(self.get_parameter("use_tracking_x_for_angle").value)
        self.tracking_x_target = float(self.get_parameter("tracking_x_target").value)
        self.tracking_x_deadband = max(0.0, float(self.get_parameter("tracking_x_deadband").value))
        self.tracking_x_timeout_sec = max(0.01, float(self.get_parameter("tracking_x_timeout_sec").value))
        self.enable_angle_prediction = bool(self.get_parameter("enable_angle_prediction").value)
        self.max_prediction_sec = max(0.0, float(self.get_parameter("max_prediction_sec").value))
        self.angle_rate_filter_alpha = float(
            np.clip(float(self.get_parameter("angle_rate_filter_alpha").value), 0.0, 1.0)
        )
        self.max_angle_rate = max(0.0, float(self.get_parameter("max_angle_rate").value))
        self.enable_tracking_only_fallback = bool(
            self.get_parameter("enable_tracking_only_fallback").value
        )
        self.tracking_only_linear_speed = max(
            0.0, float(self.get_parameter("tracking_only_linear_speed").value)
        )
        self.tracking_only_kp_w = float(self.get_parameter("tracking_only_kp_w").value)
        self.tracking_only_kd_w = float(self.get_parameter("tracking_only_kd_w").value)
        self.image_width = max(2.0, float(self.get_parameter("image_width").value))
        self.fx = max(1e-6, float(self.get_parameter("fx").value))
        self.cx = float(self.get_parameter("cx").value)
        self.yaw_cam_to_control = float(self.get_parameter("yaw_cam_to_control").value)
        self.tracking_angle_gain = max(0.1, float(self.get_parameter("tracking_angle_gain").value))
        self.distance_filter_alpha = float(self.get_parameter("distance_filter_alpha").value)
        self.angle_filter_alpha = float(self.get_parameter("angle_filter_alpha").value)
        self.min_valid_distance = float(self.get_parameter("min_valid_distance").value)
        self.max_valid_distance = float(self.get_parameter("max_valid_distance").value)
        self.enable_distance_speed_profile = bool(self.get_parameter("enable_distance_speed_profile").value)
        self.reverse_distance_threshold = float(self.get_parameter("reverse_distance_threshold").value)
        self.stop_reverse_distance = float(self.get_parameter("stop_reverse_distance").value)
        self.speed_profile_near_distance = float(self.get_parameter("speed_profile_near_distance").value)
        self.speed_profile_far_distance = float(self.get_parameter("speed_profile_far_distance").value)
        self.speed_profile_min_v = float(self.get_parameter("speed_profile_min_v").value)
        self.speed_profile_max_v = float(self.get_parameter("speed_profile_max_v").value)
        self.angle_priority_threshold = max(0.0, float(self.get_parameter("angle_priority_threshold").value))
        self.min_heading_speed_scale = float(self.get_parameter("min_heading_speed_scale").value)
        self.max_accel = max(0.0, float(self.get_parameter("max_accel").value))
        self.max_w_accel = max(0.0, float(self.get_parameter("max_w_accel").value))

        self.speed_profile_far_distance = max(
            self.speed_profile_far_distance, self.speed_profile_near_distance + 1e-3
        )
        self.speed_profile_min_v = max(0.0, self.speed_profile_min_v)
        self.speed_profile_max_v = max(self.speed_profile_min_v, self.speed_profile_max_v)
        self.min_heading_speed_scale = float(np.clip(self.min_heading_speed_scale, 0.0, 1.0))

        self.state_lock = Lock()
        self.person_distance = None
        self.person_angle = None
        self.latest_tracking_x = None
        self.latest_tracking_x_stamp_ns = 0
        self.latest_tracking_angle = None
        self.latest_tracking_angle_rate = 0.0
        self.latest_tracking_angle_stamp_ns = 0
        self.last_person_stamp_ns = 0
        self.have_first_person_frame = False
        self.have_first_tracking_frame = False
        self.last_detected_flag = -1
        self.last_detected_stamp_ns = 0
        self.continuous_lost_since_ns = 0
        self.last_warn_ns = 0
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 1)
        self.create_subscription(Vector3Stamped, self.person_topic, self.person_cb, 10)
        self.create_subscription(Vector3Stamped, self.tracking_state_topic, self.tracking_state_cb, 10)
        self.timer = self.create_timer(1.0 / max(self.control_hz, 1e-3), self.control_tick)

        self.get_logger().info(
            "Host person-follow MPC ready. "
            f"person_topic={self.person_topic} tracking_state_topic={self.tracking_state_topic} "
            f"cmd_topic={self.cmd_topic} "
            f"hz={self.control_hz:.1f} dt={self.dt:.3f} N={self.horizon} "
            f"d_ref={self.desired_distance:.2f} a_ref={self.desired_angle:.2f} "
            f"tracking_timeout={self.tracking_state_timeout_sec:.3f}s"
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
        self.last_cmd_v = 0.0
        self.last_cmd_w = 0.0

    def clip_v(self, v):
        vmax = self.max_v
        if self.allow_reverse:
            vmin = -abs(self.max_reverse_v)
        else:
            vmin = 0.0
        return float(np.clip(v, vmin, vmax))

    def clip_w(self, w):
        return float(np.clip(w, -self.max_w, self.max_w))

    def map_output_w(self, w_cmd):
        if self.reverse_angular_output:
            w_cmd = -w_cmd
        return self.clip_w(w_cmd)

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
            self.have_first_person_frame = True

    def tracking_state_cb(self, msg):
        detected_raw = float(msg.vector.z)
        if detected_raw not in (-1.0, 1.0):
            self.warn_throttle(1.0, f"Ignore tracking_state with invalid detected={detected_raw}")
            return

        now_ns = self.get_clock().now().nanoseconds
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        if stamp_ns <= 0:
            stamp_ns = now_ns

        detected = int(detected_raw)
        x_norm = float(msg.vector.x)
        with self.state_lock:
            self.have_first_tracking_frame = True
            self.last_detected_flag = detected
            self.last_detected_stamp_ns = stamp_ns
            if detected == 1 and 0.0 <= x_norm <= 1.0:
                a_new = self.tracking_x_to_angle(x_norm)
                if self.latest_tracking_angle is not None and self.latest_tracking_angle_stamp_ns > 0:
                    dt = (stamp_ns - self.latest_tracking_angle_stamp_ns) * 1e-9
                    if dt > 1e-3:
                        rate_raw = wrap_to_pi(a_new - self.latest_tracking_angle) / dt
                        if self.max_angle_rate > 0.0:
                            rate_raw = float(np.clip(rate_raw, -self.max_angle_rate, self.max_angle_rate))
                        alpha = self.angle_rate_filter_alpha
                        self.latest_tracking_angle_rate = (
                            alpha * self.latest_tracking_angle_rate + (1.0 - alpha) * rate_raw
                        )
                else:
                    self.latest_tracking_angle_rate = 0.0

                self.latest_tracking_x = x_norm
                self.latest_tracking_x_stamp_ns = stamp_ns
                self.latest_tracking_angle = a_new
                self.latest_tracking_angle_stamp_ns = stamp_ns
            if detected == 1:
                self.continuous_lost_since_ns = 0
            elif self.continuous_lost_since_ns == 0:
                self.continuous_lost_since_ns = now_ns

    def tracking_x_to_angle(self, x_norm):
        x_norm = float(np.clip(x_norm, 0.0, 1.0))
        if abs(x_norm - self.tracking_x_target) <= self.tracking_x_deadband:
            return self.desired_angle

        u = x_norm * max(self.image_width - 1.0, 1.0)
        theta_cam = self.tracking_angle_gain * math.atan2((u - self.cx), self.fx)
        return wrap_to_pi(theta_cam + self.yaw_cam_to_control)

    def predict_tracking_angle(self, now_ns, angle, angle_rate, stamp_ns):
        if angle is None or stamp_ns <= 0:
            return None, 0.0, 0.0

        age = max(0.0, (now_ns - stamp_ns) * 1e-9)
        if not self.enable_angle_prediction:
            return angle, angle_rate, age

        pred_dt = min(age, self.max_prediction_sec)
        return wrap_to_pi(angle + angle_rate * pred_dt), angle_rate, age

    def accel_limit_w(self, w_cmd):
        if self.max_w_accel <= 1e-6:
            self.last_cmd_w = w_cmd
            return w_cmd

        dw_lim = self.max_w_accel * self.dt
        w_out = float(np.clip(w_cmd, self.last_cmd_w - dw_lim, self.last_cmd_w + dw_lim))
        self.last_cmd_w = w_out
        return w_out

    def publish_tracking_only_cmd(self, angle, angle_rate):
        w_cmd = (
            self.tracking_only_kp_w * wrap_to_pi(angle - self.desired_angle)
            + self.tracking_only_kd_w * angle_rate
        )
        w_cmd = self.map_output_w(w_cmd)
        w_cmd = self.accel_limit_w(w_cmd)
        v_cmd = min(self.tracking_only_linear_speed, self.max_v)
        v_cmd = v_cmd * self.heading_speed_scale(angle - self.desired_angle)
        v_cmd = self.accel_limit_v(v_cmd)
        if self.force_zero_linear_velocity:
            v_cmd = 0.0
            self.last_cmd_v = 0.0

        cmd = Twist()
        cmd.linear.x = float(v_cmd)
        cmd.angular.z = float(w_cmd)
        self.cmd_pub.publish(cmd)

    def distance_speed_limit(self, distance):
        if not self.enable_distance_speed_profile:
            return self.max_v

        near = self.speed_profile_near_distance
        far = self.speed_profile_far_distance
        d = max(distance, 0.0)
        if d <= near:
            return min(self.max_v, self.speed_profile_min_v)
        if d >= far:
            return min(self.max_v, self.speed_profile_max_v)

        ratio = (d - near) / max(far - near, 1e-6)
        v_lim = self.speed_profile_min_v + ratio * (self.speed_profile_max_v - self.speed_profile_min_v)
        return min(self.max_v, max(0.0, v_lim))

    def apply_near_distance_reverse(self, v_cmd, distance, angle):
        if not self.allow_reverse:
            return v_cmd
        if abs(wrap_to_pi(angle - self.desired_angle)) > 0.5:
            return v_cmd
        if distance >= self.stop_reverse_distance:
            return v_cmd

        if distance <= self.reverse_distance_threshold:
            return -abs(self.max_reverse_v)

        ratio = (self.stop_reverse_distance - distance) / max(
            self.stop_reverse_distance - self.reverse_distance_threshold, 1e-6
        )
        return min(v_cmd, -abs(self.max_reverse_v) * float(np.clip(ratio, 0.0, 1.0)))

    def heading_speed_scale(self, angle_error):
        if self.angle_priority_threshold <= 1e-6:
            return 1.0
        a = abs(wrap_to_pi(angle_error))
        if a <= self.angle_priority_threshold:
            return 1.0

        max_angle = math.pi * 0.5
        if a >= max_angle:
            return self.min_heading_speed_scale

        ratio = (a - self.angle_priority_threshold) / max(max_angle - self.angle_priority_threshold, 1e-6)
        scale = 1.0 - ratio * (1.0 - self.min_heading_speed_scale)
        return float(np.clip(scale, self.min_heading_speed_scale, 1.0))

    def accel_limit_v(self, v_cmd):
        if self.max_accel <= 1e-6:
            self.last_cmd_v = v_cmd
            return v_cmd

        dv_lim = self.max_accel * self.dt
        v_out = float(np.clip(v_cmd, self.last_cmd_v - dv_lim, self.last_cmd_v + dv_lim))
        self.last_cmd_v = v_out
        return v_out

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
        xr[:, 1] = self.desired_angle
        x_nom[0, :] = x0

        for k in range(n):
            d = x_nom[k, 0]
            a = x_nom[k, 1]
            e_d = d - self.desired_distance
            e_a = wrap_to_pi(a - self.desired_angle)

            v_ff = self.kff_dist * e_d * max(0.1, math.cos(a))
            if not self.allow_reverse:
                v_ff = max(v_ff, 0.0)
            w_ff = self.kff_angle * e_a

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
            a_scan = self.person_angle
            x_track = self.latest_tracking_x
            x_track_stamp_ns = self.latest_tracking_x_stamp_ns
            tracking_angle = self.latest_tracking_angle
            tracking_angle_rate = self.latest_tracking_angle_rate
            tracking_angle_stamp_ns = self.latest_tracking_angle_stamp_ns
            stamp_ns = self.last_person_stamp_ns
            have_first_frame = self.have_first_person_frame
            have_first_tracking_frame = self.have_first_tracking_frame
            detected_flag = self.last_detected_flag
            detected_stamp_ns = self.last_detected_stamp_ns
            lost_since_ns = self.continuous_lost_since_ns

        predicted_angle, predicted_angle_rate, predicted_age = self.predict_tracking_angle(
            now_ns, tracking_angle, tracking_angle_rate, tracking_angle_stamp_ns
        )
        tracking_available = (
            detected_flag == 1
            and predicted_angle is not None
            and predicted_age <= self.tracking_x_timeout_sec
        )

        if self.require_first_tracking_frame and not have_first_tracking_frame:
            self.warn_throttle(1.0, "Waiting for first tracking_state frame before starting control.")
            if self.stop_on_lost_target:
                self.publish_zero()
            return

        if not have_first_frame:
            if self.enable_tracking_only_fallback and tracking_available:
                self.publish_tracking_only_cmd(predicted_angle, predicted_angle_rate)
                return
            self.warn_throttle(1.0, "Waiting for first person frame before starting control.")
            if self.stop_on_lost_target:
                self.publish_zero()
            return

        if detected_stamp_ns > 0:
            detect_age = (now_ns - detected_stamp_ns) * 1e-9
            if detect_age > self.tracking_state_timeout_sec:
                self.warn_throttle(
                    1.0,
                    f"Tracking state timeout: age={detect_age:.3f}s > "
                    f"{self.tracking_state_timeout_sec:.3f}s, publish zero."
                )
                if self.stop_on_lost_target:
                    self.publish_zero()
                return

        if detected_flag == -1 and lost_since_ns > 0:
            lost_for = (now_ns - lost_since_ns) * 1e-9
            if lost_for > self.target_lost_duration_sec:
                self.warn_throttle(
                    1.0,
                    f"Target lost continuously for {lost_for:.2f}s (>{self.target_lost_duration_sec:.2f}s), publish zero."
                )
                if self.stop_on_lost_target:
                    self.publish_zero()
                return

        if d is None or a_scan is None:
            if self.enable_tracking_only_fallback and tracking_available:
                self.publish_tracking_only_cmd(predicted_angle, predicted_angle_rate)
                return
            if self.stop_on_lost_target:
                self.publish_zero()
            return

        a = a_scan
        if self.use_tracking_x_for_angle and tracking_available:
            a = predicted_angle

        age = (now_ns - stamp_ns) * 1e-9
        if age > self.msg_timeout:
            if self.enable_tracking_only_fallback and tracking_available:
                self.publish_tracking_only_cmd(predicted_angle, predicted_angle_rate)
                return
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

        if v_cmd > 0.0:
            v_lim_distance = self.distance_speed_limit(d)
            v_lim_heading = v_lim_distance * self.heading_speed_scale(a - self.desired_angle)
            v_cmd = min(v_cmd, v_lim_heading)
        v_cmd = self.apply_near_distance_reverse(v_cmd, d, a)

        v_cmd = self.accel_limit_v(v_cmd)
        if self.force_zero_linear_velocity:
            v_cmd = 0.0
            self.last_cmd_v = 0.0
        w_cmd = self.map_output_w(w_cmd)
        w_cmd = self.accel_limit_w(w_cmd)

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
