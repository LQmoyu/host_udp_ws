#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Small pure-Python helpers shared by the person-following ROS nodes."""

from dataclasses import dataclass


@dataclass
class FarJumpDecision:
    hold: bool
    distance_for_filter: float
    pending_frames: int
    reason: str


class FarJumpSuppressor:
    """Debounce sudden far-distance jumps before allowing forward motion.

    A single bad lidar match often looks like the target suddenly moved far
    away. We hold that jump for a few frames, but accept it if it persists.
    """

    def __init__(
        self,
        enabled=True,
        jump_threshold=0.35,
        required_frames=3,
        hold_sec=0.35,
        max_forward_v=0.0,
    ):
        self.enabled = bool(enabled)
        self.jump_threshold = max(0.0, float(jump_threshold))
        self.required_frames = max(1, int(required_frames))
        self.hold_sec = max(0.0, float(hold_sec))
        self.max_forward_v = max(0.0, float(max_forward_v))
        self.pending_frames = 0
        self.hold_until_ns = 0

    def reset(self):
        self.pending_frames = 0
        self.hold_until_ns = 0

    def update(self, prev_distance, raw_distance, now_ns):
        if not self.enabled or prev_distance is None:
            self.reset()
            return FarJumpDecision(False, raw_distance, 0, "disabled_or_first")

        jump = float(raw_distance) - float(prev_distance)
        if jump <= self.jump_threshold:
            self.reset()
            return FarJumpDecision(False, raw_distance, 0, "normal")

        self.pending_frames += 1
        if self.pending_frames < self.required_frames:
            self.hold_until_ns = int(now_ns + self.hold_sec * 1e9)
            return FarJumpDecision(True, prev_distance, self.pending_frames, "far_jump_pending")

        self.reset()
        return FarJumpDecision(False, raw_distance, self.required_frames, "far_jump_confirmed")


def tracking_only_linear_limit(normal_speed, safe_speed, max_v, safe_linear=False):
    """Return the forward speed limit for vision-only tracking."""
    normal = max(0.0, float(normal_speed))
    safe = max(0.0, float(safe_speed))
    max_forward = max(0.0, float(max_v))
    limit = min(normal, safe) if safe_linear else normal
    return min(limit, max_forward)


def _make_cluster(items):
    ranges = [item[1] for item in items]
    center = sum(ranges) / max(1, len(ranges))
    return {
        "items": items,
        "count": len(items),
        "min": min(ranges),
        "max": max(ranges),
        "center": center,
        "spread": max(ranges) - min(ranges),
    }


def _pick_item(cluster, target_range=None):
    items = cluster["items"]
    if target_range is None:
        target_range = cluster["center"]
    return min(items, key=lambda item: (abs(item[1] - target_range), item[3]))


def select_stable_cluster(
    candidates,
    previous_range=None,
    cluster_tolerance=0.22,
    min_cluster_points=1,
    max_jump_up=0.35,
    max_jump_down=0.55,
    near_prefer_margin=0.20,
    background_spread=0.60,
):
    """Select a lidar candidate from a range-stable sector cluster.

    Candidate tuple format is `(idx, range, angle, angle_error)`.
    The function prefers a cluster close to the previous filtered range, with
    a safety bias toward nearer clusters when the sector also contains far
    background points.
    """

    if not candidates:
        return None, {
            "cluster_count": 0,
            "selected_reason": "empty",
            "selected_cluster_min": 0.0,
            "selected_cluster_max": 0.0,
            "selected_cluster_points": 0,
        }

    tol = max(0.01, float(cluster_tolerance))
    min_points = max(1, int(min_cluster_points))
    sorted_items = sorted(candidates, key=lambda item: item[1])
    clusters = []
    current = [sorted_items[0]]
    for item in sorted_items[1:]:
        if item[1] - current[-1][1] <= tol:
            current.append(item)
        else:
            clusters.append(_make_cluster(current))
            current = [item]
    clusters.append(_make_cluster(current))

    usable = [c for c in clusters if c["count"] >= min_points]
    if not usable:
        usable = clusters

    raw_min = sorted_items[0][1]
    raw_max = sorted_items[-1][1]
    sector_spread = raw_max - raw_min
    nearest_cluster = min(usable, key=lambda c: (c["min"], -c["count"]))

    reason = "nearest_no_previous"
    selected_cluster = nearest_cluster
    target = selected_cluster["center"]

    if previous_range is not None:
        prev = float(previous_range)
        near_margin = max(0.0, float(near_prefer_margin))
        max_up = max(0.0, float(max_jump_up))
        max_down = max(0.0, float(max_jump_down))
        spread_gate = max(0.0, float(background_spread))

        near_is_safe = nearest_cluster["center"] <= prev + near_margin
        has_background_spread = sector_spread >= spread_gate
        if has_background_spread and near_is_safe:
            selected_cluster = nearest_cluster
            reason = "background_near"
            target = min(prev, selected_cluster["center"])
        else:
            stable = []
            for cluster in usable:
                delta = cluster["center"] - prev
                if delta >= 0.0 and delta <= max_up:
                    stable.append(cluster)
                elif delta < 0.0 and abs(delta) <= max_down:
                    stable.append(cluster)

            if stable:
                selected_cluster = min(stable, key=lambda c: (abs(c["center"] - prev), -c["count"]))
                reason = "stable"
                target = prev
            elif nearest_cluster["center"] < prev:
                selected_cluster = nearest_cluster
                reason = "near_safety"
                target = selected_cluster["center"]
            else:
                selected_cluster = min(usable, key=lambda c: (abs(c["center"] - prev), -c["count"]))
                reason = "unconfirmed_far"
                target = min(selected_cluster["center"], prev)

    selected = _pick_item(selected_cluster, target)
    info = {
        "cluster_count": len(clusters),
        "selected_reason": reason,
        "selected_cluster_min": selected_cluster["min"],
        "selected_cluster_max": selected_cluster["max"],
        "selected_cluster_points": selected_cluster["count"],
    }
    return selected, info
