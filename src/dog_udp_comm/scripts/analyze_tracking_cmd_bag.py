#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Analyze alignment between `/tracking` and `/track_cmd_vel` from a rosbag2 folder.

The script reads bag messages via rosbag2_py and computes:
- frame counts and validity ratios for tracking
- nearest cmd match rate
- time offset stats between tracking and cmd
- angular command agreement with a centering controller target x=0.5
"""

import argparse
import bisect
import csv
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


@dataclass
class TrackingFrame:
    bag_time_ns: int
    x: float
    y: float
    detected: int
    ts_ms: float
    age_ms: float
    valid_for_control: bool


@dataclass
class CmdSample:
    bag_time_ns: int
    linear_x: float
    angular_z: float


@dataclass
class MatchResult:
    tracking_time_ns: int
    cmd_time_ns: int
    dt_ms: float
    x: float
    detected: int
    expected_w: float
    actual_w: float
    abs_error_w: float
    sign_match: bool


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
    if ts_raw >= 1.0e17:
        return ts_raw / 1.0e6
    if ts_raw >= 1.0e14:
        return ts_raw / 1.0e3
    if ts_raw >= 1.0e11:
        return ts_raw
    if ts_raw >= 1.0e9:
        return ts_raw * 1.0e3
    return ts_raw


def expected_angular_from_x(
    x: float,
    target_x: float,
    x_tolerance: float,
    k_angular: float,
    max_angular: float,
    reverse_angular: bool,
) -> float:
    err_x = target_x - x
    if abs(err_x) <= x_tolerance:
        return 0.0
    sign = -1.0 if reverse_angular else 1.0
    w = sign * k_angular * err_x
    return max(-max_angular, min(max_angular, w))


def quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    idx = int(round((len(values) - 1) * q))
    return sorted(values)[idx]


def open_reader_auto(uri: str, storage_hint: str) -> rosbag2_py.SequentialReader:
    errors: List[str] = []
    candidates = [storage_hint] if storage_hint != "auto" else ["sqlite3", "mcap"]
    for sid in candidates:
        try:
            reader = rosbag2_py.SequentialReader()
            storage_options = rosbag2_py.StorageOptions(uri=uri, storage_id=sid)
            converter_options = rosbag2_py.ConverterOptions("", "")
            reader.open(storage_options, converter_options)
            return reader
        except Exception as exc:
            errors.append(f"{sid}: {exc}")
    raise RuntimeError("Failed to open bag with storage ids: " + " | ".join(errors))


def read_topics(
    bag_path: str,
    tracking_topic: str,
    cmd_topic: str,
    storage_hint: str,
    max_allowed_age_ms: float,
    max_future_age_ms: float,
) -> Tuple[List[TrackingFrame], List[CmdSample], Dict[str, str]]:
    reader = open_reader_auto(bag_path, storage_hint)
    topic_info = reader.get_all_topics_and_types()
    type_map = {it.name: it.type for it in topic_info}

    if tracking_topic not in type_map:
        raise RuntimeError(f"Topic not found in bag: {tracking_topic}")
    if cmd_topic not in type_map:
        raise RuntimeError(f"Topic not found in bag: {cmd_topic}")

    tracking_type = get_message(type_map[tracking_topic])
    cmd_type = get_message(type_map[cmd_topic])

    tracking_frames: List[TrackingFrame] = []
    cmd_samples: List[CmdSample] = []

    while reader.has_next():
        topic, data, bag_time_ns = reader.read_next()
        if topic == tracking_topic:
            msg = deserialize_message(data, tracking_type)
            parsed = parse_tracking_text(msg.data)
            if parsed is None:
                continue
            x, y, detected, ts_raw = parsed
            ts_ms = normalize_timestamp_to_ms(ts_raw)
            age_ms = bag_time_ns / 1.0e6 - ts_ms
            valid_detected = (detected == 1 and 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)
            valid_age = (age_ms <= max_allowed_age_ms and age_ms >= -abs(max_future_age_ms))
            tracking_frames.append(
                TrackingFrame(
                    bag_time_ns=int(bag_time_ns),
                    x=float(x),
                    y=float(y),
                    detected=int(detected),
                    ts_ms=float(ts_ms),
                    age_ms=float(age_ms),
                    valid_for_control=bool(valid_detected and valid_age),
                )
            )
        elif topic == cmd_topic:
            msg = deserialize_message(data, cmd_type)
            cmd_samples.append(
                CmdSample(
                    bag_time_ns=int(bag_time_ns),
                    linear_x=float(msg.linear.x),
                    angular_z=float(msg.angular.z),
                )
            )
    return tracking_frames, cmd_samples, type_map


def match_tracking_cmd(
    tracking_frames: List[TrackingFrame],
    cmd_samples: List[CmdSample],
    target_x: float,
    x_tolerance: float,
    k_angular: float,
    max_angular: float,
    reverse_angular: bool,
    match_window_ms: float,
    sign_epsilon: float,
) -> Tuple[List[MatchResult], int]:
    if not cmd_samples:
        return [], 0

    cmd_times = [it.bag_time_ns for it in cmd_samples]
    matches: List[MatchResult] = []
    unmatched = 0
    win_ns = int(match_window_ms * 1.0e6)

    for trk in tracking_frames:
        if not trk.valid_for_control:
            continue
        idx = bisect.bisect_left(cmd_times, trk.bag_time_ns)
        candidates: List[int] = []
        if idx < len(cmd_times):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        if not candidates:
            unmatched += 1
            continue

        best_idx = min(candidates, key=lambda i: abs(cmd_times[i] - trk.bag_time_ns))
        dt_ns = cmd_times[best_idx] - trk.bag_time_ns
        if abs(dt_ns) > win_ns:
            unmatched += 1
            continue

        cmd = cmd_samples[best_idx]
        expected_w = expected_angular_from_x(
            trk.x, target_x, x_tolerance, k_angular, max_angular, reverse_angular
        )
        actual_w = cmd.angular_z
        if abs(expected_w) <= sign_epsilon and abs(actual_w) <= sign_epsilon:
            sign_match = True
        elif abs(expected_w) <= sign_epsilon:
            sign_match = abs(actual_w) <= sign_epsilon
        else:
            sign_match = (expected_w * actual_w) > 0.0 or abs(actual_w) <= sign_epsilon

        matches.append(
            MatchResult(
                tracking_time_ns=trk.bag_time_ns,
                cmd_time_ns=cmd.bag_time_ns,
                dt_ms=dt_ns / 1.0e6,
                x=trk.x,
                detected=trk.detected,
                expected_w=expected_w,
                actual_w=actual_w,
                abs_error_w=abs(actual_w - expected_w),
                sign_match=sign_match,
            )
        )
    return matches, unmatched


def print_summary(
    bag_path: str,
    type_map: Dict[str, str],
    tracking_topic: str,
    cmd_topic: str,
    tracking_frames: List[TrackingFrame],
    cmd_samples: List[CmdSample],
    matches: List[MatchResult],
    unmatched: int,
) -> None:
    total_tracking = len(tracking_frames)
    total_cmd = len(cmd_samples)
    total_valid = sum(1 for it in tracking_frames if it.valid_for_control)
    detected_pos = sum(1 for it in tracking_frames if it.detected == 1)
    detected_neg = sum(1 for it in tracking_frames if it.detected == -1)
    age_values = [it.age_ms for it in tracking_frames]
    dt_values = [it.dt_ms for it in matches]
    err_values = [it.abs_error_w for it in matches]
    sign_ok = sum(1 for it in matches if it.sign_match)

    print("=== Bag Basic Info ===")
    print(f"bag: {bag_path}")
    print(f"{tracking_topic}: {type_map.get(tracking_topic, 'N/A')}")
    print(f"{cmd_topic}: {type_map.get(cmd_topic, 'N/A')}")
    print()

    print("=== Tracking Stats ===")
    print(f"tracking frames: {total_tracking}")
    print(f"detected=1 frames: {detected_pos}")
    print(f"detected=-1 frames: {detected_neg}")
    print(f"valid-for-control frames: {total_valid}")
    if age_values:
        print(
            "tracking age_ms: "
            f"min={min(age_values):.2f}, p50={statistics.median(age_values):.2f}, "
            f"p95={quantile(age_values, 0.95):.2f}, max={max(age_values):.2f}"
        )
    print()

    print("=== Cmd Stats ===")
    print(f"cmd samples: {total_cmd}")
    print()

    print("=== Alignment Stats ===")
    print(f"matched frames: {len(matches)}")
    print(f"unmatched valid frames: {unmatched}")
    if total_valid > 0:
        print(f"match rate: {len(matches) / total_valid * 100.0:.2f}%")
    if dt_values:
        print(
            "dt_ms (cmd - tracking): "
            f"min={min(dt_values):.2f}, p50={statistics.median(dt_values):.2f}, "
            f"p95={quantile(dt_values, 0.95):.2f}, max={max(dt_values):.2f}"
        )
    if err_values:
        print(
            "abs_error_w: "
            f"mean={statistics.mean(err_values):.4f}, "
            f"p50={statistics.median(err_values):.4f}, "
            f"p95={quantile(err_values, 0.95):.4f}"
        )
        print(f"sign agreement: {sign_ok}/{len(matches)} ({sign_ok / len(matches) * 100.0:.2f}%)")


def dump_csv(path: str, matches: List[MatchResult]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tracking_time_ns",
                "cmd_time_ns",
                "dt_ms",
                "x",
                "detected",
                "expected_w",
                "actual_w",
                "abs_error_w",
                "sign_match",
            ]
        )
        for it in matches:
            writer.writerow(
                [
                    it.tracking_time_ns,
                    it.cmd_time_ns,
                    f"{it.dt_ms:.6f}",
                    f"{it.x:.6f}",
                    it.detected,
                    f"{it.expected_w:.6f}",
                    f"{it.actual_w:.6f}",
                    f"{it.abs_error_w:.6f}",
                    int(it.sign_match),
                ]
            )


def show_samples(matches: List[MatchResult], n: int) -> None:
    if n <= 0 or not matches:
        return
    print()
    print(f"=== First {min(n, len(matches))} matched samples ===")
    for it in matches[:n]:
        print(
            f"dt_ms={it.dt_ms:7.3f}, x={it.x:.3f}, "
            f"expected_w={it.expected_w:+.3f}, actual_w={it.actual_w:+.3f}, "
            f"err={it.abs_error_w:.3f}, sign_ok={it.sign_match}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Analyze /tracking and /track_cmd_vel alignment from rosbag2.")
    p.add_argument("bag_path", type=str, help="Path to rosbag2 folder (contains metadata.yaml).")
    p.add_argument("--storage", type=str, default="auto", choices=["auto", "sqlite3", "mcap"])
    p.add_argument("--tracking-topic", type=str, default="/tracking")
    p.add_argument("--cmd-topic", type=str, default="/track_cmd_vel")
    p.add_argument("--target-x", type=float, default=0.5)
    p.add_argument("--x-tolerance", type=float, default=0.03)
    p.add_argument("--k-angular", type=float, default=2.0)
    p.add_argument("--max-angular", type=float, default=1.2)
    p.add_argument("--reverse-angular", action="store_true")
    p.add_argument("--max-allowed-age-ms", type=float, default=500.0)
    p.add_argument("--max-future-age-ms", type=float, default=200.0)
    p.add_argument("--match-window-ms", type=float, default=150.0)
    p.add_argument("--sign-epsilon", type=float, default=0.02)
    p.add_argument("--show-samples", type=int, default=10)
    p.add_argument("--csv-out", type=str, default="")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    bag_path = str(Path(args.bag_path).expanduser().resolve())
    if not Path(bag_path).exists():
        print(f"Bag path does not exist: {bag_path}")
        return 2

    tracking_frames, cmd_samples, type_map = read_topics(
        bag_path=bag_path,
        tracking_topic=args.tracking_topic,
        cmd_topic=args.cmd_topic,
        storage_hint=args.storage,
        max_allowed_age_ms=args.max_allowed_age_ms,
        max_future_age_ms=args.max_future_age_ms,
    )

    matches, unmatched = match_tracking_cmd(
        tracking_frames=tracking_frames,
        cmd_samples=cmd_samples,
        target_x=args.target_x,
        x_tolerance=args.x_tolerance,
        k_angular=args.k_angular,
        max_angular=args.max_angular,
        reverse_angular=args.reverse_angular,
        match_window_ms=args.match_window_ms,
        sign_epsilon=args.sign_epsilon,
    )

    print_summary(
        bag_path=bag_path,
        type_map=type_map,
        tracking_topic=args.tracking_topic,
        cmd_topic=args.cmd_topic,
        tracking_frames=tracking_frames,
        cmd_samples=cmd_samples,
        matches=matches,
        unmatched=unmatched,
    )
    show_samples(matches, args.show_samples)

    if args.csv_out:
        csv_out = str(Path(args.csv_out).expanduser().resolve())
        dump_csv(csv_out, matches)
        print()
        print(f"CSV written: {csv_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
