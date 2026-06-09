import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from person_follow_utils import FarJumpSuppressor, select_stable_cluster


def test_stable_cluster_prefers_near_cluster_over_background_jump():
    candidates = [
        (10, 1.24, 0.00, 0.02),
        (11, 1.31, 0.02, 0.01),
        (12, 1.42, 0.03, 0.02),
        (13, 2.30, 0.04, 0.03),
        (14, 2.65, 0.05, 0.04),
    ]

    selected, info = select_stable_cluster(
        candidates,
        previous_range=1.34,
        cluster_tolerance=0.22,
        min_cluster_points=1,
        max_jump_up=0.35,
        max_jump_down=0.55,
        near_prefer_margin=0.20,
        background_spread=0.60,
    )

    assert selected[1] < 1.5
    assert info["cluster_count"] >= 2
    assert info["selected_reason"] in ("stable", "near_safety", "background_near")


def test_stable_cluster_accepts_consistent_far_target_after_previous_far_range():
    candidates = [
        (20, 2.10, 0.00, 0.03),
        (21, 2.18, 0.01, 0.02),
        (22, 2.26, 0.02, 0.01),
    ]

    selected, info = select_stable_cluster(
        candidates,
        previous_range=2.15,
        cluster_tolerance=0.22,
        min_cluster_points=1,
        max_jump_up=0.35,
        max_jump_down=0.55,
        near_prefer_margin=0.20,
        background_spread=0.60,
    )

    assert 2.0 < selected[1] < 2.4
    assert info["selected_reason"] == "stable"


def test_far_jump_suppressor_holds_until_distance_jump_is_confirmed():
    suppressor = FarJumpSuppressor(
        enabled=True,
        jump_threshold=0.35,
        required_frames=3,
        hold_sec=0.40,
        max_forward_v=0.0,
    )

    now_ns = 1_000_000_000
    raw_distances = [2.25, 2.30, 2.28]
    decisions = [
        suppressor.update(prev_distance=1.35, raw_distance=d, now_ns=now_ns + i * 70_000_000)
        for i, d in enumerate(raw_distances)
    ]

    assert decisions[0].hold is True
    assert decisions[1].hold is True
    assert decisions[2].hold is False
    assert decisions[0].distance_for_filter == 1.35
    assert decisions[2].distance_for_filter == raw_distances[2]


def test_far_jump_suppressor_does_not_block_small_distance_increase():
    suppressor = FarJumpSuppressor(enabled=True, jump_threshold=0.35, required_frames=3)
    decision = suppressor.update(prev_distance=1.35, raw_distance=1.50, now_ns=1_000_000_000)
    assert decision.hold is False
    assert decision.distance_for_filter == 1.50


if __name__ == "__main__":
    tests = [name for name in globals() if name.startswith("test_")]
    for name in tests:
        globals()[name]()
    print(f"PASS {len(tests)} tests")
