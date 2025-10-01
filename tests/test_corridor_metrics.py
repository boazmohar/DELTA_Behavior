from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from behavioral_analysis.analysis.corridor_metrics import (
    HitRateBin,
    LickRateBin,
    aggregate_hit_rates,
    compute_hit_rates_by_position,
    compute_lick_rates_by_position,
    enumerate_position_bins,
)


def make_cue_result(position: float, rewarded: bool) -> dict:
    return {"position": position, "hasGivenReward": rewarded}


def make_entry(msg: str, data: dict | None = None) -> dict:
    payload = {"msg": msg}
    if data is not None:
        payload["data"] = data
    return payload


def test_compute_hit_rates_by_position_groups_by_bin() -> None:
    entries = [
        make_entry("Cue Result", make_cue_result(1050, True)),
        make_entry("Cue Result", make_cue_result(1950, False)),
        make_entry("Cue Result", make_cue_result(1800, True)),
        make_entry("Cue Result", make_cue_result(980, False)),
        make_entry("Cue Result", make_cue_result(1710, True)),
    ]

    results = compute_hit_rates_by_position(entries, bin_size=1000)

    assert results == [
        HitRateBin(bin_start=0, hits=0, total=1, hit_rate=0.0),
        HitRateBin(
            bin_start=1000,
            hits=3,
            total=4,
            hit_rate=pytest.approx(0.75),
        ),
    ]


def test_aggregate_hit_rates_ignores_invalid_positions() -> None:
    cue_results = [
        make_cue_result("not-a-number", True),
        make_cue_result(500, True),
        {"position": None, "hasGivenReward": False},
    ]

    aggregated = aggregate_hit_rates(cue_results, bin_size=250)

    assert aggregated == [
        HitRateBin(bin_start=500, hits=1, total=1, hit_rate=1.0),
    ]


def test_compute_lick_rates_by_position_counts_all_bins() -> None:
    entries = [
        make_entry("Path Position", {"position": 10}),
        make_entry("Lick"),
        make_entry("Path Position", {"position": 120}),
        make_entry("Lick"),
        make_entry("Lick"),
        make_entry("Path Position", {"position": 220}),
        make_entry("Lick"),
    ]

    results = compute_lick_rates_by_position(entries, bin_size=100)

    expected_bins = enumerate_position_bins(10, 220, 100)
    assert [bin_stats.bin_start for bin_stats in results] == list(expected_bins)
    assert [bin_stats.lick_count for bin_stats in results] == [1, 2, 1]
    assert [bin_stats.lick_rate_per_unit for bin_stats in results] == [
        pytest.approx(1 / 100),
        pytest.approx(2 / 100),
        pytest.approx(1 / 100),
    ]


def test_invalid_bin_size_raises() -> None:
    with pytest.raises(ValueError):
        aggregate_hit_rates([], bin_size=0)

    with pytest.raises(ValueError):
        enumerate_position_bins(0, 10, -5)


def test_missing_cue_results_raises() -> None:
    with pytest.raises(ValueError, match="No cue results"):
        compute_hit_rates_by_position([], bin_size=100)


def test_missing_path_positions_raises() -> None:
    entries = [make_entry("Lick")]
    with pytest.raises(ValueError, match="No path positions"):
        compute_lick_rates_by_position(entries, bin_size=50)
