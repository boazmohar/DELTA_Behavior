"""Utilities for computing corridor-position metrics from Unity logs."""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence

LogEntry = Dict[str, Any]
CueResult = Dict[str, Any]


@dataclass(frozen=True)
class HitRateBin:
    """Summary statistics for cue hit rate within a position bin."""

    bin_start: int
    hits: int
    total: int
    hit_rate: float


@dataclass(frozen=True)
class LickRateBin:
    """Summary statistics for lick density within a position bin."""

    bin_start: int
    bin_end: int
    lick_count: int
    lick_rate_per_unit: float


def load_log_entries(path: str | pathlib.Path) -> List[LogEntry]:
    """Load raw Unity JSON log entries.

    Args:
        path: Path to the JSON export produced by the Unity task.

    Returns:
        List of log entry dictionaries. Entries that are not dictionaries are
        skipped to maintain a predictable structure for downstream functions.
    """

    json_path = pathlib.Path(path)
    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if not isinstance(payload, list):
        raise ValueError("Unity log JSON must contain a list at the top level.")

    entries: List[LogEntry] = []
    for entry in payload:
        if isinstance(entry, MutableMapping):
            entries.append(dict(entry))
    return entries


def extract_cue_results(entries: Iterable[LogEntry]) -> List[CueResult]:
    """Return cue result dictionaries embedded in the Unity log."""

    cue_results: List[CueResult] = []
    for entry in entries:
        if entry.get("msg") != "Cue Result":
            continue
        data = entry.get("data")
        if isinstance(data, MutableMapping):
            cue_results.append(dict(data))
    return cue_results


def _ensure_positive_bin_size(bin_size: int) -> None:
    if bin_size <= 0:
        raise ValueError("bin_size must be a positive integer")


def _compute_bin_start(position: float, bin_size: int) -> int:
    return int(math.floor(position / bin_size) * bin_size)


def aggregate_hit_rates(
    cue_results: Iterable[CueResult],
    bin_size: int,
) -> List[HitRateBin]:
    """Aggregate cue hit rates within fixed-width position bins."""

    _ensure_positive_bin_size(bin_size)

    stats: Dict[int, Dict[str, int]] = {}
    for cue in cue_results:
        position = cue.get("position")
        if position is None:
            continue
        try:
            position_value = float(position)
        except (TypeError, ValueError):
            continue

        bin_start = _compute_bin_start(position_value, bin_size)
        bucket = stats.setdefault(bin_start, {"hits": 0, "total": 0})
        bucket["total"] += 1
        if cue.get("hasGivenReward"):
            bucket["hits"] += 1

    results: List[HitRateBin] = []
    for bin_start, counts in stats.items():
        total = counts["total"]
        hits = counts["hits"]
        hit_rate = hits / total if total else math.nan
        results.append(HitRateBin(bin_start=bin_start, hits=hits, total=total, hit_rate=hit_rate))

    results.sort(key=lambda row: row.bin_start)
    return results


def compute_hit_rates_by_position(
    log_entries: Iterable[LogEntry],
    bin_size: int,
) -> List[HitRateBin]:
    """Convenience wrapper to compute hit rates directly from log entries."""

    cue_results = extract_cue_results(log_entries)
    if not cue_results:
        raise ValueError("No cue results found in provided log entries.")
    return aggregate_hit_rates(cue_results, bin_size)


def extract_path_positions(entries: Iterable[LogEntry]) -> List[float]:
    """Extract numeric path-position samples from Unity log entries."""

    positions: List[float] = []
    for entry in entries:
        if entry.get("msg") != "Path Position":
            continue
        data = entry.get("data")
        if not isinstance(data, MutableMapping):
            continue
        position = data.get("position")
        if position is None:
            continue
        try:
            positions.append(float(position))
        except (TypeError, ValueError):
            continue
    return positions


def extract_lick_positions(entries: Iterable[LogEntry]) -> List[float]:
    """Return the corridor position recorded at each lick event."""

    lick_positions: List[float] = []
    current_position: float | None = None

    for entry in entries:
        msg = entry.get("msg")
        if msg == "Path Position":
            data = entry.get("data")
            if not isinstance(data, MutableMapping):
                continue
            position = data.get("position")
            if position is None:
                continue
            try:
                current_position = float(position)
            except (TypeError, ValueError):
                continue
        elif msg == "Lick":
            if current_position is not None:
                lick_positions.append(current_position)

    return lick_positions


def enumerate_position_bins(
    min_position: float,
    max_position: float,
    bin_size: int,
) -> Sequence[int]:
    """Return the inclusive sequence of bin starts covering the range."""

    _ensure_positive_bin_size(bin_size)
    if min_position > max_position:
        raise ValueError("min_position cannot be greater than max_position")

    min_bin = _compute_bin_start(min_position, bin_size)
    max_bin = _compute_bin_start(max_position, bin_size)
    return list(range(int(min_bin), int(max_bin) + bin_size, bin_size))


def aggregate_lick_rates(
    lick_positions: Iterable[float],
    bin_starts: Sequence[int],
    bin_size: int,
) -> List[LickRateBin]:
    """Aggregate lick counts and normalise by corridor bin width."""

    _ensure_positive_bin_size(bin_size)

    counts = {start: 0 for start in bin_starts}

    for position in lick_positions:
        try:
            position_value = float(position)
        except (TypeError, ValueError):
            continue
        bin_start = _compute_bin_start(position_value, bin_size)
        if bin_start not in counts:
            continue
        counts[bin_start] += 1

    results: List[LickRateBin] = []
    for start in bin_starts:
        lick_count = counts[start]
        results.append(
            LickRateBin(
                bin_start=start,
                bin_end=start + bin_size,
                lick_count=lick_count,
                lick_rate_per_unit=lick_count / bin_size,
            )
        )

    return results


def compute_lick_rates_by_position(
    log_entries: Iterable[LogEntry],
    bin_size: int,
) -> List[LickRateBin]:
    """Convenience wrapper to compute lick density directly from log entries."""

    path_positions = extract_path_positions(log_entries)
    if not path_positions:
        raise ValueError("No path positions found in provided log entries.")

    min_position = min(path_positions)
    max_position = max(path_positions)
    bin_starts = enumerate_position_bins(min_position, max_position, bin_size)
    lick_positions = extract_lick_positions(log_entries)

    return aggregate_lick_rates(lick_positions, bin_starts, bin_size)


__all__ = [
    "LogEntry",
    "CueResult",
    "HitRateBin",
    "LickRateBin",
    "load_log_entries",
    "extract_cue_results",
    "aggregate_hit_rates",
    "compute_hit_rates_by_position",
    "extract_path_positions",
    "extract_lick_positions",
    "enumerate_position_bins",
    "aggregate_lick_rates",
    "compute_lick_rates_by_position",
]
