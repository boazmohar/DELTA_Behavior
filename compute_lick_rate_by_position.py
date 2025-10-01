"""Compute lick density by corridor position using Unity JSON logs.

The script loads a Unity behavioral log exported as JSON, pairs each lick
event with the most recent path-position sample, aggregates licks into
regular spatial bins, and writes the results to a TSV file. Every corridor
section within the recorded range is included in the output even if no licks
occurred there. The ``lick_rate_per_unit`` column reports licks per corridor
unit (``licks / bin_size``) so zero-lick sections appear with an explicit
rate of ``0``.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, Iterable, List, Sequence, Tuple

LogEntry = Dict[str, object]


def load_log_entries(path: pathlib.Path) -> List[LogEntry]:
    """Load the full list of log entries from the Unity JSON export."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_path_positions(entries: Iterable[LogEntry]) -> List[float]:
    """Extract all recorded path positions from the log."""
    positions: List[float] = []
    for entry in entries:
        if entry.get("msg") != "Path Position":
            continue
        data = entry.get("data", {})
        position = data.get("position")
        if position is None:
            continue
        try:
            positions.append(float(position))
        except (TypeError, ValueError):
            continue
    return positions


def extract_lick_positions(entries: Iterable[LogEntry]) -> List[float]:
    """Return the corridor position at each lick event."""
    lick_positions: List[float] = []
    current_position: float | None = None

    for entry in entries:
        msg = entry.get("msg")
        if msg == "Path Position":
            data = entry.get("data", {})
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


def enumerate_bins(
    min_position: float, max_position: float, bin_size: int
) -> Sequence[int]:
    """Return the inclusive list of bin starts that covers the position range."""
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")

    min_bin = math.floor(min_position / bin_size) * bin_size
    max_bin = math.floor(max_position / bin_size) * bin_size
    return list(range(int(min_bin), int(max_bin) + bin_size, bin_size))


def aggregate_lick_rates(
    lick_positions: Iterable[float],
    bin_starts: Sequence[int],
    bin_size: int,
) -> List[Tuple[int, int, int, float]]:
    """Aggregate lick counts and rates for the provided bin starts."""
    counts = {start: 0 for start in bin_starts}

    for position in lick_positions:
        bin_start = int(math.floor(position / bin_size) * bin_size)
        if bin_start not in counts:
            # Ignore licks that fall outside the recorded corridor range.
            continue
        counts[bin_start] += 1

    aggregated: List[Tuple[int, int, int, float]] = []
    for start in bin_starts:
        licks = counts[start]
        rate = licks / bin_size
        aggregated.append((start, start + bin_size, licks, rate))

    return aggregated


def write_lick_table(
    aggregated: Sequence[Tuple[int, int, int, float]], output_path: pathlib.Path
) -> None:
    """Write the aggregated lick statistics to a TSV file."""
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "position_bin_start\tposition_bin_end\tlicks\tlick_rate_per_unit\n"
        )
        for start, end, licks, rate in aggregated:
            fh.write(f"{start}\t{end}\t{licks}\t{rate:.6f}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_path",
        type=pathlib.Path,
        help="Path to the Unity JSON log (e.g. 'Log BM35 ... json').",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=100,
        help="Width of the corridor position bins (default: 100 units).",
    )
    parser.add_argument(
        "--output-prefix",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional prefix for the output TSV. When omitted, the TSV is saved"
            " alongside the input JSON using the log file's stem."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path: pathlib.Path = args.log_path

    entries = load_log_entries(log_path)
    path_positions = extract_path_positions(entries)
    if not path_positions:
        raise SystemExit("No path position samples found in the provided log file.")

    min_position = min(path_positions)
    max_position = max(path_positions)

    lick_positions = extract_lick_positions(entries)
    bin_starts = enumerate_bins(min_position, max_position, args.bin_size)
    aggregated = aggregate_lick_rates(lick_positions, bin_starts, args.bin_size)

    if args.output_prefix is None:
        output_prefix = log_path.with_suffix("")
    else:
        output_prefix = args.output_prefix

    output_path = pathlib.Path(f"{output_prefix}_lick_rate_by_position.tsv")
    write_lick_table(aggregated, output_path)
    print(f"Wrote lick-rate table to {output_path}")


if __name__ == "__main__":
    main()