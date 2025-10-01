"""Compute hit rate as a function of the animal's original corridor position.

This script parses a Unity log exported as JSON (e.g. "Log BM35 ... json")
and aggregates cue outcomes by the cue's spawn position along the corridor.

The script reports hit rates per position bin and produces a plot saved next
to the input file when matplotlib is available. A hit is defined as an event
where a reward was delivered (`hasGivenReward` set to True in the log).
"""

from __future__ import annotations

import argparse
from importlib import util as importlib_util
import json
import math
import pathlib
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

_matplotlib_spec = importlib_util.find_spec("matplotlib")
if _matplotlib_spec is not None:
    import matplotlib.pyplot as plt  # type: ignore
else:
    plt = None  # type: ignore

CueResult = Dict[str, object]


def load_cue_results(path: pathlib.Path) -> List[CueResult]:
    """Return the list of cue result dictionaries from the Unity log."""
    with path.open("r", encoding="utf-8") as fh:
        log_entries = json.load(fh)

    cue_results: List[CueResult] = []
    for entry in log_entries:
        if entry.get("msg") == "Cue Result":
            cue_results.append(entry["data"])
    return cue_results


def aggregate_hit_rates(
    cue_results: Iterable[CueResult], bin_size: int
) -> List[Tuple[int, int, int, float]]:
    """Aggregate cue hits by the cue's original corridor position.

    Args:
        cue_results: Iterator of cue result dictionaries from the log.
        bin_size: Width of the position bins.

    Returns:
        A list of tuples of the form (bin_start, hits, total, hit_rate)
        sorted by bin_start.
    """

    stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {"hits": 0, "total": 0})

    for cue in cue_results:
        position = int(cue["position"])
        bin_start = (position // bin_size) * bin_size
        stats[bin_start]["total"] += 1
        if cue.get("hasGivenReward"):
            stats[bin_start]["hits"] += 1

    aggregated: List[Tuple[int, int, int, float]] = []
    for bin_start, counts in stats.items():
        total = counts["total"]
        hits = counts["hits"]
        hit_rate = hits / total if total else math.nan
        aggregated.append((bin_start, hits, total, hit_rate))

    aggregated.sort(key=lambda row: row[0])
    return aggregated


def write_summary_table(
    aggregated: List[Tuple[int, int, int, float]], output_path: pathlib.Path
) -> None:
    """Write the aggregated hit-rate table to a TSV file."""

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("position_bin_start\thits\ttotal\thit_rate\n")
        for bin_start, hits, total, hit_rate in aggregated:
            fh.write(f"{bin_start}\t{hits}\t{total}\t{hit_rate:.3f}\n")


def plot_hit_rate(
    aggregated: List[Tuple[int, int, int, float]], output_path: pathlib.Path
) -> None:
    """Save a line plot of hit rate vs. original position when possible."""

    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment.")

    positions = [row[0] for row in aggregated]
    hit_rates = [row[3] for row in aggregated]

    plt.figure(figsize=(10, 4))
    plt.plot(positions, hit_rates, marker="o", linestyle="-", linewidth=1.5)
    plt.xlabel("Original corridor position (bin start)")
    plt.ylabel("Hit rate")
    plt.title("Hit rate vs. original corridor position")
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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
        default=1000,
        help="Width of the position bins in corridor units (default: 1000).",
    )
    parser.add_argument(
        "--output-prefix",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional prefix for output files. When omitted, outputs are saved"
            " alongside the input JSON using the log file's stem."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting even if matplotlib is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path: pathlib.Path = args.log_path
    cue_results = load_cue_results(log_path)

    if not cue_results:
        raise SystemExit("No cue results found in the provided log file.")

    aggregated = aggregate_hit_rates(cue_results, args.bin_size)

    if args.output_prefix is None:
        output_prefix = log_path.with_suffix("")
    else:
        output_prefix = args.output_prefix

    tsv_path = pathlib.Path(f"{output_prefix}_hit_rate_by_position.tsv")
    write_summary_table(aggregated, tsv_path)
    print(f"Wrote hit-rate table to {tsv_path}")

    if not args.no_plot:
        plot_path = pathlib.Path(f"{output_prefix}_hit_rate_by_position.png")
        try:
            plot_hit_rate(aggregated, plot_path)
        except RuntimeError as exc:
            print(f"Skipping plot: {exc}")
        else:
            print(f"Wrote hit-rate plot to {plot_path}")


if __name__ == "__main__":
    main()