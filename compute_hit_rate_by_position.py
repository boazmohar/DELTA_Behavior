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
import pathlib
from typing import Iterable, List

_matplotlib_spec = importlib_util.find_spec("matplotlib")
if _matplotlib_spec is not None:
    import matplotlib.pyplot as plt  # type: ignore
else:
    plt = None  # type: ignore

from behavioral_analysis.analysis.corridor_metrics import (
    HitRateBin,
    compute_hit_rates_by_position,
    load_log_entries,
)


def write_summary_table(
    aggregated: Iterable[HitRateBin], output_path: pathlib.Path
) -> None:
    """Write the aggregated hit-rate table to a TSV file."""

    with output_path.open("w", encoding="utf-8") as fh:
        fh.write("position_bin_start\thits\ttotal\thit_rate\n")
        for bin_stats in aggregated:
            fh.write(
                f"{bin_stats.bin_start}\t{bin_stats.hits}\t"
                f"{bin_stats.total}\t{bin_stats.hit_rate:.3f}\n"
            )


def plot_hit_rate(aggregated: List[HitRateBin], output_path: pathlib.Path) -> None:
    """Save a line plot of hit rate vs. original position when possible."""

    if plt is None:
        raise RuntimeError("matplotlib is not available in this environment.")

    positions = [row.bin_start for row in aggregated]
    hit_rates = [row.hit_rate for row in aggregated]

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
    log_entries = load_log_entries(log_path)
    try:
        aggregated = compute_hit_rates_by_position(log_entries, args.bin_size)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

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
