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
import pathlib
from typing import Iterable

from behavioral_analysis.analysis.corridor_metrics import (
    LickRateBin,
    compute_lick_rates_by_position,
    load_log_entries,
)


def write_lick_table(
    aggregated: Iterable[LickRateBin], output_path: pathlib.Path
) -> None:
    """Write the aggregated lick statistics to a TSV file."""
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "position_bin_start\tposition_bin_end\tlicks\tlick_rate_per_unit\n"
        )
        for stats in aggregated:
            fh.write(
                f"{stats.bin_start}\t{stats.bin_end}\t"
                f"{stats.lick_count}\t{stats.lick_rate_per_unit:.6f}\n"
            )


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
    try:
        aggregated = compute_lick_rates_by_position(entries, args.bin_size)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.output_prefix is None:
        output_prefix = log_path.with_suffix("")
    else:
        output_prefix = args.output_prefix

    output_path = pathlib.Path(f"{output_prefix}_lick_rate_by_position.tsv")
    write_lick_table(aggregated, output_path)
    print(f"Wrote lick-rate table to {output_path}")


if __name__ == "__main__":
    main()
