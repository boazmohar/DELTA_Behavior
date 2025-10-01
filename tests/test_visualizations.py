#!/usr/bin/env python3
"""Generate visualization artifacts from an existing HDF5 + trials CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from behavioral_analysis.visualization.trial_visualizer import (
    load_data,
    plot_trial_outcomes_timeline,
    plot_performance_summary,
    plot_position_trace_with_events,
    plot_learning_curves,
    create_session_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hdf5_path", type=Path, help="Processed HDF5 file.")
    parser.add_argument(
        "--trials-csv",
        type=Path,
        default=None,
        help="Optional trials CSV. Defaults to <hdf5_stem>_trials.csv next to the HDF5 file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated visualisations (default: alongside HDF5).",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=30,
        help="Number of trials per bin for learning curves (default: 30).",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Do not create the PDF session report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.hdf5_path.exists():
        raise SystemExit(f"HDF5 file not found: {args.hdf5_path}")

    trials_csv = args.trials_csv
    if trials_csv is None:
        trials_csv = args.hdf5_path.with_name(f"{args.hdf5_path.stem}_trials.csv")

    if not trials_csv.exists():
        raise SystemExit(f"Trials CSV not found: {trials_csv}")

    output_dir = args.output_dir or args.hdf5_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== TESTING VISUALIZATION FUNCTIONS ===\n")
    print(f"Loading data from {args.hdf5_path} and {trials_csv}...")

    hdf5_data, trials_df = load_data(str(args.hdf5_path), str(trials_csv))

    if trials_df is None:
        print("Error: Could not load trial data")
        return 1

    print(f"Loaded {len(trials_df)} trials from {trials_df['corridor'].nunique()} corridors")
    print(f"Session duration: {trials_df['session_time_min'].max():.1f} minutes")

    if "was_hit" not in trials_df.columns:
        trials_df["was_hit"] = trials_df["outcome"].isin(["Hit", "FA"])

    print("\nGenerating visualizations...")
    outcomes_fig = plot_trial_outcomes_timeline(trials_df)
    outcomes_path = output_dir / f"{args.hdf5_path.stem}_trial_outcomes.png"
    outcomes_fig.savefig(outcomes_path, dpi=100, bbox_inches="tight")
    plt.close(outcomes_fig)

    summary_fig = plot_performance_summary(trials_df)
    summary_path = output_dir / f"{args.hdf5_path.stem}_performance_summary.png"
    summary_fig.savefig(summary_path, dpi=100, bbox_inches="tight")
    plt.close(summary_fig)

    trajectory_fig = plot_position_trace_with_events(hdf5_data, trials_df.copy(), time_range=(0, 5))
    trajectory_path = output_dir / f"{args.hdf5_path.stem}_position_trace.png"
    trajectory_fig.savefig(trajectory_path, dpi=100, bbox_inches="tight")
    plt.close(trajectory_fig)

    try:
        learning_fig = plot_learning_curves(trials_df, bin_size=args.bin_size)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"    Warning: Could not create learning curves: {exc}")
        learning_fig = None

    if learning_fig is not None:
        learning_path = output_dir / f"{args.hdf5_path.stem}_learning_curves.png"
        learning_fig.savefig(learning_path, dpi=100, bbox_inches="tight")
        plt.close(learning_fig)
    else:
        learning_path = None

    if not args.skip_report:
        report_path = output_dir / f"{args.hdf5_path.stem}_session_report.pdf"
        create_session_report(str(args.hdf5_path), str(trials_csv), save_path=str(report_path))
    else:
        report_path = None

    print("\nVisualizations saved:")
    print(f"  - {outcomes_path}")
    print(f"  - {summary_path}")
    print(f"  - {trajectory_path}")
    if learning_path:
        print(f"  - {learning_path}")
    if report_path:
        print(f"  - {report_path}")

    print("\n✓ Visualization test complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
