#!/usr/bin/env python3
"""End-to-end pipeline: process JSON then generate summary visualisations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from behavioral_analysis.processing import process_json_to_hdf5
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
    parser.add_argument("input_json", type=Path, help="Unity JSON log to process.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Directory for generated outputs (default: ./outputs).",
    )
    parser.add_argument(
        "--corridor-length",
        type=float,
        default=500.0,
        help="Corridor length in cm used for global position calculations.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=30,
        help="Trial bin size for learning-curve plots.",
    )
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip generation of the PDF session report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output from processing functions.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input_json.exists():
        raise SystemExit(f"Input JSON not found: {args.input_json}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.input_json.stem
    hdf5_path = output_dir / f"{stem}.h5"
    csv_path = output_dir / f"{stem}_trials.csv"

    print("=" * 60)
    print("COMPLETE BEHAVIORAL DATA PROCESSING")
    print("=" * 60)

    print("\nStep 1: Processing JSON to HDF5...")
    process_json_to_hdf5(
        input_file=str(args.input_json),
        output_file=str(hdf5_path),
        corridor_length_cm=args.corridor_length,
        include_trials=True,
        verbose=not args.quiet,
    )
    print(f"\n✓ Saved HDF5 to: {hdf5_path}")

    print("\n" + "=" * 60)
    print("Step 2: Loading data and generating visualizations...")
    hdf5_data, trials_df = load_data(str(hdf5_path), None)

    if trials_df is None:
        print("Error: Could not load trial data")
        return 1

    if "was_hit" not in trials_df.columns:
        trials_df["was_hit"] = trials_df["outcome"].isin(["Hit", "FA"])

    trials_df.to_csv(csv_path, index=False)
    print(f"✓ Saved trials to: {csv_path}")

    print("\nData Summary:")
    print(f"  - Total trials: {len(trials_df)}")
    print(f"  - Session duration: {trials_df['session_time_min'].max():.1f} minutes")
    print(f"  - Corridors traversed: {trials_df['corridor'].nunique()}")

    print("\nPerformance Metrics:")
    rewarding = trials_df["is_rewarding"].sum()
    non_rewarding = (~trials_df["is_rewarding"]).sum()
    hit_rate = (trials_df["outcome"] == "Hit").sum() / rewarding if rewarding else 0.0
    fa_rate = (trials_df["outcome"] == "FA").sum() / non_rewarding if non_rewarding else 0.0
    print(f"  - Overall accuracy: {trials_df['correct'].mean():.1%}")
    print(f"  - Hit rate: {hit_rate:.1%}")
    print(f"  - FA rate: {fa_rate:.1%}")

    if "mouse_global_position_cm" in trials_df.columns:
        print("\n✓ Mouse positions at hit time are included!")
        print(
            "  Mouse position range: "
            f"{trials_df['mouse_global_position_cm'].min():.1f} to "
            f"{trials_df['mouse_global_position_cm'].max():.1f} cm"
        )
    else:
        print("\n⚠ Mouse positions not included (using cue positions)")

    print("\nGenerating visualizations...")
    timeline_fig = plot_trial_outcomes_timeline(trials_df)
    timeline_path = output_dir / f"{stem}_trial_outcomes.png"
    timeline_fig.savefig(timeline_path, dpi=100, bbox_inches="tight")
    plt.close(timeline_fig)

    summary_fig = plot_performance_summary(trials_df)
    summary_path = output_dir / f"{stem}_performance_summary.png"
    summary_fig.savefig(summary_path, dpi=100, bbox_inches="tight")
    plt.close(summary_fig)

    position_fig = plot_position_trace_with_events(hdf5_data, trials_df.copy(), time_range=(0, 5))
    position_path = output_dir / f"{stem}_position_trace.png"
    position_fig.savefig(position_path, dpi=100, bbox_inches="tight")
    plt.close(position_fig)

    try:
        learning_fig = plot_learning_curves(trials_df, bin_size=args.bin_size)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"    Warning: Could not create learning curves: {exc}")
        learning_fig = None

    if learning_fig is not None:
        learning_path = output_dir / f"{stem}_learning_curves.png"
        learning_fig.savefig(learning_path, dpi=100, bbox_inches="tight")
        plt.close(learning_fig)
    else:
        learning_path = None

    if not args.skip_report:
        report_path = output_dir / f"{stem}_session_report.pdf"
        create_session_report(str(hdf5_path), str(csv_path), save_path=str(report_path))
    else:
        report_path = None

    print("\n✓ Artifacts saved:")
    print(f"  - {timeline_path}")
    print(f"  - {summary_path}")
    print(f"  - {position_path}")
    if learning_path:
        print(f"  - {learning_path}")
    if report_path:
        print(f"  - {report_path}")

    print("\n" + "=" * 60)
    print("✓ PROCESSING COMPLETE!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
