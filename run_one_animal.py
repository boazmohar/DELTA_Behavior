#!/usr/bin/env python3
"""Process one or more DELTA JSON logs into HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from behavioral_analysis.processing.json_to_hdf5_processor import process_json_to_hdf5


def derive_output_paths(json_path: Path, output_dir: Path, csv_dir: Path) -> tuple[Path, Path]:
    stem = json_path.stem
    return output_dir / f"{stem}.h5", csv_dir / f"{stem}_trials.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="One or more Unity JSON log files to process.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Directory where HDF5 files will be written (default: ./outputs).",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Directory for exported trials CSV files (default: same as output-dir).",
    )
    parser.add_argument(
        "--corridor-length",
        type=float,
        default=500.0,
        help="Corridor length in cm for global position calculations.",
    )
    parser.add_argument(
        "--no-trials",
        action="store_true",
        help="Disable trial DataFrame creation to speed up processing.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-processing inspection of the generated HDF5.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce console output from the processing pipeline.",
    )
    return parser.parse_args()


def verify_output(hdf5_path: Path, csv_path: Path, include_trials: bool) -> None:
    print("\n=== VERIFICATION ===")
    with pd.HDFStore(str(hdf5_path), "r") as store:
        print("\nAvailable datasets:")
        for key in store.keys():
            df = store[key]
            print(f"  {key}: {df.shape}")

        if include_trials and "/events/Trials" in store:
            trials = store["/events/Trials"]
            print(f"\n✓ Trials dataframe: {len(trials)} trials")
            print(f"  Columns: {list(trials.columns)}")

            required = ["trial_id", "corridor", "outcome", "is_rewarding", "global_position_cm"]
            missing = [col for col in required if col not in trials.columns]
            if missing:
                print(f"  ⚠ Missing columns: {missing}")
            else:
                print("  ✓ All required columns present")

            print("\n  Outcome breakdown:")
            for outcome, count in trials["outcome"].value_counts().items():
                print(f"    {outcome}: {count} ({count / len(trials) * 100:.1f}%)")

            if "trial_id" in trials.columns:
                print(f"\n  ✓ Trial IDs: {trials['trial_id'].min()} to {trials['trial_id'].max()}")

            csv_path.parent.mkdir(parents=True, exist_ok=True)
            trials.to_csv(csv_path, index=False)
            print(f"\n  ✓ Saved trials CSV to: {csv_path}")


def main() -> int:
    args = parse_args()

    output_dir = args.output_dir
    csv_dir = args.csv_dir or output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    for json_path in args.json_files:
        if not json_path.exists():
            print(f"✗ Skipping missing file: {json_path}")
            continue

        hdf5_path, csv_path = derive_output_paths(json_path, output_dir, csv_dir)

        print("=" * 60)
        print(f"Processing {json_path}")
        print("=" * 60)

        result = process_json_to_hdf5(
            input_file=str(json_path),
            output_file=str(hdf5_path),
            corridor_length_cm=args.corridor_length,
            include_combined=False,
            include_trials=not args.no_trials,
            enable_monotonic_position=True,
            overwrite=True,
            verbose=not args.quiet,
        )

        print(f"\n✓ Processing complete: {result}")

        if not args.skip_verify:
            verify_output(hdf5_path, csv_path, include_trials=not args.no_trials)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
