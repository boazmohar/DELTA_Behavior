#!/usr/bin/env python3
"""Minimal helper to convert JSON logs into HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from behavioral_analysis.processing.json_to_hdf5_processor import process_json_to_hdf5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="One or more Unity JSON log files to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs",
        help="Directory to write the resulting HDF5 files (default: ./outputs).",
    )
    parser.add_argument(
        "--include-trials",
        action="store_true",
        help="Generate the trials DataFrame (disabled by default for speed).",
    )
    parser.add_argument(
        "--corridor-length",
        type=float,
        default=200.0,
        help="Corridor length used for global position calculation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose processing output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for json_path in args.json_files:
        if not json_path.exists():
            print(f"Skipping missing file: {json_path}")
            continue

        output_path = args.output_dir / f"{json_path.stem}.h5"
        print(f"Processing {json_path} -> {output_path}")
        process_json_to_hdf5(
            input_file=str(json_path),
            output_file=str(output_path),
            corridor_length_cm=args.corridor_length,
            include_trials=args.include_trials,
            verbose=not args.quiet,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
