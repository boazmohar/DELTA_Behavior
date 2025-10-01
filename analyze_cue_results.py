from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


BIN_SIZE_DEFAULT = 200


@dataclass
class CueResult:
    file_path: Path
    time: float
    position: float
    is_hit: bool


def find_json_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def load_cue_results(path: Path) -> Iterable[CueResult]:
    with path.open("r", encoding="utf-8") as f:
        try:
            entries = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON file {path}") from exc

    if not isinstance(entries, list):
        return []

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("msg") != "Cue Result":
            continue
        data = entry.get("data", {})
        position = data.get("position")
        if position is None:
            continue
        has_reward = bool(data.get("hasGivenReward") or data.get("isRewarding"))
        time = entry.get("time")
        if time is None:
            continue
        yield CueResult(
            file_path=path,
            time=float(time),
            position=float(position),
            is_hit=has_reward,
        )


def summarize_results(results: Iterable[CueResult], bin_size: int) -> None:
    results = list(results)
    if not results:
        print("No cue results found across the provided files.")
        return

    earliest = min(results, key=lambda r: r.position)
    print("Earliest cue result:")
    print(f"  File      : {earliest.file_path}")
    print(f"  Time      : {earliest.time}")
    print(f"  Position  : {earliest.position}")
    print(f"  Hit       : {earliest.is_hit}")
    print()

    binned = defaultdict(lambda: {"hits": 0, "total": 0})
    for res in results:
        bin_start = bin_size * int(res.position // bin_size)
        binned[bin_start]["total"] += 1
        if res.is_hit:
            binned[bin_start]["hits"] += 1

    print(f"Hit rate by position bin (bin size = {bin_size}):")
    print(f"{'Bin Start':>10} {'Bin End':>10} {'Total':>10} {'Hits':>10} {'Hit Rate':>10}")
    for bin_start in sorted(binned):
        total = binned[bin_start]["total"]
        hits = binned[bin_start]["hits"]
        hit_rate = hits / total if total else 0.0
        bin_end = bin_start + bin_size
        print(f"{bin_start:10d} {bin_end:10d} {total:10d} {hits:10d} {hit_rate:10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze cue results across JSON log files.")
    parser.add_argument(
        "root",
        nargs="?",
        default=Path.cwd(),
        type=Path,
        help="Root directory to search for JSON files. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--bin-size",
        type=int,
        default=BIN_SIZE_DEFAULT,
        help="Size of the position bins used for hit rate calculation (default: %(default)s)",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    json_files = find_json_files(root)
    if not json_files:
        print(f"No JSON files found under {root}.")
        return

    print("Found JSON files:")
    for path in json_files:
        print(f"  {path}")
    print()

    all_results = []
    for path in json_files:
        all_results.extend(load_cue_results(path))

    summarize_results(all_results, args.bin_size)


if __name__ == "__main__":
    main()