"""
Utilities for multi-day, per-animal behavioral analysis workflows.

This module provides helpers to:
- discover JSON log files for a specific animal
- convert those logs to HDF5 files using the standard processing pipeline
- assemble trial-level data across days with running accuracy metrics
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import pandas as pd

from behavioral_analysis.processing.json_to_hdf5_processor import process_json_to_hdf5
from behavioral_analysis.processing.trial_matcher import calculate_performance_metrics

_DATE_PATTERN = re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})")
_SESSION_PATTERN = re.compile(r"session[_\\s-]*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class SessionConversion:
    """Record describing a converted session for an individual animal."""

    animal_id: str
    json_path: Path
    hdf5_path: Path
    session_date: date
    session_number: Optional[int]
    session_label: str


@dataclass(frozen=True)
class DayTransition:
    """Marker for the first trial index corresponding to a new experimental day."""

    session_date: date
    first_trial_index: int


@dataclass(frozen=True)
class ConversionFailure:
    """Details about a session conversion that failed to process."""

    json_path: Path
    error: str


@dataclass(frozen=True)
class MultiDayTrialsResult:
    """
    Container for concatenated trials and derived metadata across days.

    Attributes:
        trials: Trial-level dataframe ordered across all sessions.
        day_transitions: List of day transition markers (first trial index per day).
        session_summary: Per-session performance summary table.
    """

    trials: pd.DataFrame
    day_transitions: List[DayTransition]
    session_summary: pd.DataFrame


def collect_json_logs(animal_id: str, search_root: Path) -> List[Path]:
    """
    Recursively gather JSON log files for the requested animal.

    Args:
        animal_id: Animal identifier to match within filenames.
        search_root: Directory to search (searched recursively).

    Returns:
        Sorted list of JSON log Paths.

    Raises:
        FileNotFoundError: If the search_root directory does not exist.
    """
    root = Path(search_root)
    if not root.exists():
        raise FileNotFoundError(f"Search root does not exist: {root}")

    token = animal_id.lower()
    json_files = [
        path
        for path in root.rglob("*.json")
        if token in path.name.lower()
    ]
    return sorted(json_files)


def _infer_session_metadata(json_path: Path) -> Tuple[date, Optional[int]]:
    """Infer session date and number from a JSON filename or fall back to mtime."""
    stem = json_path.stem

    date_match = _DATE_PATTERN.search(stem) or _DATE_PATTERN.search(json_path.as_posix())
    if date_match:
        year, month, day = map(int, date_match.groups())
        session_date = date(year, month, day)
    else:
        session_date = datetime.fromtimestamp(json_path.stat().st_mtime).date()

    session_match = _SESSION_PATTERN.search(stem)
    session_number = int(session_match.group(1)) if session_match else None

    return session_date, session_number


def _build_session_label(session_date: date, session_number: Optional[int], stem: str) -> str:
    """Create a human-readable session label."""
    if session_number is not None:
        return f"{session_date.isoformat()} - session {session_number:d}"
    return f"{session_date.isoformat()} - {stem}"


def convert_json_sessions(
    json_paths: Sequence[Path],
    animal_id: str,
    output_dir: Path,
    *,
    corridor_length_cm: float = 500.0,
    include_trials: bool = True,
    overwrite: bool = False,
    verbose: bool = True,
    skip_failures: bool = False,
    return_failures: bool = False,
) -> Union[List[SessionConversion], Tuple[List[SessionConversion], List[ConversionFailure]]]:
    """
    Convert JSON logs to HDF5 files for a single animal.

    Args:
        json_paths: Iterable of JSON files to process.
        animal_id: Animal identifier (used for metadata only).
        output_dir: Directory where HDF5 outputs will be written.
        corridor_length_cm: Corridor length passed to the processor.
        include_trials: Whether to include trial tables in the outputs.
        overwrite: If False, skip processing when the destination already exists.
        verbose: Forward verbose flag to the processing pipeline.

    Returns:
        If return_failures is False (default), returns a list of successful SessionConversion
        records. When return_failures is True, returns a tuple of (conversions, failures).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conversions: List[SessionConversion] = []
    failures: List[ConversionFailure] = []

    for json_path in sorted(json_paths):
        session_date, session_number = _infer_session_metadata(json_path)
        stem = json_path.stem
        hdf5_path = output_dir / f"{stem}.h5"

        should_process = overwrite or not hdf5_path.exists()
        if should_process:
            try:
                process_json_to_hdf5(
                    input_file=str(json_path),
                    output_file=str(hdf5_path),
                    corridor_length_cm=corridor_length_cm,
                    include_combined=False,
                    include_trials=include_trials,
                    enable_monotonic_position=True,
                    overwrite=True,
                    verbose=verbose,
                )
            except Exception as exc:  # pragma: no cover - pass-through to caller
                if skip_failures:
                    error_message = f"{type(exc).__name__}: {exc}"
                    failures.append(ConversionFailure(json_path=json_path, error=error_message))
                    if verbose:
                        print(f"✗ Failed to process {json_path}: {error_message}")
                    continue
                raise

        session_label = _build_session_label(session_date, session_number, stem)
        conversions.append(
            SessionConversion(
                animal_id=animal_id,
                json_path=json_path,
                hdf5_path=hdf5_path,
                session_date=session_date,
                session_number=session_number,
                session_label=session_label,
            )
        )

    if return_failures:
        return conversions, failures
    return conversions


def prepare_multi_day_trials(
    conversions: Sequence[SessionConversion],
    *,
    rolling_window: int = 20,
) -> MultiDayTrialsResult:
    """
    Assemble trial tables across sessions and calculate running accuracy.

    Args:
        conversions: Processed session records.
        rolling_window: Window size (in trials) for rolling accuracy percentage.

    Returns:
        MultiDayTrialsResult containing concatenated trials, day transitions, and summaries.

    Raises:
        ValueError: If no conversions are provided or required datasets are missing.
    """
    if not conversions:
        raise ValueError("No session conversions were provided.")

    ordered = sorted(
        conversions,
        key=lambda record: (
            record.session_date,
            record.session_number if record.session_number is not None else 0,
            record.json_path.name.lower(),
        ),
    )

    trial_frames: List[pd.DataFrame] = []
    session_rows: List[dict] = []

    for session_index, conversion in enumerate(ordered, start=1):
        with pd.HDFStore(str(conversion.hdf5_path), "r") as store:
            if "/events/Trials" not in store:
                raise ValueError(f"Trials dataset missing in {conversion.hdf5_path}")
            session_trials = store["/events/Trials"].copy()

        if "correct" not in session_trials.columns:
            session_trials["correct"] = session_trials["outcome"].isin(["Hit", "CR"])

        sort_columns = [col for col in ("trial_id", "cue_onset_ms", "cue_outcome_ms") if col in session_trials.columns]
        if sort_columns:
            session_trials = session_trials.sort_values(sort_columns).reset_index(drop=True)
        else:
            session_trials = session_trials.reset_index(drop=True)

        session_trials["trial_in_session"] = session_trials.index + 1
        session_trials["session_order"] = session_index
        session_trials["session_date"] = conversion.session_date
        session_trials["session_label"] = conversion.session_label
        session_trials["session_number"] = conversion.session_number
        session_trials["source_json"] = conversion.json_path.as_posix()
        session_trials["source_hdf5"] = conversion.hdf5_path.as_posix()

        metrics = calculate_performance_metrics(session_trials)
        session_rows.append(
            {
                "session_order": session_index,
                "session_date": conversion.session_date,
                "session_label": conversion.session_label,
                "session_number": conversion.session_number,
                "source_json": conversion.json_path.as_posix(),
                "source_hdf5": conversion.hdf5_path.as_posix(),
                "n_trials": metrics["n_trials"],
                "accuracy_pct": metrics["accuracy"] * 100.0,
                "hit_rate_pct": metrics["hit_rate"] * 100.0,
                "fa_rate_pct": metrics["fa_rate"] * 100.0,
            }
        )

        trial_frames.append(session_trials)

    trials = pd.concat(trial_frames, ignore_index=True)
    trials = trials.sort_values(
        ["session_order", "trial_in_session"],
        kind="mergesort",
    ).reset_index(drop=True)

    trials["trial_index_global"] = trials.index + 1
    trials["rolling_accuracy_pct"] = (
        trials["correct"].astype(float).rolling(rolling_window, min_periods=1).mean() * 100.0
    )

    day_transitions = [
        DayTransition(session_date=session_date, first_trial_index=int(group["trial_index_global"].iloc[0]))
        for session_date, group in trials.groupby("session_date", sort=True)
    ]

    session_summary = pd.DataFrame(session_rows).sort_values("session_order").reset_index(drop=True)

    return MultiDayTrialsResult(
        trials=trials,
        day_transitions=day_transitions,
        session_summary=session_summary,
    )
