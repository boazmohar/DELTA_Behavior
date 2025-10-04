
"""Helper utilities for inspecting cue-aligned licking behavior in notebooks.

This module wraps the existing processing pipeline so exploratory notebooks can stay
lightweight: heavy data manipulation lives here while the notebook focuses on
configuration and presentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from behavioral_analysis.io.json_parser import parse_json_file, get_event_types
from behavioral_analysis.io.dataframe_builder import extract_events_by_type
from behavioral_analysis.processing.corridor_detector_simple import (
    detect_corridors_simple,
    add_corridor_info_to_events,
)
from behavioral_analysis.processing.trial_matcher import create_trial_dataframe


sns.set_theme(style="whitegrid", context="notebook")


@dataclass
class SessionArtifacts:
    """Container for per-session tables used in exploratory notebooks."""

    json_path: Path
    events: List[Dict[str, Any]]
    event_counts: pd.Series
    dataframes: Dict[str, pd.DataFrame]
    updated_events: Dict[str, pd.DataFrame]
    corridor_info: pd.DataFrame
    position_with_corridors: Optional[pd.DataFrame]
    trials: pd.DataFrame
    licks_with_position: pd.DataFrame


@dataclass
class LickTrialAlignment:
    """Outputs after tagging lick events relative to cue windows."""

    trials: pd.DataFrame
    licks_with_position: pd.DataFrame
    licks_with_trial: pd.DataFrame


def load_session_artifacts(
    json_path: Path,
    corridor_length_cm: float = 500.0,
    verbose: bool = False,
) -> SessionArtifacts:
    """Parse a session JSON log and return processed data tables.

    Args:
        json_path: Path to the JSON log file.
        corridor_length_cm: Corridor length used for global position calculations.
        verbose: Whether to print progress from the underlying pipeline helpers.

    Returns:
        SessionArtifacts with processed DataFrames.
    """

    json_path = Path(json_path).expanduser().resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON log not found: {json_path}")

    events = parse_json_file(str(json_path), verbose=verbose)
    event_counts = pd.Series(get_event_types(events)).sort_values(ascending=False)

    raw_frames = extract_events_by_type(events, verbose=verbose)

    cue_state_key = "Cue State" if "Cue State" in raw_frames else "Cue_State"
    cue_result_key = "Cue Result" if "Cue Result" in raw_frames else "Cue_Result"

    position_df = raw_frames.get("Position")
    corridor_info, position_with_corridors = detect_corridors_simple(
        raw_frames[cue_state_key],
        position_df,
        corridor_length_cm=corridor_length_cm,
        verbose=verbose,
        cue_result_df=raw_frames.get(cue_result_key),
    )

    updated_events = add_corridor_info_to_events(
        raw_frames,
        corridor_info,
        corridor_length_cm=corridor_length_cm,
        verbose=verbose,
        position_df=position_with_corridors,
    )

    trials_df = create_trial_dataframe(
        updated_events[cue_state_key],
        updated_events[cue_result_key],
        corridor_length_cm=corridor_length_cm,
        verbose=verbose,
        position_df=position_with_corridors,
    )

    licks_with_position = align_lick_events_to_position(
        updated_events,
        position_with_corridors,
    )

    return SessionArtifacts(
        json_path=json_path,
        events=events,
        event_counts=event_counts,
        dataframes=raw_frames,
        updated_events=updated_events,
        corridor_info=corridor_info,
        position_with_corridors=position_with_corridors,
        trials=trials_df,
        licks_with_position=licks_with_position,
    )


def align_lick_events_to_position(
    updated_events: Dict[str, pd.DataFrame],
    position_with_corridors: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Attach global position & corridor metadata to lick timestamps."""

    if position_with_corridors is None:
        return pd.DataFrame()

    lick_df = updated_events.get("Lick")
    if lick_df is None or lick_df.empty:
        return pd.DataFrame()

    position_lookup = (
        position_with_corridors[["time", "global_position_cm", "corridor_id"]]
        .dropna(subset=["global_position_cm"])
        .sort_values("time")
    )

    if position_lookup.empty:
        return pd.DataFrame()

    licks = pd.merge_asof(
        lick_df.sort_values("time"),
        position_lookup,
        on="time",
        direction="nearest",
    )

    if "corridor_id_x" in licks.columns:
        licks = licks.drop(columns=["corridor_id_x"])
    if "corridor_id_y" in licks.columns:
        licks = licks.rename(columns={"corridor_id_y": "corridor_id"})

    licks["time_s"] = licks["time"] / 1000.0
    licks["session_time_min"] = licks["time"] / 60000.0

    return licks.reset_index(drop=True)


def align_licks_to_trials(
    trials: pd.DataFrame,
    licks_with_position: pd.DataFrame,
    cue_window_half_width_cm: float,
) -> LickTrialAlignment:
    """Determine which licks fall inside each cue window and count them per trial."""

    trials_out = trials.copy()
    licks_out = licks_with_position.copy()
    if "licks_in_window" in trials_out.columns:
        trials_out = trials_out.drop(columns=["licks_in_window"])
    trials_out["licks_in_window"] = 0

    if trials_out.empty or licks_out.empty:
        if not licks_out.empty:
            licks_out["in_cue_window"] = False
        return LickTrialAlignment(trials=trials_out, licks_with_position=licks_out, licks_with_trial=pd.DataFrame())

    trials_lookup = trials_out[[
        "trial_id",
        "global_position_cm",
        "corridor",
        "session_time_min",
        "outcome",
        "is_rewarding",
    ]].copy()
    trials_lookup = trials_lookup.rename(
        columns={
            "global_position_cm": "cue_global_position_cm",
            "session_time_min": "cue_onset_min",
        }
    )
    trials_lookup = trials_lookup.dropna(subset=["cue_global_position_cm"]).sort_values("cue_global_position_cm")

    lick_candidates = licks_out.dropna(subset=["global_position_cm"]).sort_values("global_position_cm")

    if trials_lookup.empty or lick_candidates.empty:
        licks_out["in_cue_window"] = False
        return LickTrialAlignment(trials=trials_out, licks_with_position=licks_out, licks_with_trial=pd.DataFrame())

    matched = pd.merge_asof(
        lick_candidates,
        trials_lookup,
        left_on="global_position_cm",
        right_on="cue_global_position_cm",
        direction="nearest",
    )

    matched["lick_offset_from_cue_cm"] = matched["global_position_cm"] - matched["cue_global_position_cm"]
    matched["in_cue_window"] = matched["lick_offset_from_cue_cm"].abs() <= cue_window_half_width_cm

    licks_out["in_cue_window"] = False
    licks_out.loc[matched.index, "in_cue_window"] = matched["in_cue_window"].fillna(False).values

    licks_with_trial = matched[matched["in_cue_window"] & matched["trial_id"].notna()].copy()

    if not licks_with_trial.empty:
        licks_per_trial = licks_with_trial.groupby("trial_id").size()
        trials_out["licks_in_window"] = (
            trials_out["trial_id"].map(licks_per_trial).fillna(0).astype(int)
        )

    return LickTrialAlignment(
        trials=trials_out,
        licks_with_position=licks_out,
        licks_with_trial=licks_with_trial,
    )


def build_trial_summary(trials: pd.DataFrame) -> pd.DataFrame:
    """Return counts of outcomes split by cue type."""

    if trials.empty:
        return pd.DataFrame(columns=["cue_type", "outcome", "count"])

    summary = (
        trials.groupby(["cue_type", "outcome"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    return summary


def build_lick_summary(licks_with_trial: pd.DataFrame) -> pd.DataFrame:
    """Aggregate lick counts by outcome and reward contingency."""

    if licks_with_trial.empty:
        return pd.DataFrame(columns=["outcome", "is_rewarding", "licks_in_window"])

    summary = (
        licks_with_trial.groupby(["outcome", "is_rewarding"])
        .size()
        .rename("licks_in_window")
        .reset_index()
        .sort_values("licks_in_window", ascending=False)
    )
    return summary


def plot_trial_overview(
    trials: pd.DataFrame,
    licks_with_position: pd.DataFrame,
    cue_window_half_width_cm: float = 10.0,
    x_axis: str = "time",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Scatter plot of trials and lick events along session time or corridor ID."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.figure  # noqa: F841  # keep reference for interactive backends

    if trials.empty:
        ax.text(0.5, 0.5, "No trial data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    if x_axis not in {"time", "corridor"}:
        raise ValueError("x_axis must be 'time' or 'corridor'")

    trials_plot = trials.copy()
    licks_plot = licks_with_position.copy()

    if x_axis == "time":
        trial_x = trials_plot["cue_onset_ms"] / 60000.0
        trial_y = trials_plot["corridor"]
        x_label = "Session time (min)"
        y_label = "Corridor ID"
        if not licks_plot.empty:
            lick_x = licks_plot["session_time_min"]
            lick_y = licks_plot["corridor_id"]
    else:  # corridor on x-axis
        trial_x = trials_plot["corridor"]
        trial_y = trials_plot["session_time_min"]
        x_label = "Corridor ID"
        y_label = "Session time (min)"
        if not licks_plot.empty:
            lick_x = licks_plot["corridor_id"]
            lick_y = licks_plot["session_time_min"]

    outcome_markers = {"Hit": "o", "Miss": "X", "FA": "s", "CR": "D"}
    reward_colors = {True: "#2ecc71", False: "#95a5a6"}

    if not licks_plot.empty:
        in_window = licks_plot.get("in_cue_window", pd.Series(False, index=licks_plot.index))
        out_window = ~in_window

        if out_window.any():
            ax.scatter(
                lick_x[out_window],
                lick_y[out_window],
                s=14,
                color="#b8b8b8",
                alpha=0.4,
                label="Lick outside cue window",
            )
        if in_window.any():
            ax.scatter(
                lick_x[in_window],
                lick_y[in_window],
                s=20,
                color="#1f77b4",
                alpha=0.6,
                label="Lick in cue window",
            )

    for outcome, marker in outcome_markers.items():
        subset = trials_plot[trials_plot["outcome"] == outcome]
        if subset.empty:
            continue
        ax.scatter(
            trial_x.loc[subset.index],
            trial_y.loc[subset.index],
            marker=marker,
            s=80,
            c=subset["is_rewarding"].map(reward_colors),
            edgecolors="black",
            linewidths=0.4,
            label=outcome,
            alpha=0.85,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    title_bits = ["Cue & lick overview"]
    if x_axis == "time":
        title_bits.append("(x = session time)")
    else:
        title_bits.append("(x = corridor)")
    title_bits.append(f"±{cue_window_half_width_cm:g} cm window")
    ax.set_title(" ".join(title_bits))
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ordered_labels = list(unique.keys())
        ordered_handles = [unique[label] for label in ordered_labels]
        ax.legend(ordered_handles, ordered_labels, loc="upper right", framealpha=0.95)

    if x_axis == "time" and not trials_plot.empty:
        ax.set_ylim(trials_plot["corridor"].min() - 0.5, trials_plot["corridor"].max() + 0.5)
    elif x_axis == "corridor" and not trials_plot.empty:
        ax.set_xlim(trials_plot["corridor"].min() - 0.5, trials_plot["corridor"].max() + 0.5)

    return ax


def plot_lick_offset_histogram(
    licks_with_trial: pd.DataFrame,
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot histogram of lick offsets from each cue center."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
    else:
        fig = ax.figure  # noqa: F841

    if licks_with_trial.empty:
        ax.text(0.5, 0.5, "No licks inside cue windows", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax

    sns.histplot(
        data=licks_with_trial,
        x="lick_offset_from_cue_cm",
        hue="outcome",
        multiple="stack",
        bins=bins,
        palette="Set2",
        ax=ax,
    )
    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("Lick offset from cue center (cm)")
    ax.set_ylabel("Count")
    ax.set_title("Lick positions relative to cue center")
    ax.grid(True, alpha=0.3)

    return ax


__all__ = [
    "SessionArtifacts",
    "LickTrialAlignment",
    "load_session_artifacts",
    "align_lick_events_to_position",
    "align_licks_to_trials",
    "build_trial_summary",
    "build_lick_summary",
    "plot_trial_overview",
    "plot_lick_offset_histogram",
]
