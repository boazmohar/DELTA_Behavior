"""Lick-to-cue alignment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class LickTrialAlignment:
    """Outputs after tagging lick events relative to cue windows."""

    trials: pd.DataFrame
    licks_with_position: pd.DataFrame
    licks_with_trial: pd.DataFrame




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
        return LickTrialAlignment(
            trials=trials_out,
            licks_with_position=licks_out,
            licks_with_trial=pd.DataFrame(),
        )

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
        return LickTrialAlignment(
            trials=trials_out,
            licks_with_position=licks_out,
            licks_with_trial=pd.DataFrame(),
        )

    matched = pd.merge_asof(
        lick_candidates,
        trials_lookup,
        left_on="global_position_cm",
        right_on="cue_global_position_cm",
        direction="nearest",
    )

    matched["lick_offset_from_cue_cm"] = (
        matched["global_position_cm"] - matched["cue_global_position_cm"]
    )
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


__all__ = ["LickTrialAlignment", "align_lick_events_to_position", "align_licks_to_trials"]