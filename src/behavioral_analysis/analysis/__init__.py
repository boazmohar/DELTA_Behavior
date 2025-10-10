"""Analysis utilities for DELTA behavioral data."""

from behavioral_analysis.analysis.corridor_metrics import (
    HitRateBin,
    LickRateBin,
    aggregate_hit_rates,
    aggregate_lick_rates,
    compute_hit_rates_by_position,
    compute_lick_rates_by_position,
    enumerate_position_bins,
    extract_cue_results,
    extract_lick_positions,
    extract_path_positions,
    load_log_entries,
)

from behavioral_analysis.analysis.lick_alignment import (
    LickTrialAlignment,
    align_lick_events_to_position,
    align_licks_to_trials,
)

__all__ = [
    "HitRateBin",
    "LickRateBin",
    "aggregate_hit_rates",
    "aggregate_lick_rates",
    "compute_hit_rates_by_position",
    "compute_lick_rates_by_position",
    "enumerate_position_bins",
    "extract_cue_results",
    "extract_lick_positions",
    "extract_path_positions",
    "load_log_entries",
    "LickTrialAlignment",
    "align_lick_events_to_position",
    "align_licks_to_trials",
]
