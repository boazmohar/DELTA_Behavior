"""
Optostim-specific helpers for MATLAB-generated behavioral logs.

These utilities reshape MATLAB Log events into tidy trial/lick/reward tables
that mirror the structure used by the corridor-based pipeline, but without
requiring Unity cue state/result events.

Key behaviors:
- Trials are derived from TrialStart/StimulusDelivered/TrialEnd entries.
- Reaction times ignore licks that occur before the trial start.
- Only licks on or after the trial start contribute to counts/RTs.
- is_rewarding is True only for trial_type == 'Go' (case-insensitive).
- Reward Delivery actions are split into trial-linked versus manual entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from behavioral_analysis.io.dataframe_builder import extract_events_by_type
from behavioral_analysis.io.hdf5_writer import save_to_hdf5
from behavioral_analysis.io.json_parser import parse_json_file


@dataclass
class OptostimConversionResult:
    output_h5: Path
    dataframes: Dict[str, pd.DataFrame]
    events: List[Dict]
    summary: Dict[str, int]


def normalize_matlab_logs(events: List[Dict]) -> pd.DataFrame:
    """Flatten MATLAB Log entries into a sortable DataFrame."""

    rows: List[Dict] = []
    for ev in events:
        if ev.get('msg') != 'Log':
            continue
        data = ev.get('data') or {}
        msg = data.get('msg', {}) or {}
        if not isinstance(msg, dict):
            continue

        action = msg.get('action') or msg.get('event')
        payload = msg.get('payload') or {}

        row: Dict = {
            'time_ms': ev.get('time'),
            'source': data.get('source'),
            'action': action,
        }

        if isinstance(payload, dict):
            for key, value in payload.items():
                row[key] = value

        for key in (
            'flashDuration_ms',
            'totalDuration_ms',
            'type',
            'amount',
            'valveTime',
            'withSound',
            'frequency',
            'duration',
            'rewarded',
            'outcome',
        ):
            if key in msg:
                row[key] = msg.get(key)

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('time_ms').reset_index(drop=True)
    return df


def _planned_params(row: pd.Series) -> Dict[str, float]:
    """Extract planned timing values from a TrialStart row."""

    return {
        'planned_response_window_s': row.get('responseWindow_s'),
        'planned_consumption_s': row.get('consumptionPeriod_s'),
        'planned_iti_s': row.get('itiMean_s'),
        'planned_iti_sd_s': row.get('itiSD_s'),
    }


def build_optostim_trials(
    action_log: pd.DataFrame,
    position_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Construct a trial table from MATLAB log actions.

    Reaction times ignore licks before trial start. is_rewarding is True only
    for trial_type == 'Go'.
    """

    starts = action_log[action_log['action'] == 'TrialStart'].copy()
    if starts.empty:
        return pd.DataFrame()

    if position_df is not None and not position_df.empty:
        pos_lookup = position_df[['time', 'position']].copy()
        pos_lookup['position_cm'] = pos_lookup['position'] / 100.0
        pos_lookup = pos_lookup.rename(columns={'time': 'time_ms'}).sort_values('time_ms')
    else:
        pos_lookup = None

    trials: List[Dict] = []

    start_rows = starts.sort_values('time_ms').reset_index(drop=True)
    next_starts = start_rows['time_ms'].shift(-1)

    for idx, start_row in start_rows.iterrows():
        trial_number = start_row.get('trialNumber')
        if pd.isna(trial_number):
            continue
        trial_number = int(trial_number)

        trial_rows = action_log[action_log['trialNumber'] == trial_number].sort_values('time_ms')

        stim_row = trial_rows[trial_rows['action'] == 'StimulusDelivered'].head(1)
        stim_time = stim_row['time_ms'].iloc[0] if not stim_row.empty else np.nan

        end_row = trial_rows[trial_rows['action'] == 'TrialEnd'].head(1)
        end_time = end_row['time_ms'].iloc[0] if not end_row.empty else np.nan
        outcome_raw = end_row['outcome'].iloc[0] if not end_row.empty else None
        rewarded = bool(end_row['rewarded'].iloc[0]) if not end_row.empty and 'rewarded' in end_row else False

        if outcome_raw is None:
            if not trial_rows[trial_rows['action'] == 'CorrectRejection'].empty:
                outcome_raw = 'CorrectRejection'
            elif not trial_rows[trial_rows['action'] == 'ImmediateFalseAlarm'].empty:
                outcome_raw = 'FalseAlarm'

        lick_rows = trial_rows[trial_rows['action'].isin(['LickDetected', 'ImmediateFalseAlarm'])]
        lick_rows = lick_rows[lick_rows['time_ms'] >= start_row.get('time_ms')]
        lick_times = lick_rows['time_ms'].to_numpy()
        num_licks = len(lick_times)

        first_lick_after_stim = np.nan
        stim_anchor = stim_time if not np.isnan(stim_time) else start_row.get('time_ms')
        if num_licks and not np.isnan(stim_anchor):
            after_mask = lick_times >= stim_anchor
            if after_mask.any():
                first_lick_after_stim = lick_times[after_mask].min()

        reaction_time_s = (
            (first_lick_after_stim - stim_anchor) / 1000.0
            if not np.isnan(first_lick_after_stim) and not np.isnan(stim_anchor)
            else np.nan
        )

        num_licks_pre = int((lick_times < stim_anchor).sum()) if num_licks and not np.isnan(stim_anchor) else 0
        num_licks_post = num_licks - num_licks_pre if num_licks else 0

        outcome_map = {'FalseAlarm': 'FA', 'CorrectRejection': 'CR', 'Hit': 'Hit', 'Miss': 'Miss'}
        outcome_label = outcome_map.get(outcome_raw, outcome_raw or 'Unknown')
        trial_type = start_row.get('trialType')
        is_rewarding = isinstance(trial_type, str) and trial_type.strip().lower() == 'go'
        correct = outcome_label in ['Hit', 'CR']
        was_hit = outcome_label in ['Hit', 'FA']

        stim_position_cm = np.nan
        if pos_lookup is not None and not np.isnan(stim_time):
            match = pd.merge_asof(
                pd.DataFrame({'time_ms': [stim_time]}),
                pos_lookup,
                on='time_ms',
                direction='nearest',
            )
            if not match.empty:
                stim_position_cm = match['position_cm'].iloc[0]

        planned = _planned_params(start_row)

        trials.append(
            {
                'trial_id': len(trials),
                'trial_number': trial_number,
                'trial_type': trial_type,
                'is_rewarding': is_rewarding,
                'start_time_ms': start_row.get('time_ms'),
                'stim_time_ms': stim_time,
                'end_time_ms': end_time,
                'trial_duration_ms': (end_time - start_row.get('time_ms')) if not np.isnan(end_time) else np.nan,
                'outcome_raw': outcome_raw,
                'outcome': outcome_label,
                'rewarded': rewarded,
                'correct': correct,
                'was_hit': was_hit,
                'num_licks_total': num_licks,
                'num_licks_pre_stim': num_licks_pre,
                'num_licks_post_stim': num_licks_post,
                'reaction_time_s': reaction_time_s,
                'first_lick_time_ms': float(lick_times.min()) if num_licks else np.nan,
                'first_lick_after_stim_ms': first_lick_after_stim,
                'session_time_min': start_row.get('time_ms') / 60000.0 if start_row.get('time_ms') is not None else np.nan,
                'mouse_global_position_cm': stim_position_cm,
                'global_position_cm': stim_position_cm,
                'num_licks_reward': num_licks_post,
                'corridor': 0,
                **planned,
                'planned_response_plus_consumption_s': (
                    planned['planned_response_window_s'] + planned['planned_consumption_s']
                    if pd.notna(planned.get('planned_response_window_s')) and pd.notna(planned.get('planned_consumption_s'))
                    else np.nan
                ),
                'planned_iti_plus_0sd_s': planned.get('planned_iti_s'),
                'iti_actual_s': np.nan,  # filled after loop
                'iti_delta_from_planned_s': np.nan,
            }
        )

    # Compute ITIs using next trial start
    for idx, row in enumerate(trials):
        next_start = next_starts.iloc[idx]
        if pd.notna(next_start) and pd.notna(row['end_time_ms']):
            iti_s = (next_start - row['end_time_ms']) / 1000.0
            trials[idx]['iti_actual_s'] = iti_s
            planned = row.get('planned_iti_s')
            if planned is not None and not pd.isna(planned):
                trials[idx]['iti_delta_from_planned_s'] = iti_s - planned

    return pd.DataFrame(trials)


def build_lick_table(action_log: pd.DataFrame, trials_df: pd.DataFrame) -> pd.DataFrame:
    """Return licks linked to trials; drops licks before trial start."""

    lick_events = action_log[action_log['action'].isin(['LickDetected', 'ImmediateFalseAlarm'])].copy()
    if lick_events.empty or trials_df.empty:
        return pd.DataFrame()

    lick_events = lick_events.rename(columns={'time_ms': 'lick_time_ms'})
    lick_events = lick_events.dropna(subset=['trialNumber'])
    lick_events['trialNumber'] = lick_events['trialNumber'].astype(int)

    trials_min_start = trials_df[['trial_number', 'start_time_ms']].rename(columns={'trial_number': 'trialNumber'})
    lick_events = lick_events.merge(trials_min_start, on='trialNumber', how='left')
    lick_events = lick_events[lick_events['lick_time_ms'] >= lick_events['start_time_ms']]

    merged = lick_events.merge(
        trials_df[['trial_id', 'trial_number', 'stim_time_ms', 'outcome', 'trial_type']],
        left_on='trialNumber',
        right_on='trial_number',
        how='left',
    )
    merged['time_from_stim_ms'] = merged['lick_time_ms'] - merged['stim_time_ms']
    merged['time_from_stim_s'] = merged['time_from_stim_ms'] / 1000.0
    merged['is_immediate_false_alarm'] = merged['action'] == 'ImmediateFalseAlarm'
    return merged


def build_reward_tables(action_log: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split reward deliveries into trial-linked vs manual."""

    rewards = action_log[action_log['action'] == 'Reward Delivery'].copy()
    if rewards.empty:
        return {'Reward': pd.DataFrame(), 'Manual_Reward': pd.DataFrame()}

    rewards = rewards.rename(columns={'time_ms': 'reward_time_ms'})
    if 'trialNumber' in rewards:
        has_trial = rewards['trialNumber'].notna()
    else:
        has_trial = pd.Series(False, index=rewards.index)

    if 'type' in rewards:
        is_manual = rewards['type'].astype(str).str.lower().eq('manual')
    else:
        is_manual = pd.Series(False, index=rewards.index)

    trial_rewards = rewards[has_trial & ~is_manual].copy()
    manual_rewards = rewards[~(has_trial & ~is_manual)].copy()

    return {'Reward': trial_rewards, 'Manual_Reward': manual_rewards}


def convert_optostim_session(
    json_path: Path,
    output_dir: Path,
    preloaded_events: Optional[List[Dict]] = None,
    verbose: bool = True,
) -> OptostimConversionResult:
    """Full optostim conversion wrapper for a single JSON file."""

    json_path = Path(json_path)
    events_local = preloaded_events if preloaded_events is not None else parse_json_file(str(json_path), verbose=False)
    dfs = extract_events_by_type(events_local, verbose=False)
    position_df = dfs.get('Position', pd.DataFrame())
    if not position_df.empty and 'position_cm' not in position_df.columns:
        position_df['position_cm'] = position_df['position'] / 100.0

    action_log = normalize_matlab_logs(events_local)
    trials_df = build_optostim_trials(action_log, position_df)
    lick_df = build_lick_table(action_log, trials_df)
    reward_tables = build_reward_tables(action_log)

    payload: Dict[str, pd.DataFrame] = {
        'Position': position_df,
        'Trials': trials_df,
        'Lick': lick_df,
        'Matlab_Action_Log': action_log,
        **reward_tables,
    }

    metadata = {
        'source_file': str(json_path),
        'note': 'Optostim MATLAB log conversion (trials derived from TrialStart/TrialEnd/LickDetected)',
        'num_trials': len(trials_df),
        'num_licks': len(lick_df),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = output_dir / f"{json_path.stem}_optostim.h5"

    save_to_hdf5(payload, str(output_h5), metadata=metadata, overwrite=True, verbose=verbose)

    summary = {k: len(v) for k, v in payload.items()}
    return OptostimConversionResult(
        output_h5=output_h5,
        dataframes=payload,
        events=events_local,
        summary=summary,
    )
