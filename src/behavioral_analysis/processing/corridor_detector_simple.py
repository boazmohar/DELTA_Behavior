"""
Corridor detection helpers for DELTA behavior sessions.

The helpers in this module expose small, composable steps so that notebooks can
inspect intermediate results (cue/corridor assignments, position loops, etc.)
while the main pipeline still offers the `detect_corridors_simple` convenience
function used by the HDF5 export.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from behavioral_analysis.processing.trial_matcher import match_cues_robust


@dataclass
class CorridorComputationArtifacts:
    """Container for corridor detection artifacts."""

    corridor_info: pd.DataFrame
    position_loops: pd.DataFrame
    cue_state_with_corridors: pd.DataFrame
    cue_result_with_corridors: Optional[pd.DataFrame]
    cue_matches: Optional[pd.DataFrame]


def _ensure_dataframe(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.copy()


def _nanmin(*series: pd.Series) -> pd.Series:
    if not series:
        return pd.Series(dtype=float)
    concat = pd.concat(series, axis=1)
    values = concat.to_numpy(dtype=float)
    return pd.Series(np.nanmin(values, axis=1), index=concat.index)


def _nanmax(*series: pd.Series) -> pd.Series:
    if not series:
        return pd.Series(dtype=float)
    concat = pd.concat(series, axis=1)
    values = concat.to_numpy(dtype=float)
    return pd.Series(np.nanmax(values, axis=1), index=concat.index)


def annotate_cue_states_with_corridors(cue_state_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add `corridor_id`/`cue_index` columns to Cue_State events."""

    cue_state_df = _ensure_dataframe(cue_state_df)
    if cue_state_df.empty or 'id' not in cue_state_df:
        return cue_state_df

    cue_state_df = cue_state_df.sort_values('time').reset_index(drop=True)

    corridor_ids: List[int] = []
    corridor_id = 0
    for idx, cue_id in enumerate(cue_state_df['id']):
        if idx > 0 and cue_id == 0:
            corridor_id += 1
        corridor_ids.append(corridor_id)

    cue_state_df['corridor_id'] = corridor_ids
    cue_state_df['cue_index'] = cue_state_df.groupby('corridor_id').cumcount()

    return cue_state_df


def annotate_cue_results_with_corridors(cue_result_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Add `corridor_id`/`cue_index` columns to Cue_Result events."""

    cue_result_df = _ensure_dataframe(cue_result_df)
    if cue_result_df.empty or 'id' not in cue_result_df:
        return cue_result_df

    cue_result_df = cue_result_df.sort_values('time').reset_index(drop=True)

    corridor_flow = cue_result_df['id'].eq(0).cumsum()
    if len(corridor_flow) > 0 and corridor_flow.iloc[0] == 0:
        corridor_flow += 1
    cue_result_df['corridor_id'] = corridor_flow - 1
    cue_result_df['cue_index'] = cue_result_df.groupby('corridor_id').cumcount()

    return cue_result_df


def match_cue_states_to_results(
    cue_state_df: pd.DataFrame,
    cue_result_df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """Match Cue_State and Cue_Result rows using existing robust matcher."""

    cue_state_df = _ensure_dataframe(cue_state_df)
    cue_result_df = _ensure_dataframe(cue_result_df)

    if cue_state_df.empty or cue_result_df.empty:
        return pd.DataFrame(
            columns=[
                'state_idx',
                'result_idx',
                'corridor_id',
                'cue_id',
                'state_time',
                'result_time',
                'state_position',
                'result_position',
                'is_rewarding',
                'has_given_reward',
                'num_licks_reward',
                'num_licks_pre',
                'reaction_time_ms',
            ]
        )

    matches = match_cues_robust(cue_state_df, cue_result_df, verbose=verbose)

    rows = []
    for match in matches:
        state = match['state']
        result = match['result']
        state_time = state.get('time')
        result_time = result.get('time')

        rows.append(
            {
                'state_idx': match.get('state_idx'),
                'result_idx': match.get('result_idx'),
                'corridor_id': result.get('corridor_id', np.nan),
                'cue_id': state.get('id', np.nan),
                'state_time': state_time,
                'result_time': result_time,
                'state_position': state.get('position', np.nan),
                'result_position': result.get('position', np.nan),
                'is_rewarding': state.get('isRewarding'),
                'has_given_reward': result.get('hasGivenReward'),
                'num_licks_reward': result.get('numLicksInReward'),
                'num_licks_pre': result.get('numLicksInPre'),
                'reaction_time_ms': result_time - state_time if None not in (state_time, result_time) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def detect_position_loops(
    position_df: Optional[pd.DataFrame],
    low_threshold: float = 1500.0,
    high_threshold: float = 45000.0,
    verbose: bool = True
) -> pd.DataFrame:
    """Detect corridor traversals from path position data."""

    position_df = _ensure_dataframe(position_df)
    if position_df.empty or 'position' not in position_df:
        return position_df.head(0)

    df = position_df.sort_values('time').reset_index(drop=True)

    loops: List[Dict[str, float]] = []
    in_loop = False
    start_idx = 0
    start_time = 0.0
    start_pos = 0.0
    min_pos = np.inf
    max_pos = -np.inf
    reached_high = False

    for idx, row in df.iterrows():
        pos = row['position']
        if not in_loop:
            if pos <= low_threshold:
                in_loop = True
                start_idx = idx
                start_time = row['time']
                start_pos = pos
                min_pos = pos
                max_pos = pos
                reached_high = pos >= high_threshold
            continue

        min_pos = min(min_pos, pos)
        max_pos = max(max_pos, pos)
        if not reached_high and pos >= high_threshold:
            reached_high = True

        if reached_high and pos <= low_threshold:
            end_idx = idx
            end_time = row['time']
            loops.append(
                {
                    'corridor_id': len(loops),
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_position': start_pos,
                    'end_position': pos,
                    'min_position': float(min_pos),
                    'max_position': float(max_pos),
                    'loop_complete': True,
                }
            )
            in_loop = False

    if in_loop:
        end_row = df.iloc[-1]
        loops.append(
            {
                'corridor_id': len(loops),
                'start_time': start_time,
                'end_time': end_row['time'],
                'start_position': start_pos,
                'end_position': end_row['position'],
                'min_position': float(min_pos if np.isfinite(min_pos) else start_pos),
                'max_position': float(max_pos if np.isfinite(max_pos) else start_pos),
                'loop_complete': False,
            }
        )

    loops_df = pd.DataFrame(loops)
    if verbose and not loops_df.empty:
        print(f"  Detected {len(loops_df)} position-based corridor loops")

    return loops_df


def _corridor_table_from_cue_states(cue_state_df: pd.DataFrame) -> pd.DataFrame:
    if cue_state_df.empty:
        return cue_state_df.head(0)

    grouped = cue_state_df.groupby('corridor_id')['time'].agg(['min', 'max']).reset_index()
    grouped.rename(columns={'min': 'start_time', 'max': 'end_time'}, inplace=True)
    grouped['start_position'] = cue_state_df.groupby('corridor_id')['position'].first().values
    grouped['end_position'] = cue_state_df.groupby('corridor_id')['position'].last().values
    grouped['min_position'] = cue_state_df.groupby('corridor_id')['position'].min().values
    grouped['max_position'] = cue_state_df.groupby('corridor_id')['position'].max().values
    grouped['loop_complete'] = pd.NA
    return grouped


def summarize_corridor_info(
    position_loops: pd.DataFrame,
    cue_state_df: pd.DataFrame,
    cue_result_df: Optional[pd.DataFrame] = None,
    cue_matches: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Combine timing/statistics from position loops and cue matches."""

    position_loops = _ensure_dataframe(position_loops)
    cue_state_df = _ensure_dataframe(cue_state_df)
    cue_result_df = _ensure_dataframe(cue_result_df)
    cue_matches = _ensure_dataframe(cue_matches)

    if position_loops.empty:
        base = _corridor_table_from_cue_states(cue_state_df)
    else:
        base = position_loops.copy()

    if base.empty:
        return base

    base = base.sort_values('corridor_id').reset_index(drop=True)

    state_stats = pd.DataFrame()
    if not cue_state_df.empty:
        state_stats = cue_state_df.groupby('corridor_id')['time'].agg(
            first_state_time='min',
            last_state_time='max',
        )
    if not state_stats.empty:
        base = base.merge(state_stats, on='corridor_id', how='left')

    result_stats = pd.DataFrame()
    if not cue_result_df.empty:
        result_stats = cue_result_df.groupby('corridor_id')['time'].agg(
            first_cue_time='min',
            last_cue_time='max',
        )
    if not result_stats.empty:
        base = base.merge(result_stats, on='corridor_id', how='left')

    match_stats = pd.DataFrame()
    if not cue_matches.empty:
        match_stats = cue_matches.groupby('corridor_id').agg(
            num_matched_cues=('cue_id', 'count'),
            first_match_time=('result_time', 'min'),
            last_match_time=('result_time', 'max'),
        )
    if not match_stats.empty:
        base = base.merge(match_stats, on='corridor_id', how='left')

    # Ensure start/end times cover all cue events
    start_candidates = [base.get('start_time')]
    if 'first_state_time' in base:
        start_candidates.append(base['first_state_time'])
    if 'first_cue_time' in base:
        start_candidates.append(base['first_cue_time'])
    if 'first_match_time' in base:
        start_candidates.append(base['first_match_time'])
    valid_start = [s for s in start_candidates if s is not None]
    base['start_time'] = _nanmin(*valid_start) if valid_start else pd.Series(np.nan, index=base.index)

    end_candidates = [base.get('end_time')]
    if 'last_state_time' in base:
        end_candidates.append(base['last_state_time'])
    if 'last_cue_time' in base:
        end_candidates.append(base['last_cue_time'])
    if 'last_match_time' in base:
        end_candidates.append(base['last_match_time'])
    valid_end = [s for s in end_candidates if s is not None]
    base['end_time'] = _nanmax(*valid_end) if valid_end else pd.Series(np.nan, index=base.index)

    # Guard against missing data (e.g., synthetic tests with minimal info)
    base['start_time'] = base['start_time'].fillna(0.0)
    base['end_time'] = base['end_time'].fillna(base['start_time'])

    # Add a tiny epsilon so strict < end comparisons keep the last event
    epsilon = 1e-6
    base['end_time'] = base['end_time'] + epsilon

    # Keep corridor ordering sane relative to the next corridor start
    for idx in range(len(base) - 1):
        next_start = base.loc[idx + 1, 'start_time']
        if pd.notna(next_start):
            base.loc[idx, 'end_time'] = min(base.loc[idx, 'end_time'], next_start)

    base['duration_ms'] = base['end_time'] - base['start_time']
    base['start_position_cm'] = base['start_position'] / 250.0
    base['end_position_cm'] = base['end_position'] / 250.0
    base['max_position_cm'] = base['max_position'] / 250.0
    base['trigger'] = np.where(base['corridor_id'] == 0, 'first_cue', 'cue_reset')

    if not cue_result_df.empty:
        counts = cue_result_df.groupby('corridor_id').agg(
            num_cue_results=('id', 'count'),
        )
        base = base.merge(counts, on='corridor_id', how='left')
    else:
        base['num_cue_results'] = np.nan

    if 'num_matched_cues' not in base:
        base['num_matched_cues'] = np.nan

    return base


def compute_corridor_artifacts(
    cue_state_df: Optional[pd.DataFrame],
    position_df: Optional[pd.DataFrame] = None,
    cue_result_df: Optional[pd.DataFrame] = None,
    corridor_length_cm: float = 500.0,
    verbose: bool = True,
) -> CorridorComputationArtifacts:
    """Derive all intermediate artifacts needed to describe corridors."""

    if verbose:
        print("Detecting corridor structure...")

    # Retained for API compatibility – corridor length is applied later when
    # global positions are generated but isn't needed for the artifact summary.
    _ = corridor_length_cm

    cue_state_with_corridors = annotate_cue_states_with_corridors(cue_state_df)
    cue_result_with_corridors = annotate_cue_results_with_corridors(cue_result_df)
    position_loops = detect_position_loops(position_df, verbose=verbose)

    cue_matches = match_cue_states_to_results(
        cue_state_with_corridors,
        cue_result_with_corridors,
        verbose=verbose,
    ) if not cue_state_with_corridors.empty and not cue_result_with_corridors.empty else pd.DataFrame()

    corridor_info = summarize_corridor_info(
        position_loops,
        cue_state_with_corridors,
        cue_result_with_corridors,
        cue_matches,
    )

    return CorridorComputationArtifacts(
        corridor_info=corridor_info,
        position_loops=position_loops,
        cue_state_with_corridors=cue_state_with_corridors,
        cue_result_with_corridors=cue_result_with_corridors if not cue_result_with_corridors.empty else None,
        cue_matches=cue_matches if not cue_matches.empty else None,
    )


def detect_corridors_simple(
    cue_df: pd.DataFrame,
    position_df: Optional[pd.DataFrame] = None,
    corridor_length_cm: float = 500.0,
    verbose: bool = True,
    cue_result_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Pipeline-friendly wrapper returning corridor info and annotated positions."""

    artifacts = compute_corridor_artifacts(
        cue_state_df=cue_df,
        position_df=position_df,
        cue_result_df=cue_result_df,
        corridor_length_cm=corridor_length_cm,
        verbose=verbose,
    )

    if artifacts.corridor_info.empty and verbose:
        print("  No corridor boundaries detected")

    position_with_corridors = None
    if position_df is not None and not artifacts.corridor_info.empty:
        position_with_corridors = add_corridor_to_position(
            position_df,
            artifacts.corridor_info,
            corridor_length_cm,
            verbose,
        )

    return artifacts.corridor_info, position_with_corridors


def add_corridor_to_position(
    position_df: pd.DataFrame,
    corridor_info: pd.DataFrame,
    corridor_length_cm: float = 500.0,
    verbose: bool = True,
) -> pd.DataFrame:
    position_df = position_df.copy().sort_values('time').reset_index(drop=True)

    position_df['corridor_id'] = np.nan
    for _, corridor in corridor_info.iterrows():
        mask = (position_df['time'] >= corridor['start_time']) & (position_df['time'] < corridor['end_time'])
        position_df.loc[mask, 'corridor_id'] = corridor['corridor_id']

    position_df['cumulative_position'] = position_df['position'].copy()
    position_df['pos_diff'] = position_df['position'].diff()
    teleport_mask = position_df['pos_diff'] < -30000

    if teleport_mask.any():
        teleport_indices = position_df[teleport_mask].index.tolist()
        cumulative_offset = 0
        for tel_idx in teleport_indices:
            if tel_idx > 0:
                last_value_before = position_df.loc[tel_idx - 1, 'cumulative_position']
                position_df.loc[tel_idx:, 'cumulative_position'] = (
                    position_df.loc[tel_idx:, 'position'] + last_value_before
                )
                cumulative_offset = last_value_before
        if verbose:
            print(f"  Detected {len(teleport_indices)} teleports, created cumulative position")

    position_df['position_cm'] = position_df['position'] / 250.0
    position_df['cumulative_position_cm'] = position_df['cumulative_position'] / 250.0

    mask = ~position_df['corridor_id'].isna()
    position_df.loc[mask, 'global_position_cm'] = position_df.loc[mask, 'cumulative_position_cm']

    position_df = position_df.drop(columns=['pos_diff', 'cumulative_position', 'cumulative_position_cm'], errors='ignore')

    if verbose:
        n_assigned = mask.sum()
        total = len(position_df)
        print(f"  Assigned corridor IDs to {n_assigned}/{total} position events ({n_assigned / total * 100:.1f}%)")
        position_sorted = position_df.sort_values('time')
        global_diff = position_sorted['global_position_cm'].diff()
        n_negative = (global_diff < -0.1).sum()
        if n_negative > 0:
            print(f"  WARNING: {n_negative} non-monotonic global position events remain")
        else:
            print("  ✓ Global position is monotonically increasing")

    return position_df


def add_corridor_info_to_events(
    dataframes: Dict[str, pd.DataFrame],
    corridor_info: pd.DataFrame,
    corridor_length_cm: float = 500.0,
    verbose: bool = True,
    position_df: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    updated = {}

    teleport_offsets: Dict[float, float] = {}

    if position_df is not None and 'Position' in dataframes:
        pos_sorted = position_df.sort_values('time').reset_index(drop=True)
        pos_sorted['pos_diff'] = pos_sorted['position'].diff()
        teleport_mask = pos_sorted['pos_diff'] < -30000

        if teleport_mask.any():
            cumulative_offset = 0
            for idx in pos_sorted[teleport_mask].index:
                if idx > 0:
                    tel_time = pos_sorted.loc[idx, 'time']
                    last_value_before = pos_sorted.loc[idx - 1, 'position']
                    cumulative_offset += last_value_before
                    teleport_offsets[tel_time] = cumulative_offset

    for event_type, df in dataframes.items():
        if df is None or len(df) == 0:
            updated[event_type] = df
            continue

        df = df.copy().sort_values('time').reset_index(drop=True)

        if event_type in ['Cue State', 'Cue_State']:
            df = annotate_cue_states_with_corridors(df)
        elif event_type in ['Cue Result', 'Cue_Result']:
            df = annotate_cue_results_with_corridors(df)
        else:
            df['corridor_id'] = np.nan
            for _, corridor in corridor_info.iterrows():
                mask = (df['time'] >= corridor['start_time']) & (df['time'] < corridor['end_time'])
                df.loc[mask, 'corridor_id'] = corridor['corridor_id']

        if 'position' in df.columns and event_type != 'Position':
            if event_type in ['Cue State', 'Cue_State', 'Cue Result', 'Cue_Result']:
                df['position_cm'] = df['position'] / 250.0
                mask = ~df['corridor_id'].isna()
                df.loc[mask, 'global_position_cm'] = (
                    df.loc[mask, 'corridor_id'] * corridor_length_cm + df.loc[mask, 'position_cm']
                )
            else:
                df['cumulative_offset'] = 0
                for tel_time, offset in teleport_offsets.items():
                    df.loc[df['time'] >= tel_time, 'cumulative_offset'] = offset
                df['cumulative_position'] = df['position'] + df['cumulative_offset']
                df['position_cm'] = df['position'] / 250.0
                df['global_position_cm'] = df['cumulative_position'] / 250.0
                df = df.drop(columns=['cumulative_offset', 'cumulative_position'], errors='ignore')

        updated[event_type] = df

    if verbose:
        print(f"Added corridor information to {len(updated)} event types")

    return updated
