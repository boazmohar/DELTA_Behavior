"""
Simplified corridor detector based on Cue_State ID counting.

Corridors are defined by the cycling of cue IDs (0-6).
Each time we see ID=0 after ID=6, it's a new corridor.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


def detect_corridors_simple(
    cue_df: pd.DataFrame,
    position_df: Optional[pd.DataFrame] = None,
    corridor_length_cm: float = 200.0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Detect corridors based on Cue_State ID cycling (0-6).

    Args:
        cue_df: DataFrame with Cue_State events
        position_df: Optional position data
        corridor_length_cm: Length of each corridor in cm
        verbose: Print progress information

    Returns:
        Tuple of (corridor_info, position_with_corridors)
    """
    if verbose:
        print("Detecting corridors based on Cue_State ID cycling...")

    # Sort by time
    cue_df = cue_df.sort_values('time').reset_index(drop=True)

    # Detect corridor transitions based on cue ID cycling
    corridor_starts = []
    corridor_counter = 0

    # First corridor starts with the first cue
    if len(cue_df) > 0:
        corridor_starts.append({
            'corridor_id': 0,
            'start_time': cue_df.iloc[0]['time'],
            'trigger': 'first_cue'
        })

    # Find transitions where ID goes from any value to 0 (except the first)
    for idx in range(1, len(cue_df)):
        if cue_df.iloc[idx]['id'] == 0:
            corridor_counter += 1
            corridor_starts.append({
                'corridor_id': corridor_counter,
                'start_time': cue_df.iloc[idx]['time'],
                'trigger': 'cue_reset'
            })

    # Convert to DataFrame
    corridor_info = pd.DataFrame(corridor_starts)

    # Add end times
    for idx in range(len(corridor_info) - 1):
        corridor_info.loc[idx, 'end_time'] = corridor_info.loc[idx + 1, 'start_time']

    # Last corridor ends at the last event time
    if position_df is not None and len(position_df) > 0:
        corridor_info.loc[len(corridor_info) - 1, 'end_time'] = position_df['time'].max()
    else:
        corridor_info.loc[len(corridor_info) - 1, 'end_time'] = cue_df['time'].max()

    if verbose:
        print(f"Detected {len(corridor_info)} corridors")

    # Process position data if provided
    position_with_corridors = None
    if position_df is not None:
        position_with_corridors = add_corridor_to_position(
            position_df,
            corridor_info,
            corridor_length_cm,
            verbose
        )

    return corridor_info, position_with_corridors


def add_corridor_to_position(
    position_df: pd.DataFrame,
    corridor_info: pd.DataFrame,
    corridor_length_cm: float = 200.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Add corridor IDs and global position to position data.
    Creates cumulative position by adding position values across teleports.

    Args:
        position_df: Position DataFrame
        corridor_info: Corridor information
        corridor_length_cm: Length of each corridor
        verbose: Print progress

    Returns:
        Position DataFrame with corridor and global position
    """
    position_df = position_df.copy().sort_values('time').reset_index(drop=True)

    # Assign corridor IDs based on time
    position_df['corridor_id'] = np.nan

    for _, corridor in corridor_info.iterrows():
        mask = (position_df['time'] >= corridor['start_time']) & \
               (position_df['time'] < corridor['end_time'])
        position_df.loc[mask, 'corridor_id'] = corridor['corridor_id']

    # Create cumulative position that handles teleports
    # Work with raw position values first
    position_df['cumulative_position'] = position_df['position'].copy()

    # Detect teleports (large backward jumps in raw position)
    position_df['pos_diff'] = position_df['position'].diff()
    teleport_mask = position_df['pos_diff'] < -30000

    if teleport_mask.any():
        teleport_indices = position_df[teleport_mask].index.tolist()

        # Track cumulative offset from teleports
        cumulative_offset = 0

        for i, tel_idx in enumerate(teleport_indices):
            if tel_idx > 0:
                # Get the last value before teleport
                last_value_before = position_df.loc[tel_idx - 1, 'cumulative_position']
                # Get the first value after teleport (current position)
                first_value_after = position_df.loc[tel_idx, 'position']

                # Calculate new offset: last cumulative value + current position value
                # This ensures continuity: the new position continues from where we left off
                new_offset = last_value_before

                # Apply this offset to all positions from this teleport onwards
                position_df.loc[tel_idx:, 'cumulative_position'] = (
                    position_df.loc[tel_idx:, 'position'] + new_offset
                )

                cumulative_offset = new_offset

        if verbose:
            print(f"  Detected {len(teleport_indices)} teleports, created cumulative position")

    # Now convert cumulative position to cm
    # Using the original conversion factor: 50000 units = 200 cm
    position_df['position_cm'] = position_df['position'] / 250.0
    position_df['cumulative_position_cm'] = position_df['cumulative_position'] / 250.0

    # Calculate global position using corridor offset and cumulative position
    # For corridor 1 (id=0), offset is 0; for corridor 2 (id=1), offset is 200cm, etc.
    mask = ~position_df['corridor_id'].isna()

    # Use cumulative position for events within corridors
    position_df.loc[mask, 'global_position_cm'] = (
        (position_df.loc[mask, 'corridor_id'] - 1) * corridor_length_cm +
        position_df.loc[mask, 'cumulative_position_cm'] % corridor_length_cm
    )

    # Alternative: Just use the cumulative position directly as global position
    # This might be more accurate since teleports happen within corridors
    position_df.loc[mask, 'global_position_cm'] = position_df.loc[mask, 'cumulative_position_cm']

    # Drop helper columns
    position_df = position_df.drop(columns=['pos_diff', 'cumulative_position', 'cumulative_position_cm'], errors='ignore')

    if verbose:
        n_assigned = mask.sum()
        total = len(position_df)
        print(f"  Assigned corridor IDs to {n_assigned}/{total} position events ({n_assigned/total*100:.1f}%)")

        # Check monotonicity
        position_sorted = position_df.sort_values('time')
        global_diff = position_sorted['global_position_cm'].diff()
        n_negative = (global_diff < -0.1).sum()  # Allow tiny rounding errors
        if n_negative > 0:
            print(f"  WARNING: {n_negative} non-monotonic global position events remain")
        else:
            print(f"  ✓ Global position is monotonically increasing")

    return position_df


def add_corridor_info_to_events(
    dataframes: Dict[str, pd.DataFrame],
    corridor_info: pd.DataFrame,
    corridor_length_cm: float = 200.0,
    verbose: bool = True,
    position_df: Optional[pd.DataFrame] = None
) -> Dict[str, pd.DataFrame]:
    """
    Add corridor and global position information to all event types.

    Args:
        dataframes: Dictionary of event DataFrames
        corridor_info: Corridor information
        corridor_length_cm: Length of each corridor
        verbose: Print progress
        position_df: Position dataframe with cumulative positions (for reference)

    Returns:
        Updated dictionary of DataFrames
    """
    updated = {}

    # Get teleport information from position_df if available
    teleport_times = []
    teleport_offsets = {}

    if position_df is not None and 'Position' in dataframes:
        # The Position df already has corrected global positions
        # We'll use it as reference for other events
        pos_sorted = position_df.sort_values('time').reset_index(drop=True)
        pos_sorted['pos_diff'] = pos_sorted['position'].diff()
        teleport_mask = pos_sorted['pos_diff'] < -30000

        if teleport_mask.any():
            # Build a map of teleport times and their cumulative offsets
            cumulative_offset = 0
            for idx in pos_sorted[teleport_mask].index:
                if idx > 0:
                    tel_time = pos_sorted.loc[idx, 'time']
                    last_value_before = pos_sorted.loc[idx - 1, 'position']
                    cumulative_offset += last_value_before
                    teleport_times.append(tel_time)
                    teleport_offsets[tel_time] = cumulative_offset

    for event_type, df in dataframes.items():
        if df is None or len(df) == 0:
            updated[event_type] = df
            continue

        df = df.copy().sort_values('time').reset_index(drop=True)

        # Assign corridor IDs based on time
        df['corridor_id'] = np.nan

        for _, corridor in corridor_info.iterrows():
            mask = (df['time'] >= corridor['start_time']) & \
                   (df['time'] < corridor['end_time'])
            df.loc[mask, 'corridor_id'] = corridor['corridor_id']

        # If has position, calculate global position
        if 'position' in df.columns and event_type != 'Position':
            # For Cue_State and Cue_Result, positions are predefined cue locations
            # They represent fixed positions in the virtual environment
            # They should NOT get cumulative offset - only corridor offset
            if event_type in ['Cue State', 'Cue_State', 'Cue Result', 'Cue_Result']:
                # Convert to cm
                df['position_cm'] = df['position'] / 250.0

                # Global position for cues: just corridor offset + position
                # NO cumulative offset because cues are fixed in virtual space
                mask = ~df['corridor_id'].isna()
                df.loc[mask, 'global_position_cm'] = (
                    df.loc[mask, 'corridor_id'] * corridor_length_cm +
                    df.loc[mask, 'position_cm']
                )
            else:
                # For other events with position, apply cumulative offset
                df['cumulative_offset'] = 0

                for tel_time, offset in teleport_offsets.items():
                    # Apply offset to all events after this teleport
                    df.loc[df['time'] >= tel_time, 'cumulative_offset'] = offset

                # Add offset to position before converting
                df['cumulative_position'] = df['position'] + df['cumulative_offset']

                # Convert to cm
                df['position_cm'] = df['position'] / 250.0
                df['global_position_cm'] = df['cumulative_position'] / 250.0

                # Drop helper columns
                df = df.drop(columns=['cumulative_offset', 'cumulative_position'], errors='ignore')

        updated[event_type] = df

    if verbose:
        print(f"Added corridor information to {len(updated)} event types")

    return updated