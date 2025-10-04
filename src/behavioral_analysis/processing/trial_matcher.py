"""
Trial matching module for behavioral analysis.
Robustly matches Cue_State to Cue_Result events to create trial-based dataframes.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List


def match_cues_robust(
    cue_state_df: pd.DataFrame,
    cue_result_df: pd.DataFrame,
    verbose: bool = True
) -> List[Dict]:
    """
    Robustly match Cue_State to Cue_Result events.

    Uses a multi-phase matching approach:
    1. Match by exact position AND time constraint (result must occur after state)
    2. Remove matched items from both sets before proceeding
    3. Returns only matched cues (unmatched are considered not hit)

    Args:
        cue_state_df: DataFrame of Cue_State events
        cue_result_df: DataFrame of Cue_Result events
        verbose: Whether to print matching statistics

    Returns:
        List of match dictionaries containing state and result pairs
    """
    # Sort by time to ensure chronological order
    cue_state_df = cue_state_df.sort_values('time').reset_index(drop=True)
    cue_result_df = cue_result_df.sort_values('time').reset_index(drop=True)

    if verbose:
        print(f"Matching {len(cue_state_df)} Cue_State events to {len(cue_result_df)} Cue_Result events...")

    # Create working copies with indices
    state_working = cue_state_df.copy()
    result_working = cue_result_df.copy()
    state_working['original_index'] = state_working.index
    result_working['original_index'] = result_working.index

    # Track matches
    matches = []
    matched_state_indices = set()
    matched_result_indices = set()

    # Phase 1: Match by position with time constraint
    for idx, state in state_working.iterrows():
        if idx in matched_state_indices:
            continue

        # Find results with same position that occur after this state
        potential_matches = result_working[
            (~result_working['original_index'].isin(matched_result_indices)) &
            (result_working['position'] == state['position']) &
            (result_working['time'] > state['time'])
        ]

        if len(potential_matches) > 0:
            # Take the first matching result (earliest in time)
            result = potential_matches.iloc[0]

            matches.append({
                'state_idx': idx,
                'result_idx': result['original_index'],
                'state': state,
                'result': result
            })

            matched_state_indices.add(idx)
            matched_result_indices.add(result['original_index'])

    if verbose:
        print(f"  Matched {len(matches)} cues ({len(matches)/len(cue_state_df)*100:.1f}%)")
        print(f"  Unmatched (not hit): {len(cue_state_df) - len(matches)} cues")

    return matches


def create_trial_dataframe(
    cue_state_df: pd.DataFrame,
    cue_result_df: pd.DataFrame,
    corridor_length_cm: float = 500.0,
    verbose: bool = True,
    position_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Create a trial-based dataframe from Cue_State and Cue_Result events.

    Each trial represents a cue that was presented and responded to.
    Trials are classified into behavioral outcomes:
    - Hit: Rewarding cue + licked
    - Miss: Rewarding cue + no lick
    - FA (False Alarm): Non-rewarding cue + licked
    - CR (Correct Rejection): Non-rewarding cue + no lick

    Args:
        cue_state_df: DataFrame of Cue_State events
        cue_result_df: DataFrame of Cue_Result events
        corridor_length_cm: Length of each corridor in cm
        verbose: Whether to print processing statistics

    Returns:
        DataFrame with one row per trial
    """
    # Calculate corridor based on cue counting (7 cues per corridor, IDs 0-6)
    corridor_counter = 0
    cue_state_corridors = []

    cue_state_df = cue_state_df.sort_values('time').reset_index(drop=True)

    for idx, row in cue_state_df.iterrows():
        cue_id = row['id']
        if cue_id == 0 and idx > 0:
            corridor_counter += 1
        cue_state_corridors.append(corridor_counter)

    cue_state_df['cue_corridor'] = cue_state_corridors

    # Convert positions to cm
    cue_state_df['position_cm'] = cue_state_df['position'] / 100.0
    cue_result_df['position_cm'] = cue_result_df['position'] / 100.0

    # Calculate global position
    cue_state_df['cue_global_position_cm'] = (
        cue_state_df['cue_corridor'] * corridor_length_cm +
        cue_state_df['position_cm']
    )

    # Match cues
    matches = match_cues_robust(cue_state_df, cue_result_df, verbose)

    # Build trial dataframe
    trials = []

    for trial_num, match in enumerate(matches):
        state = match['state']
        result = match['result']

        # Determine trial outcome
        is_rewarding = state.get('isRewarding', False)
        gave_reward = result.get('hasGivenReward', False)
        num_licks = result.get('numLicksInReward', 0)

        # Classify outcome
        if is_rewarding:
            outcome = 'Hit' if (gave_reward or num_licks > 0) else 'Miss'
        else:
            outcome = 'FA' if (gave_reward or num_licks > 0) else 'CR'

        # Use the STATE's global position for cue (fixed position in virtual space)
        # But use the RESULT's corridor for where it was actually hit
        cue_global_pos = state['cue_global_position_cm']
        result_corridor = result.get('corridor_id', state['cue_corridor'])

        trial = {
            # Trial identifiers
            'trial_id': trial_num,
            'corridor': result_corridor,  # Use corridor where cue was hit
            'cue_id_in_corridor': state['id'],

            # Cue properties
            'cue_type': 'Rewarding' if is_rewarding else 'Non-rewarding',
            'is_rewarding': is_rewarding,

            # Position information (cue's fixed position in virtual space)
            'cue_position': state['position'],
            'cue_position_cm': state['position_cm'],
            'global_position_cm': cue_global_pos,  # Use state's global position

            # Timing
            'cue_onset_ms': state['time'],
            'cue_hit_ms': result['time'],
            'reaction_time_s': (result['time'] - state['time']) / 1000.0,

            # Response
            'num_licks_pre': result.get('numLicksInPre', 0),
            'num_licks_reward': num_licks,
            'gave_reward': gave_reward,

            # Outcome
            'outcome': outcome,
            'correct': outcome in ['Hit', 'CR'],
        }

        trials.append(trial)

    # Create DataFrame
    trials_df = pd.DataFrame(trials)

    # Add derived metrics
    trials_df['licked'] = trials_df['outcome'].isin(['Hit', 'FA'])
    trials_df['session_time_min'] = trials_df['cue_onset_ms'] / 60000.0

    # Add mouse position at hit time if position data is provided
    if position_df is not None and len(trials_df) > 0:
        if verbose:
            print("  Adding mouse position at hit time...")

        # Sort position by time for efficient lookup
        position_sorted = position_df.sort_values('time').reset_index(drop=True)

        mouse_positions = []
        mouse_global_positions = []

        for _, trial in trials_df.iterrows():
            # Find closest position event to hit time
            time_diffs = abs(position_sorted['time'] - trial['cue_hit_ms'])
            if len(time_diffs) > 0:
                closest_idx = time_diffs.idxmin()
                mouse = position_sorted.loc[closest_idx]
                mouse_positions.append(mouse.get('position', np.nan))
                mouse_global_positions.append(mouse.get('global_position_cm', np.nan))
            else:
                mouse_positions.append(np.nan)
                mouse_global_positions.append(np.nan)

        trials_df['mouse_position'] = mouse_positions
        trials_df['mouse_global_position_cm'] = mouse_global_positions

    if verbose:
        print(f"\nCreated trial dataframe with {len(trials_df)} trials")
        print(f"  Outcomes: Hit={(trials_df['outcome']=='Hit').sum()}, "
              f"Miss={( trials_df['outcome']=='Miss').sum()}, "
              f"FA={(trials_df['outcome']=='FA').sum()}, "
              f"CR={(trials_df['outcome']=='CR').sum()}")

    return trials_df


def calculate_performance_metrics(trials_df: pd.DataFrame) -> Dict:
    """
    Calculate behavioral performance metrics from trial dataframe.

    Args:
        trials_df: Trial dataframe from create_trial_dataframe()

    Returns:
        Dictionary of performance metrics
    """
    metrics = {}

    # Basic counts
    metrics['n_trials'] = len(trials_df)
    metrics['n_hit'] = (trials_df['outcome'] == 'Hit').sum()
    metrics['n_miss'] = (trials_df['outcome'] == 'Miss').sum()
    metrics['n_fa'] = (trials_df['outcome'] == 'FA').sum()
    metrics['n_cr'] = (trials_df['outcome'] == 'CR').sum()

    # Rates
    n_rewarding = trials_df['is_rewarding'].sum()
    n_non_rewarding = (~trials_df['is_rewarding']).sum()

    metrics['hit_rate'] = metrics['n_hit'] / n_rewarding if n_rewarding > 0 else 0
    metrics['miss_rate'] = metrics['n_miss'] / n_rewarding if n_rewarding > 0 else 0
    metrics['fa_rate'] = metrics['n_fa'] / n_non_rewarding if n_non_rewarding > 0 else 0
    metrics['cr_rate'] = metrics['n_cr'] / n_non_rewarding if n_non_rewarding > 0 else 0

    # Overall accuracy
    metrics['accuracy'] = trials_df['correct'].mean()

    # D-prime (if applicable)
    hit_rate = metrics['hit_rate']
    fa_rate = metrics['fa_rate']

    if 0 < hit_rate < 1 and 0 < fa_rate < 1:
        try:
            from scipy import stats
            hit_z = stats.norm.ppf(hit_rate)
            fa_z = stats.norm.ppf(fa_rate)
            metrics['d_prime'] = hit_z - fa_z
        except:
            metrics['d_prime'] = np.nan
    else:
        metrics['d_prime'] = np.nan

    # Reaction times
    metrics['mean_reaction_time'] = trials_df['reaction_time_s'].mean()
    metrics['median_reaction_time'] = trials_df['reaction_time_s'].median()

    return metrics
