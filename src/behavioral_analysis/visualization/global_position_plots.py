"""
Plotting functions for global position data in DELTA behavioral experiments.

This module provides specialized plotting functions for visualizing behavioral data
using the global monotonically increasing position.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Union, Any
import os


def plot_global_position_overview(
    position_df: pd.DataFrame,
    corridor_info: Optional[pd.DataFrame] = None,
    cue_df: Optional[pd.DataFrame] = None,
    global_position_column: str = 'global_position_cm',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    verbose: bool = True
) -> plt.Figure:
    """
    Create an overview plot of global position across the entire session.

    Args:
        position_df: DataFrame containing position data with global position
        corridor_info: Optional DataFrame with corridor information
        cue_df: Optional DataFrame containing cue events
        global_position_column: Column name for global position values
        output_path: Optional path to save the figure
        figsize: Figure size as (width, height) tuple
        verbose: Whether to print progress information

    Returns:
        Matplotlib Figure object
    """
    if verbose:
        print("Creating global position overview plot...")
        start_time = time.time()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Check if we have global position data
    if global_position_column not in position_df.columns:
        if verbose:
            print(f"Warning: Global position column '{global_position_column}' not found. "
                  f"Available columns: {', '.join(position_df.columns)}")
        return fig

    # Sample data for better visualization if very large
    sample_step = max(1, len(position_df) // 10000)
    sampled_df = position_df.iloc[::sample_step].sort_values('time')

    # Plot global position
    ax.plot(sampled_df['time'], sampled_df[global_position_column],
            'r-', alpha=0.7, linewidth=1.0, label='Global Position')

    # Add corridor boundaries as vertical lines
    if corridor_info is not None:
        corridors = corridor_info['corridor_id'].sort_values().unique()
        corridor_starts = {}

        for _, corridor in corridor_info.iterrows():
            corridor_starts[corridor['corridor_id']] = corridor['start_time']

        # Only show some corridor boundaries to avoid crowding
        step = max(1, len(corridors) // 20)  # Show at most 20 corridor boundaries
        for corridor_id in corridors[::step]:
            if corridor_id in corridor_starts:
                ax.axvline(x=corridor_starts[corridor_id], color='green',
                          linestyle='--', alpha=0.5)
                ax.text(corridor_starts[corridor_id], ax.get_ylim()[1] * 0.95,
                       f"Corridor {int(corridor_id)}", rotation=90, va='top', fontsize=8)

    # Add cues if available
    if cue_df is not None:
        # Check if we have rewarding info
        has_rewarding_info = 'isRewarding' in cue_df.columns

        if has_rewarding_info:
            # Sample cues for visibility if too many
            sample_step_cue = max(1, len(cue_df) // 200)
            sampled_cues = cue_df.iloc[::sample_step_cue]

            # Plot rewarding cues
            rewarding_cues = sampled_cues[sampled_cues['isRewarding']]
            if global_position_column in rewarding_cues.columns and not rewarding_cues.empty:
                ax.scatter(rewarding_cues['time'], rewarding_cues[global_position_column],
                          marker='^', color='green', s=50, alpha=0.7, label='Rewarding Cue')

            # Plot non-rewarding cues
            non_rewarding_cues = sampled_cues[~sampled_cues['isRewarding']]
            if global_position_column in non_rewarding_cues.columns and not non_rewarding_cues.empty:
                ax.scatter(non_rewarding_cues['time'], non_rewarding_cues[global_position_column],
                          marker='v', color='red', s=50, alpha=0.7, label='Non-rewarding Cue')

    # Set labels and title
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Global Position (cm)')
    ax.set_title('Global Position Across All Corridors')
    ax.legend()

    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved global position overview plot to {output_path}")

    if verbose:
        end_time = time.time()
        print(f"Created global position overview plot in {end_time - start_time:.2f} seconds")

    return fig


def plot_corridor_with_global_position(
    corridor_id: int,
    position_df: pd.DataFrame,
    corridor_info: pd.DataFrame,
    events_dict: Optional[Dict[str, pd.DataFrame]] = None,
    global_position_column: str = 'global_position_cm',
    original_position_column: str = 'position_cm',
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    verbose: bool = True
) -> plt.Figure:
    """
    Create a detailed plot of a single corridor showing both original and global positions.

    Args:
        corridor_id: ID of the corridor to visualize
        position_df: DataFrame containing position data with global position
        corridor_info: DataFrame with corridor information
        events_dict: Optional dictionary with other event DataFrames (Cue_Result, Lick, Reward)
        global_position_column: Column name for global position values
        original_position_column: Column name for original position values
        output_path: Optional path to save the figure
        figsize: Figure size as (width, height) tuple
        verbose: Whether to print progress information

    Returns:
        Matplotlib Figure object
    """
    if verbose:
        print(f"Creating corridor plot with global position for corridor {corridor_id}...")
        start_time = time.time()

    # Check if we have the necessary columns
    required_columns = ['time', 'corridor_id']
    for col in required_columns:
        if col not in position_df.columns:
            if verbose:
                print(f"Warning: Required column '{col}' not found in position_df. "
                      f"Available columns: {', '.join(position_df.columns)}")
            return plt.figure()  # Return empty figure

    # Get corridor boundaries
    corridor_data = corridor_info[corridor_info['corridor_id'] == corridor_id]
    if corridor_data.empty:
        if verbose:
            print(f"Warning: No data found for corridor {corridor_id}")
        return plt.figure()  # Return empty figure

    corridor_row = corridor_data.iloc[0]
    start_time_ms = corridor_row['start_time']
    end_time_ms = corridor_row['end_time']

    # Filter position data for this corridor
    corridor_positions = position_df[position_df['corridor_id'] == corridor_id].copy()
    corridor_positions['time_sec'] = (corridor_positions['time'] - start_time_ms) / 1000.0

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot position traces - original and global
    # First check which columns are available
    available_columns = []
    if original_position_column in corridor_positions.columns:
        available_columns.append(original_position_column)
    if global_position_column in corridor_positions.columns:
        available_columns.append(global_position_column)

    # Plot available position traces
    if original_position_column in available_columns:
        ax.plot(corridor_positions['time_sec'], corridor_positions[original_position_column],
                'k-', alpha=0.4, linewidth=1.5, label='Original Position')

    if global_position_column in available_columns:
        ax.plot(corridor_positions['time_sec'], corridor_positions[global_position_column],
                'r-', alpha=0.8, linewidth=1.5, label='Global Position')

    # Add cues if available
    if events_dict and 'Cue_Result' in events_dict:
        cue_df = events_dict['Cue_Result']
        corridor_cues = cue_df[cue_df['corridor_id'] == corridor_id]

        # Plot cues
        for _, cue in corridor_cues.iterrows():
            cue_time_sec = (cue['time'] - start_time_ms) / 1000.0
            marker = '^' if cue['isRewarding'] else 'v'
            color = 'green' if cue['isRewarding'] else 'red'

            # Plot on global position scale
            if global_position_column in available_columns and global_position_column in cue.index:
                ax.scatter(cue_time_sec, cue[global_position_column],
                          marker=marker, color=color, alpha=0.7, s=100)

            if global_position_column in cue.index:
                ax.text(cue_time_sec, cue[global_position_column] + 5,
                       f"{cue['id']}", ha='center', fontsize=8)

            # Draw vertical line to show 20cm cue length
            if global_position_column in cue.index:
                ax.plot([cue_time_sec, cue_time_sec],
                       [cue[global_position_column], cue[global_position_column] + 20],
                       color=color, linestyle='--', alpha=0.5)

    # Add licks if available
    if events_dict and 'Lick' in events_dict:
        lick_df = events_dict['Lick']
        corridor_licks = lick_df[lick_df['corridor_id'] == corridor_id]

        if not corridor_licks.empty and global_position_column in corridor_licks.columns:
            lick_times_sec = (corridor_licks['time'] - start_time_ms) / 1000.0
            ax.scatter(lick_times_sec, corridor_licks[global_position_column],
                      marker='x', color='blue', alpha=0.7, s=50, label='Licks')

    # Add rewards if available
    if events_dict and 'Reward' in events_dict:
        reward_df = events_dict['Reward']
        corridor_rewards = reward_df[reward_df['corridor_id'] == corridor_id]

        if not corridor_rewards.empty and global_position_column in corridor_rewards.columns:
            reward_times_sec = (corridor_rewards['time'] - start_time_ms) / 1000.0
            ax.scatter(reward_times_sec, corridor_rewards[global_position_column],
                      marker='*', color='gold', alpha=0.8, s=150, label='Rewards')

    # Set labels and title
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (cm)')
    ax.set_title(f'Corridor {corridor_id}: Events with Global Position')
    ax.legend()

    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"Saved corridor global position plot to {output_path}")

    if verbose:
        end_time = time.time()
        print(f"Created corridor global position plot in {end_time - start_time:.2f} seconds")

    return fig


def visualize_all_global_positions(
    hdf5_file: str,
    output_dir: str = 'plots_global_position',
    corridor_ids: List[int] = [0, 10, 20],
    global_position_column: str = 'global_position_cm',
    verbose: bool = True
) -> None:
    """
    Generate a full set of global position visualizations from an HDF5 file.

    Args:
        hdf5_file: Path to the HDF5 file containing behavioral data with global position
        output_dir: Directory where plots will be saved
        corridor_ids: List of corridor IDs to create individual plots for
        global_position_column: Column name for global position values
        verbose: Whether to print progress information
    """
    if verbose:
        print(f"Creating global position visualizations from {hdf5_file}...")
        start_time = time.time()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    with pd.HDFStore(hdf5_file, 'r') as store:
        # Check available datasets
        if verbose:
            print("Available datasets in HDF5 file:")
            for key in store.keys():
                print(f"  {key}")

        # Load position data
        position_df = None
        if '/events/Position' in store:
            position_df = store['/events/Position']
            if verbose:
                print(f"Loaded Position: {len(position_df)} entries")
        else:
            print("Error: Position data not found in HDF5 file")
            return

        # Check for global position column
        if global_position_column not in position_df.columns:
            print(f"Error: Global position column '{global_position_column}' not found in position data")
            print(f"Available columns: {', '.join(position_df.columns)}")
            return

        # Load corridor info
        corridor_info = None
        if '/events/Corridor_Info' in store:
            corridor_info = store['/events/Corridor_Info']
            if verbose:
                print(f"Loaded Corridor_Info: {len(corridor_info)} entries")
        else:
            print("Warning: Corridor_Info not found, some plots may be limited")

        # Load other event types
        events_dict = {}
        for event_type in ['Cue_Result', 'Lick', 'Reward', 'Cue_State']:
            key = f'/events/{event_type}'
            if key in store:
                events_dict[event_type] = store[key]
                if verbose:
                    print(f"Loaded {event_type}: {len(events_dict[event_type])} entries")

    # 1. Create overview of global position
    output_file = os.path.join(output_dir, "global_position_overview.png")
    plot_global_position_overview(
        position_df=position_df,
        corridor_info=corridor_info,
        cue_df=events_dict.get('Cue_State', None),
        global_position_column=global_position_column,
        output_path=output_file,
        verbose=verbose
    )

    # 2. Create individual corridor visualizations
    for corridor_id in corridor_ids:
        output_file = os.path.join(output_dir, f"corridor_{corridor_id}_global_position.png")
        plot_corridor_with_global_position(
            corridor_id=corridor_id,
            position_df=position_df,
            corridor_info=corridor_info,
            events_dict=events_dict,
            global_position_column=global_position_column,
            output_path=output_file,
            verbose=verbose
        )

    if verbose:
        end_time = time.time()
        print(f"All global position visualizations completed in {end_time - start_time:.2f} seconds")
        print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage if this module is run directly
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Create global position visualizations')
    parser.add_argument('input_file', help='Path to HDF5 file with global position data')
    parser.add_argument('--output-dir', '-o', default='plots_global_position',
                        help='Output directory for plots')
    parser.add_argument('--corridors', '-c', type=int, nargs='+', default=[0, 10, 20],
                        help='Corridor IDs to visualize (space-separated list)')
    parser.add_argument('--position-column', default='global_position_cm',
                        help='Column name for global position')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Create visualizations
    visualize_all_global_positions(
        hdf5_file=args.input_file,
        output_dir=args.output_dir,
        corridor_ids=args.corridors,
        global_position_column=args.position_column,
        verbose=not args.quiet
    )

    sys.exit(0)