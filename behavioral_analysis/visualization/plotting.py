"""
Plotting module for DELTA behavioral data.

This module provides functions for visualizing behavioral data, including:
- Position plots with events
- Lick raster plots
- Corridor visualizations
- Trial outcome plots
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple, Union, Any


def plot_position_with_events(position_df: pd.DataFrame,
                             lick_df: Optional[pd.DataFrame] = None,
                             cue_df: Optional[pd.DataFrame] = None,
                             reward_df: Optional[pd.DataFrame] = None,
                             cue_result_df: Optional[pd.DataFrame] = None,
                             corridor_info: Optional[pd.DataFrame] = None,
                             time_range: Optional[Tuple[float, float]] = None,
                             position_column: str = 'global_position_cm',
                             corridor_column: Optional[str] = 'corridor_id',
                             corridor_id: Optional[int] = None,  # Added to filter for a specific corridor
                             output_path: Optional[str] = None,
                             figsize: Tuple[int, int] = (12, 6),
                             title: Optional[str] = "Position with Events",
                             time_in_seconds: bool = True,  # Added to control time units
                             verbose: bool = True) -> plt.Figure:
    """
    Plot position data with overlaid events (licks, cues, rewards).

    Args:
        position_df: DataFrame containing position data
        lick_df: Optional DataFrame containing lick events
        cue_df: Optional DataFrame containing cue events
        reward_df: Optional DataFrame containing reward events
        corridor_info: Optional DataFrame with corridor information
        time_range: Optional tuple of (start_time, end_time) to plot
        position_column: Column name for position values ('position', 'position_cm', etc.)
        corridor_column: Column name for corridor IDs (if available)
        output_path: Optional path to save the figure
        figsize: Figure size as (width, height) tuple
        title: Plot title
        verbose: Whether to print progress information

    Returns:
        Matplotlib Figure object
    """
    if verbose:
        print("Creating position plot with events...")
        start_time = time.time()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Apply time range or corridor filter if specified
    if corridor_id is not None and corridor_info is not None and corridor_column in position_df.columns:
        # Get corridor start and end time
        corridor_row = corridor_info[corridor_info['corridor_id'] == corridor_id]
        if not corridor_row.empty:
            start_t = corridor_row.iloc[0]['start_time']
            end_t = corridor_row.iloc[0]['end_time']

            # Filter by corridor time range
            position_filtered = position_df[(position_df['time'] >= start_t) & (position_df['time'] <= end_t)]
            if lick_df is not None:
                lick_filtered = lick_df[(lick_df['time'] >= start_t) & (lick_df['time'] <= end_t)]
            else:
                lick_filtered = None
            if cue_df is not None:
                cue_filtered = cue_df[(cue_df['time'] >= start_t) & (cue_df['time'] <= end_t)]
            else:
                cue_filtered = None
            if reward_df is not None:
                reward_filtered = reward_df[(reward_df['time'] >= start_t) & (reward_df['time'] <= end_t)]
            else:
                reward_filtered = None
            if cue_result_df is not None:
                cue_result_filtered = cue_result_df[(cue_result_df['time'] >= start_t) & (cue_result_df['time'] <= end_t)]
            else:
                cue_result_filtered = None

            # Update title if using corridor filter
            if title == "Position with Events":
                title = f"Corridor {corridor_id} - Position with Events"
        else:
            # Fall back to no filtering if corridor not found
            position_filtered = position_df
            lick_filtered = lick_df
            cue_filtered = cue_df
            reward_filtered = reward_df
            cue_result_filtered = cue_result_df
    elif time_range is not None:
        start_t, end_t = time_range
        position_filtered = position_df[(position_df['time'] >= start_t) & (position_df['time'] <= end_t)]
        if lick_df is not None:
            lick_filtered = lick_df[(lick_df['time'] >= start_t) & (lick_df['time'] <= end_t)]
        else:
            lick_filtered = None
        if cue_df is not None:
            cue_filtered = cue_df[(cue_df['time'] >= start_t) & (cue_df['time'] <= end_t)]
        else:
            cue_filtered = None
        if reward_df is not None:
            reward_filtered = reward_df[(reward_df['time'] >= start_t) & (reward_df['time'] <= end_t)]
        else:
            reward_filtered = None
        if cue_result_df is not None:
            cue_result_filtered = cue_result_df[(cue_result_df['time'] >= start_t) & (cue_result_df['time'] <= end_t)]
        else:
            cue_result_filtered = None
    else:
        position_filtered = position_df
        lick_filtered = lick_df
        cue_filtered = cue_df
        reward_filtered = reward_df
        cue_result_filtered = cue_result_df

    # Plot position data
    if position_column in position_filtered.columns:
        # Note: x is time, y is position (flipped from previous version)
        ax.plot(position_filtered['time'], position_filtered[position_column],
                color='gray', alpha=0.8, linewidth=1, label='Position')
    else:
        if verbose:
            print(f"Warning: Position column '{position_column}' not found. "
                  f"Available columns: {', '.join(position_filtered.columns)}")

    # Add corridor markers if corridor information is available and not already filtered to a single corridor
    if corridor_info is not None and corridor_column in position_filtered.columns and corridor_id is None:
        # Get unique corridor IDs
        corridor_ids = position_filtered[corridor_column].dropna().unique()

        # Plot corridor boundaries
        for c_id in corridor_ids:
            corridor_data = position_filtered[position_filtered[corridor_column] == c_id]
            if len(corridor_data) > 0:
                start_time = corridor_data['time'].min()
                ax.axvline(x=start_time, color='black', linestyle='--', alpha=0.5)
                ax.text(start_time, ax.get_ylim()[1] * 0.9, f"Corridor {int(c_id)}",
                        rotation=90, verticalalignment='top')

    # Add lick markers
    if lick_filtered is not None and not lick_filtered.empty:
        lick_times = lick_filtered['time'].values
        y_min, y_max = ax.get_ylim()
        y_pos = y_min + (y_max - y_min) * 0.05  # Position lick markers at 5% of y-axis

        for lick_time in lick_times:
            ax.axvline(x=lick_time, color='blue', alpha=0.5, linewidth=1)

        # Add a single entry for the legend
        ax.axvline(x=lick_times[0], color='blue', alpha=0.5, linewidth=1, label='Lick')

    # Add cue markers
    if cue_filtered is not None and not cue_filtered.empty:
        # Check if cue type information is available
        has_rewarding_info = 'isRewarding' in cue_filtered.columns

        for _, cue in cue_filtered.iterrows():
            # Determine color based on rewarding status if available
            if has_rewarding_info:
                color = 'green' if cue['isRewarding'] else 'red'
                label = 'Rewarding Cue' if cue['isRewarding'] else 'Non-rewarding Cue'
            else:
                color = 'purple'
                label = 'Cue'

            # Only add label for first cue of each type (to avoid duplicate legend entries)
            if has_rewarding_info:
                if cue['isRewarding']:
                    if 'rewarding_added' not in locals():
                        rewarding_added = True
                        add_label = True
                    else:
                        add_label = False
                else:
                    if 'non_rewarding_added' not in locals():
                        non_rewarding_added = True
                        add_label = True
                    else:
                        add_label = False
            else:
                if 'cue_added' not in locals():
                    cue_added = True
                    add_label = True
                else:
                    add_label = False

            # Plot the cue marker
            ax.axvline(x=cue['time'], color=color, alpha=0.7, linewidth=1.5,
                      label=label if add_label else None)

    # Add reward markers
    if reward_filtered is not None and not reward_filtered.empty:
        reward_times = reward_filtered['time'].values

        for reward_time in reward_times:
            ax.axvline(x=reward_time, color='green', alpha=0.7, linewidth=2)

        # Add a single entry for the legend
        ax.axvline(x=reward_times[0], color='green', alpha=0.7, linewidth=2, label='Reward')

    # Add cue result markers with trial outcome information
    if cue_result_df is not None and not cue_result_df.empty:
        # Apply time range filter if specified
        if time_range is not None:
            start_t, end_t = time_range
            cue_result_filtered = cue_result_df[(cue_result_df['time'] >= start_t) &
                                              (cue_result_df['time'] <= end_t)]
        else:
            cue_result_filtered = cue_result_df

        # Process each cue result to determine trial outcome
        for _, cue_result in cue_result_filtered.iterrows():
            # Get information needed to determine outcome
            is_rewarding = cue_result.get('isRewarding', False)
            has_licks = (cue_result.get('numLicksInPre', 0) + cue_result.get('numLicksInReward', 0)) > 0
            has_reward = cue_result.get('hasGivenReward', False)

            # Determine outcome and marker properties
            if is_rewarding and has_licks:  # Hit
                outcome = 'Hit'
                marker = 'o'
                color = 'green'
                marker_size = 100
            elif is_rewarding and not has_licks:  # Miss
                outcome = 'Miss'
                marker = 'x'
                color = 'orange'
                marker_size = 100
            elif not is_rewarding and has_licks:  # False Alarm
                outcome = 'FA'
                marker = 'v'
                color = 'red'
                marker_size = 100
            else:  # Correct Rejection
                outcome = 'CR'
                marker = 's'
                color = 'blue'
                marker_size = 100

            # Get position value if available
            if position_column in cue_result_filtered.columns:
                y_pos = cue_result[position_column]
            elif 'position' in cue_result_filtered.columns:
                y_pos = cue_result['position']
            else:
                # If no position, use a fixed position relative to the plot limits
                y_min, y_max = ax.get_ylim()
                y_pos = y_min + (y_max - y_min) * 0.1

            # Plot the marker
            scatter = ax.scatter(cue_result['time'], y_pos, marker=marker, s=marker_size,
                               color=color, alpha=0.7, label=outcome if outcome not in locals() else '')

            # Add text label with outcome and lick count
            lick_count = cue_result.get('numLicksInPre', 0) + cue_result.get('numLicksInReward', 0)
            if lick_count > 0:
                ax.text(cue_result['time'], y_pos, f"{outcome}\n{lick_count}",
                        ha='center', va='bottom', fontsize=8)

    # Set labels and title
    time_unit = 's' if time_in_seconds else 'ms'

    # If displaying a specific corridor, adjust time to be relative to corridor start
    if corridor_id is not None and corridor_info is not None:
        corridor_row = corridor_info[corridor_info['corridor_id'] == corridor_id]
        if not corridor_row.empty:
            # Get corridor start time for relative timing
            corridor_start_time = corridor_row.iloc[0]['start_time']

            # Adjust the time values for display
            time_factor = 1000.0 if time_in_seconds else 1.0

            # Set proper x-axis limits and ticks based on corridor duration
            corridor_duration = (corridor_row.iloc[0]['end_time'] - corridor_start_time) / time_factor
            ax.set_xlim(0, corridor_duration)

            # Create new evenly spaced ticks
            num_ticks = 6  # Adjust as needed
            new_ticks = np.linspace(0, corridor_duration, num_ticks)
            ax.set_xticks(new_ticks)
            ax.set_xticklabels([f"{t:.1f}" for t in new_ticks])

            # Update the x-data to be relative to corridor start
            for line in ax.get_lines():
                xdata = line.get_xdata()
                if len(xdata) > 0:  # Only update if there's data
                    adjusted_xdata = (xdata - corridor_start_time) / time_factor
                    line.set_xdata(adjusted_xdata)

            # Update scatter plots too
            for collection in ax.collections:
                offsets = collection.get_offsets()
                if len(offsets) > 0:  # Only update if there's data
                    new_offsets = np.copy(offsets)
                    new_offsets[:, 0] = (offsets[:, 0] - corridor_start_time) / time_factor
                    collection.set_offsets(new_offsets)

            ax.set_xlabel(f'Time since corridor start ({time_unit})')
    else:
        ax.set_xlabel(f'Time ({time_unit})')

    ax.set_ylabel(f'Position ({position_column})')
    if title:
        ax.set_title(title)

    # Add legend
    ax.legend(loc='upper right')

    # Apply grid
    ax.grid(True, alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=100)
        if verbose:
            print(f"Saved figure to {output_path}")

    if verbose:
        end_time = time.time()
        print(f"Created position plot in {end_time - start_time:.2f} seconds")

    return fig


def plot_lick_raster(lick_df: pd.DataFrame,
                    cue_df: Optional[pd.DataFrame] = None,
                    corridor_info: Optional[pd.DataFrame] = None,
                    align_to: str = 'time',
                    position_column: str = 'position_cm',
                    output_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (12, 6),
                    verbose: bool = True) -> plt.Figure:
    """
    Create a lick raster plot showing lick events aligned by time or position.

    Args:
        lick_df: DataFrame containing lick events
        cue_df: Optional DataFrame containing cue events
        corridor_info: Optional DataFrame with corridor information
        align_to: Alignment method ('time' or 'position')
        position_column: Column name for position values (used when align_to='position')
        output_path: Optional path to save the figure
        figsize: Figure size as (width, height) tuple
        verbose: Whether to print progress information

    Returns:
        Matplotlib Figure object
    """
    if verbose:
        print(f"Creating lick raster plot aligned to {align_to}...")
        start_time = time.time()

    # Check if we have position information for licks when aligning by position
    if align_to == 'position' and position_column not in lick_df.columns:
        if verbose:
            print(f"Warning: Cannot align by position. '{position_column}' not found in lick data.")
            print(f"Available columns: {', '.join(lick_df.columns)}")
        align_to = 'time'
        if verbose:
            print(f"Falling back to time alignment.")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get x-values based on alignment method
    if align_to == 'position':
        x_values = lick_df[position_column].values
        xlabel = f'Position ({position_column})'
    else:  # Default to time alignment
        x_values = lick_df['time'].values
        xlabel = 'Time (ms)'

    # Sort licks by corridor if available
    if 'corridor_id' in lick_df.columns:
        # Group licks by corridor
        licks_by_corridor = []
        corridor_labels = []

        for corridor_id, group in lick_df.groupby('corridor_id'):
            if align_to == 'position':
                licks_by_corridor.append(group[position_column].values)
            else:
                licks_by_corridor.append(group['time'].values)
            corridor_labels.append(f"Corridor {int(corridor_id)}")

        # Create raster plot with one row per corridor
        if licks_by_corridor:
            ax.eventplot(licks_by_corridor, linelengths=0.8, colors='blue', alpha=0.7)
            ax.set_yticks(range(len(licks_by_corridor)))
            ax.set_yticklabels(corridor_labels)
    else:
        # Simple raster with all licks in one row
        ax.eventplot([x_values], linelengths=0.8, colors='blue', alpha=0.7)
        ax.set_yticks([])

    # Add cue markers if available
    if cue_df is not None:
        # Check if cue type information is available
        has_rewarding_info = 'isRewarding' in cue_df.columns

        for _, cue in cue_df.iterrows():
            if align_to == 'position' and position_column in cue_df.columns:
                x_pos = cue[position_column]
            else:
                x_pos = cue['time']

            # Determine color based on rewarding status if available
            if has_rewarding_info:
                color = 'green' if cue['isRewarding'] else 'red'
                label = 'Rewarding Cue' if cue['isRewarding'] else 'Non-rewarding Cue'
            else:
                color = 'purple'
                label = 'Cue'

            # Only add label for first cue of each type (to avoid duplicate legend entries)
            if has_rewarding_info:
                if cue['isRewarding']:
                    if 'rewarding_added' not in locals():
                        rewarding_added = True
                        add_label = True
                    else:
                        add_label = False
                else:
                    if 'non_rewarding_added' not in locals():
                        non_rewarding_added = True
                        add_label = True
                    else:
                        add_label = False
            else:
                if 'cue_added' not in locals():
                    cue_added = True
                    add_label = True
                else:
                    add_label = False

            # Plot the cue marker
            ax.axvline(x=x_pos, color=color, alpha=0.5, linewidth=1.5,
                      label=label if add_label else None)

    # Add corridor boundaries if corridor info is available
    if corridor_info is not None and align_to == 'position':
        for _, corridor in corridor_info.iterrows():
            if 'length_cm' in corridor:
                corridor_start = corridor['corridor_id'] * corridor['length_cm']
                ax.axvline(x=corridor_start, color='black', linestyle='--', alpha=0.5,
                          label='Corridor Boundary' if _ == 0 else None)

    # Set labels and title
    ax.set_xlabel(xlabel)
    ax.set_title('Lick Raster Plot')

    # Add legend
    if cue_df is not None or (corridor_info is not None and align_to == 'position'):
        ax.legend(loc='upper right')

    # Apply grid
    ax.grid(True, alpha=0.3, axis='x')

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=100)
        if verbose:
            print(f"Saved figure to {output_path}")

    if verbose:
        end_time = time.time()
        print(f"Created lick raster plot in {end_time - start_time:.2f} seconds")

    return fig


def plot_corridor_summary(position_df: pd.DataFrame,
                         corridor_info: pd.DataFrame,
                         corridor_id: int = 0,
                         lick_df: Optional[pd.DataFrame] = None,
                         cue_df: Optional[pd.DataFrame] = None,
                         cue_result_df: Optional[pd.DataFrame] = None,
                         position_column: str = 'global_position_cm',
                         corridor_column: str = 'corridor_id',
                         output_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (12, 8),
                         time_in_seconds: bool = True,
                         verbose: bool = True) -> plt.Figure:
    """
    Create a detailed visualization of a specific corridor.

    Args:
        position_df: DataFrame containing position data
        lick_df: Optional DataFrame containing lick events
        cue_df: Optional DataFrame containing cue events
        cue_result_df: Optional DataFrame containing cue result events
        corridor_info: DataFrame with corridor information
        corridor_id: ID of the corridor to visualize
        position_column: Column name for position values
        corridor_column: Column name for corridor IDs
        output_path: Optional path to save the figure
        figsize: Figure size as (width, height) tuple
        verbose: Whether to print progress information

    Returns:
        Matplotlib Figure object
    """
    if verbose:
        print(f"Creating detailed visualization for corridor {corridor_id}...")
        start_time = time.time()

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Set default colors for trial outcomes
    outcome_colors = {
        'Hit': 'green',
        'Miss': 'orange',
        'FA': 'red',  # False Alarm
        'CR': 'blue'   # Correct Rejection
    }

    # Get corridor information
    corridor_row = corridor_info[corridor_info['corridor_id'] == corridor_id]
    if corridor_row.empty:
        if verbose:
            print(f"Warning: No information found for corridor {corridor_id}")
        return fig

    corridor_data = corridor_row.iloc[0]
    corridor_start_time = corridor_data['start_time']
    corridor_end_time = corridor_data['end_time']
    corridor_length = corridor_data.get('length_cm', 200)  # Default to 200 cm

    # Filter data for this corridor
    if corridor_column in position_df.columns:
        corridor_positions = position_df[position_df[corridor_column] == corridor_id]
    else:
        # If no corridor ID column, filter by time
        corridor_positions = position_df[
            (position_df['time'] >= corridor_start_time) &
            (position_df['time'] <= corridor_end_time)
        ]

    # Plot position data
    if position_column in corridor_positions.columns:
        # Convert milliseconds to seconds if requested
        time_factor = 1000.0 if time_in_seconds else 1.0
        time_data = (corridor_positions['time'] - corridor_start_time) / time_factor

        # Plot with time on x-axis and position on y-axis
        ax1.plot(time_data, corridor_positions[position_column],
                color='gray', alpha=0.8, linewidth=1.5, label='Position')
    else:
        if verbose:
            print(f"Warning: Position column '{position_column}' not found. "
                  f"Available columns: {', '.join(corridor_positions.columns)}")

    # Add lick markers
    if lick_df is not None:
        # Filter licks for this corridor
        if corridor_column in lick_df.columns:
            corridor_licks = lick_df[lick_df[corridor_column] == corridor_id]
        else:
            # If no corridor ID column, filter by time
            corridor_licks = lick_df[
                (lick_df['time'] >= corridor_start_time) &
                (lick_df['time'] <= corridor_end_time)
            ]

        # Plot licks
        if not corridor_licks.empty:
            if position_column in corridor_licks.columns:
                ax1.scatter(corridor_licks[position_column],
                           corridor_licks['time'] - corridor_start_time,
                           color='blue', s=30, alpha=0.7, marker='|', label='Lick')

    # Add cue markers
    if cue_df is not None:
        # Filter cues for this corridor
        if corridor_column in cue_df.columns:
            corridor_cues = cue_df[cue_df[corridor_column] == corridor_id]
        else:
            # If no corridor ID column, filter by time
            corridor_cues = cue_df[
                (cue_df['time'] >= corridor_start_time) &
                (cue_df['time'] <= corridor_end_time)
            ]

        # Plot cues
        if not corridor_cues.empty:
            # Check if cue type information is available
            has_rewarding_info = 'isRewarding' in corridor_cues.columns

            for _, cue in corridor_cues.iterrows():
                if position_column in corridor_cues.columns:
                    x_pos = cue[position_column]
                elif 'position' in corridor_cues.columns:
                    # Try using regular position
                    # May need to scale or transform
                    x_pos = cue['position']
                    if x_pos > 1000:  # Likely in arbitrary units
                        # Convert to cm (assuming 50,000 units = 200 cm)
                        x_pos = x_pos / (50000 / corridor_length)
                else:
                    # Skip if no position info
                    continue

                # Plot y position based on time
                y_pos = cue['time'] - corridor_start_time

                # Determine color based on rewarding status if available
                if has_rewarding_info:
                    color = 'green' if cue['isRewarding'] else 'red'
                    label = 'Rewarding Cue' if cue['isRewarding'] else 'Non-rewarding Cue'
                else:
                    color = 'purple'
                    label = 'Cue'

                # Only add label for first cue of each type (to avoid duplicate legend entries)
                if has_rewarding_info:
                    if cue['isRewarding']:
                        if 'rewarding_added' not in locals():
                            rewarding_added = True
                            add_label = True
                        else:
                            add_label = False
                    else:
                        if 'non_rewarding_added' not in locals():
                            non_rewarding_added = True
                            add_label = True
                        else:
                            add_label = False
                else:
                    if 'cue_added' not in locals():
                        cue_added = True
                        add_label = True
                    else:
                        add_label = False

                # Plot the cue marker
                ax1.scatter(x_pos, y_pos, color=color, s=100, alpha=0.7, marker='^',
                          label=label if add_label else None)

                # Add a horizontal line at cue position
                ax1.axhline(y=y_pos, color=color, alpha=0.2, linewidth=1)

    # Create lick histogram in second subplot
    if lick_df is not None:
        # Filter licks for this corridor
        if corridor_column in lick_df.columns:
            corridor_licks = lick_df[lick_df[corridor_column] == corridor_id]
        else:
            # If no corridor ID column, filter by time
            corridor_licks = lick_df[
                (lick_df['time'] >= corridor_start_time) &
                (lick_df['time'] <= corridor_end_time)
            ]

        # Create histogram
        if not corridor_licks.empty and position_column in corridor_licks.columns:
            # Create bins
            bins = np.linspace(0, corridor_length, int(corridor_length/5) + 1)  # 5 cm bins
            ax2.hist(corridor_licks[position_column], bins=bins, alpha=0.7, color='blue')
            ax2.set_ylabel('Lick Count')

            # Set reasonable y-limit based on histogram data
            counts, _ = np.histogram(corridor_licks[position_column], bins=bins)
            if counts.max() > 0:
                ax2.set_ylim(0, counts.max() * 1.2)  # Add 20% padding above max count
            else:
                ax2.set_ylim(0, 1)  # Default y-limit if no licks

            # Add cue result positions if available
            if cue_result_df is not None:
                # Filter cue results for this corridor
                if corridor_column in cue_result_df.columns:
                    corridor_cue_results = cue_result_df[cue_result_df[corridor_column] == corridor_id]
                else:
                    # If no corridor ID column, filter by time
                    corridor_cue_results = cue_result_df[
                        (cue_result_df['time'] >= corridor_start_time) &
                        (cue_result_df['time'] <= corridor_end_time)
                    ]

                # Plot cue result positions
                if not corridor_cue_results.empty:
                    for _, cue_result in corridor_cue_results.iterrows():
                        # Get information needed to determine outcome
                        is_rewarding = cue_result.get('isRewarding', False)
                        has_licks = (cue_result.get('numLicksInPre', 0) + cue_result.get('numLicksInReward', 0)) > 0

                        # Determine outcome and color
                        if is_rewarding and has_licks:  # Hit
                            outcome = 'Hit'
                            color = 'green'
                        elif is_rewarding and not has_licks:  # Miss
                            outcome = 'Miss'
                            color = 'orange'
                        elif not is_rewarding and has_licks:  # False Alarm
                            outcome = 'FA'
                            color = 'red'
                        else:  # Correct Rejection
                            outcome = 'CR'
                            color = 'blue'

                        # Get position value if available
                        if position_column in corridor_cue_results.columns:
                            x_pos = cue_result[position_column]
                        elif 'position' in corridor_cue_results.columns:
                            # Try using regular position
                            x_pos = cue_result['position']
                            # May need to scale based on corridor length
                            if x_pos > 1000:  # Likely in arbitrary units
                                x_pos = x_pos / (50000 / corridor_length)
                        else:
                            continue

                        # Plot vertical line at cue position
                        ax2.axvline(x=x_pos, color=color, alpha=0.7, linewidth=1.5, label=outcome if outcome not in locals() else '')

    # Add cue result markers with trial outcome information
    if cue_result_df is not None:
        # Filter cue results for this corridor
        if corridor_column in cue_result_df.columns:
            corridor_cue_results = cue_result_df[cue_result_df[corridor_column] == corridor_id]
        else:
            # If no corridor ID column, filter by time
            corridor_cue_results = cue_result_df[
                (cue_result_df['time'] >= corridor_start_time) &
                (cue_result_df['time'] <= corridor_end_time)
            ]

        # Plot cue results
        if not corridor_cue_results.empty:
            # Process each cue result to determine trial outcome
            for _, cue_result in corridor_cue_results.iterrows():
                # Get information needed to determine outcome
                is_rewarding = cue_result.get('isRewarding', False)
                has_licks = (cue_result.get('numLicksInPre', 0) + cue_result.get('numLicksInReward', 0)) > 0

                # Determine outcome and marker properties
                if is_rewarding and has_licks:  # Hit
                    outcome = 'Hit'
                    marker = '^'
                elif is_rewarding and not has_licks:  # Miss
                    outcome = 'Miss'
                    marker = 'x'
                elif not is_rewarding and has_licks:  # False Alarm
                    outcome = 'FA'
                    marker = 'v'
                else:  # Correct Rejection
                    outcome = 'CR'
                    marker = 's'

                # Get position value if available
                if position_column in corridor_cue_results.columns:
                    # Use the same position column as used for the main plot
                    x_pos = cue_result[position_column]
                elif 'cumulative_position_cm' in corridor_cue_results.columns:
                    # Prefer global position
                    x_pos = cue_result['cumulative_position_cm']
                elif 'position_cm' in corridor_cue_results.columns:
                    # Fall back to local position converted to cm
                    x_pos = cue_result['position_cm']
                elif 'position' in corridor_cue_results.columns:
                    # Last resort: use raw position and convert
                    x_pos = cue_result['position']
                    # May need to scale based on corridor length
                    if x_pos > 1000:  # Likely in arbitrary units
                        x_pos = x_pos / (50000 / corridor_length)
                        # Add the corridor start position to make it global
                        if 'corridor_id' in cue_result:
                            x_pos += cue_result['corridor_id'] * corridor_length
                else:
                    # Skip if no position info
                    continue

                # Plot the marker
                time_factor = 1000.0 if time_in_seconds else 1.0
                x_time = (cue_result['time'] - corridor_start_time) / time_factor
                # Convert local position to global position using corridor_id
                if 'cumulative_position_cm' in corridor_cue_results.columns:
                    # Use existing global position if available
                    global_x_pos = x_pos
                else:
                    # Calculate global position based on corridor_id and position
                    # First convert to cm if needed
                    if x_pos > 1000:  # Raw position in arbitrary units
                        pos_cm = x_pos / (50000 / corridor_length)
                    else:
                        pos_cm = x_pos

                    # Add offset based on corridor_id
                    global_x_pos = pos_cm + (corridor_id * corridor_length)

                ax1.scatter(x_time, global_x_pos, marker=marker, s=100,
                          color=outcome_colors[outcome], alpha=0.7,
                          label=outcome if outcome not in locals() else '')

                # Add text label with outcome using global position
                ax1.text(x_time, global_x_pos, outcome,
                        ha='center', va='bottom', fontsize=8)

                # Add a vertical line at cue time
                ax1.axvline(x=x_time, color=outcome_colors[outcome], alpha=0.2, linewidth=1)

    # Set labels and title
    time_unit = 's' if time_in_seconds else 'ms'
    ax1.set_xlabel(f'Time since corridor start ({time_unit})')
    ax1.set_ylabel(f'Position ({position_column})')
    ax1.set_title(f'Corridor {corridor_id} - Trajectory and Events')
    ax2.set_xlabel(f'Position ({position_column})')

    # Add legend
    ax1.legend(loc='upper right')

    # Apply grid
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)

    # Set axis limits to show full corridor
    # Check if we need to adjust y-limits based on actual position data in the corridor
    if position_column in corridor_positions.columns:
        y_min = corridor_positions[position_column].min()
        y_max = corridor_positions[position_column].max()
        # Add some padding
        y_range = y_max - y_min
        y_padding = y_range * 0.05  # 5% padding
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)
    else:
        # Fallback to corridor length if no position data available
        ax1.set_ylim(0, corridor_length)

    # Set x limits with padding
    x_min = 0
    time_factor = 1000.0 if time_in_seconds else 1.0
    x_max = (corridor_end_time - corridor_start_time) / time_factor
    x_padding = (x_max - x_min) * 0.05  # 5% padding
    ax1.set_xlim(x_min - x_padding, x_max + x_padding)

    # Adjust layout
    plt.tight_layout()

    # Save figure if requested
    if output_path:
        plt.savefig(output_path, dpi=100)
        if verbose:
            print(f"Saved figure to {output_path}")

    if verbose:
        end_time = time.time()
        print(f"Created corridor visualization in {end_time - start_time:.2f} seconds")

    return fig


if __name__ == "__main__":
    # Example usage if this module is run directly
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Create visualizations for behavioral data')
    parser.add_argument('input_file', help='Path to HDF5 file containing behavioral data')
    parser.add_argument('--output-dir', '-d', default='.',
                        help='Output directory for plots')
    parser.add_argument('--plot-type', '-p', choices=['position', 'raster', 'corridor'],
                        default='position', help='Type of plot to create')
    parser.add_argument('--corridor-id', '-c', type=int, default=0,
                        help='Corridor ID for corridor plots')
    parser.add_argument('--align-to', choices=['time', 'position'],
                        default='time', help='Alignment method for raster plots')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    try:
        # Load data
        with pd.HDFStore(args.input_file, 'r') as store:
            # Load position data
            position_df = None
            if '/events/Position' in store:
                position_df = store['/events/Position']
            elif '/Position' in store:
                position_df = store['/Position']
            else:
                print("Warning: No position data found in input file")

            # Load lick data
            lick_df = None
            if '/events/Lick' in store:
                lick_df = store['/events/Lick']
            elif '/Lick' in store:
                lick_df = store['/Lick']

            # Load cue data
            cue_df = None
            if '/events/Cue_State' in store:
                cue_df = store['/events/Cue_State']
            elif '/Cue_State' in store:
                cue_df = store['/Cue_State']

            # Load reward data
            reward_df = None
            if '/events/Reward' in store:
                reward_df = store['/events/Reward']
            elif '/Reward' in store:
                reward_df = store['/Reward']

            # Load corridor info
            corridor_info = None
            if '/Corridor_Info' in store:
                corridor_info = store['/Corridor_Info']
            elif '/corridor_info' in store:
                corridor_info = store['/corridor_info']

        # Create output path
        import os
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]

        # Create the plot based on type
        if args.plot_type == 'position':
            output_path = os.path.join(args.output_dir, f"{base_name}_position_plot.png")
            plot_position_with_events(
                position_df=position_df,
                lick_df=lick_df,
                cue_df=cue_df,
                reward_df=reward_df,
                corridor_info=corridor_info,
                output_path=output_path,
                verbose=not args.quiet
            )
            print(f"Created position plot: {output_path}")

        elif args.plot_type == 'raster':
            output_path = os.path.join(args.output_dir, f"{base_name}_lick_raster.png")
            plot_lick_raster(
                lick_df=lick_df,
                cue_df=cue_df,
                corridor_info=corridor_info,
                align_to=args.align_to,
                output_path=output_path,
                verbose=not args.quiet
            )
            print(f"Created lick raster plot: {output_path}")

        elif args.plot_type == 'corridor':
            output_path = os.path.join(args.output_dir, f"{base_name}_corridor_{args.corridor_id}.png")
            plot_corridor_summary(
                position_df=position_df,
                lick_df=lick_df,
                cue_df=cue_df,
                corridor_info=corridor_info,
                corridor_id=args.corridor_id,
                output_path=output_path,
                verbose=not args.quiet
            )
            print(f"Created corridor visualization: {output_path}")

        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)