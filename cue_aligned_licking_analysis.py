#!/usr/bin/env python3
"""
Create cue-aligned licking analysis where 0 = cue position
Bins licks in 5cm intervals relative to each cue
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_cue_aligned_licking_analysis():
    # Load the data from HDF5
    print("Loading data from HDF5...")
    with pd.HDFStore('behavioral_analysis/output', 'r') as store:
        trials_df = store['events/Trials']
        position_df = store['events/Position']
        lick_df = store['events/Lick']

    print(f"Loaded {len(trials_df)} trials, {len(lick_df)} licks")

    # Map lick timestamps to global positions using position data
    print("Mapping licks to global positions...")
    lick_global_positions = []
    for lick_time in lick_df['time'].values:
        # Find the closest position timestamp
        time_diff = np.abs(position_df['time'].values - lick_time)
        closest_idx = np.argmin(time_diff)
        lick_global_positions.append(position_df.iloc[closest_idx]['global_position_cm'])

    lick_df['global_position_cm'] = lick_global_positions

    # Define bins relative to cue (cue position = 0)
    bin_size = 5  # cm
    bins = np.arange(-50, 105, bin_size)  # From -50cm before cue to +100cm after
    bin_centers = bins[:-1] + bin_size/2

    # Initialize storage for lick counts
    rewarded_licks = np.zeros(len(bins) - 1)
    unrewarded_licks = np.zeros(len(bins) - 1)
    rewarded_trials = 0
    unrewarded_trials = 0

    print(f"Processing {len(trials_df)} trials...")

    for _, trial in trials_df.iterrows():
        cue_position = trial['global_position_cm']
        is_rewarded = trial['is_rewarding']

        # Get licks for this trial (between cue appearance and result)
        trial_start = trial['cue_onset_ms'] / 1000.0  # Convert to seconds
        if pd.notna(trial['cue_hit_ms']):
            trial_end = trial['cue_hit_ms'] / 1000.0
        else:
            trial_end = trial_start + 5.0  # 5 second timeout

        # Filter licks for this trial
        trial_licks = lick_df[
            (lick_df['time'] >= trial_start) &
            (lick_df['time'] <= trial_end)
        ].copy()

        if len(trial_licks) > 0:
            # Calculate relative position (position relative to cue)
            trial_licks['relative_position'] = trial_licks['global_position_cm'] - cue_position

            # Bin the licks
            lick_hist, _ = np.histogram(trial_licks['relative_position'], bins=bins)

            if is_rewarded:
                rewarded_licks += lick_hist
                rewarded_trials += 1
            else:
                unrewarded_licks += lick_hist
                unrewarded_trials += 1

    # Normalize by number of trials
    if rewarded_trials > 0:
        rewarded_rate = rewarded_licks / rewarded_trials
    else:
        rewarded_rate = rewarded_licks

    if unrewarded_trials > 0:
        unrewarded_rate = unrewarded_licks / unrewarded_trials
    else:
        unrewarded_rate = unrewarded_licks

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Rewarded trials
    bars1 = ax1.bar(bin_centers, rewarded_rate, width=bin_size*0.9,
                    color='green', alpha=0.7, edgecolor='darkgreen')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Cue Position')
    ax1.set_ylabel('Licks per Trial', fontsize=12)
    ax1.set_title(f'Cue-Aligned Licking Rate - Rewarded Trials (n={rewarded_trials})', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Unrewarded trials
    bars2 = ax2.bar(bin_centers, unrewarded_rate, width=bin_size*0.9,
                    color='blue', alpha=0.7, edgecolor='darkblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Cue Position')
    ax2.set_xlabel('Position Relative to Cue (cm)', fontsize=12)
    ax2.set_ylabel('Licks per Trial', fontsize=12)
    ax2.set_title(f'Cue-Aligned Licking Rate - Unrewarded Trials (n={unrewarded_trials})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Add shaded region for typical reward zone (0-20cm after cue)
    for ax in [ax1, ax2]:
        ax.axvspan(0, 20, alpha=0.1, color='yellow', label='Typical Reward Zone')

    # Set x-axis limits to focus on relevant range
    ax2.set_xlim(-50, 100)

    plt.tight_layout()
    plt.savefig('cue_aligned_licking_rate.png', dpi=150, bbox_inches='tight')
    print("Saved cue-aligned licking rate plot to cue_aligned_licking_rate.png")

    # Create a summary statistics file
    with open('cue_aligned_licking_stats.txt', 'w') as f:
        f.write("Cue-Aligned Licking Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total trials analyzed: {len(trials_df)}\n")
        f.write(f"Rewarded trials: {rewarded_trials}\n")
        f.write(f"Unrewarded trials: {unrewarded_trials}\n\n")

        f.write("Top licking zones (relative to cue at 0cm):\n")
        f.write("-" * 40 + "\n")

        # Find peak licking zones for rewarded
        top_rewarded_idx = np.argsort(rewarded_rate)[-5:][::-1]
        f.write("\nRewarded trials:\n")
        for idx in top_rewarded_idx:
            if rewarded_rate[idx] > 0:
                f.write(f"  {bin_centers[idx]:+.0f} to {bin_centers[idx]+bin_size:+.0f} cm: "
                       f"{rewarded_rate[idx]:.2f} licks/trial\n")

        # Find peak licking zones for unrewarded
        top_unrewarded_idx = np.argsort(unrewarded_rate)[-5:][::-1]
        f.write("\nUnrewarded trials:\n")
        for idx in top_unrewarded_idx:
            if unrewarded_rate[idx] > 0:
                f.write(f"  {bin_centers[idx]:+.0f} to {bin_centers[idx]+bin_size:+.0f} cm: "
                       f"{unrewarded_rate[idx]:.2f} licks/trial\n")

        # Calculate pre-cue vs post-cue licking
        pre_cue_mask = bin_centers < 0
        post_cue_mask = bin_centers >= 0

        f.write("\n" + "-" * 40 + "\n")
        f.write("Spatial distribution:\n")
        f.write(f"\nRewarded trials:\n")
        f.write(f"  Pre-cue licking (<0 cm): {rewarded_rate[pre_cue_mask].sum():.2f} licks/trial\n")
        f.write(f"  Post-cue licking (≥0 cm): {rewarded_rate[post_cue_mask].sum():.2f} licks/trial\n")

        f.write(f"\nUnrewarded trials:\n")
        f.write(f"  Pre-cue licking (<0 cm): {unrewarded_rate[pre_cue_mask].sum():.2f} licks/trial\n")
        f.write(f"  Post-cue licking (≥0 cm): {unrewarded_rate[post_cue_mask].sum():.2f} licks/trial\n")

    print("Saved analysis summary to cue_aligned_licking_stats.txt")

    # Also create a raw count version
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Rewarded trials - raw counts
    bars3 = ax3.bar(bin_centers, rewarded_licks, width=bin_size*0.9,
                    color='green', alpha=0.7, edgecolor='darkgreen')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Cue Position')
    ax3.set_ylabel('Total Licks', fontsize=12)
    ax3.set_title(f'Cue-Aligned Total Licks - Rewarded Trials (n={rewarded_trials})', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Unrewarded trials - raw counts
    bars4 = ax4.bar(bin_centers, unrewarded_licks, width=bin_size*0.9,
                    color='blue', alpha=0.7, edgecolor='darkblue')
    ax4.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Cue Position')
    ax4.set_xlabel('Position Relative to Cue (cm)', fontsize=12)
    ax4.set_ylabel('Total Licks', fontsize=12)
    ax4.set_title(f'Cue-Aligned Total Licks - Unrewarded Trials (n={unrewarded_trials})', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # Add shaded region for typical reward zone
    for ax in [ax3, ax4]:
        ax.axvspan(0, 20, alpha=0.1, color='yellow', label='Typical Reward Zone')

    ax4.set_xlim(-50, 100)

    plt.tight_layout()
    plt.savefig('cue_aligned_licking_counts.png', dpi=150, bbox_inches='tight')
    print("Saved cue-aligned licking counts plot to cue_aligned_licking_counts.png")

    return rewarded_rate, unrewarded_rate, bin_centers

if __name__ == "__main__":
    rewarded_rate, unrewarded_rate, bin_centers = create_cue_aligned_licking_analysis()
    print("\nAnalysis complete!")