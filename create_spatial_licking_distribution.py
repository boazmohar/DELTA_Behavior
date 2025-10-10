#!/usr/bin/env python3
"""
Create spatial licking distribution plots showing WHERE in the corridor the animal licks.
Bins licks by position (5 cm bins) for rewarded and unrewarded cues.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load data
print("Loading data...")
with pd.HDFStore('BM35_FINAL.h5', 'r') as store:
    trials = store['/events/Trials']
    licks = store['/events/Lick']
    position = store['/events/Position']
    cue_state = store['/events/Cue_State']

# Sort position data by time for lick position lookup
position = position.sort_values('time').reset_index(drop=True)

# Create position bins (5 cm bins from 0 to 200 cm)
bin_size = 5  # cm
bins = np.arange(0, 205, bin_size)
bin_centers = bins[:-1] + bin_size/2

# Initialize arrays for lick counts
rewarded_lick_counts = np.zeros(len(bins)-1)
unrewarded_lick_counts = np.zeros(len(bins)-1)

# Also track cue positions
rewarded_cue_positions = []
unrewarded_cue_positions = []

print("\nProcessing licks and mapping to positions...")

# Process each lick
for _, lick in licks.iterrows():
    lick_time = lick['time']
    
    # Find closest position event to this lick
    time_diffs = abs(position['time'] - lick_time)
    if len(time_diffs) > 0:
        closest_idx = time_diffs.idxmin()
        lick_pos = position.loc[closest_idx, 'position_cm']
        lick_corridor = position.loc[closest_idx, 'corridor_id']
        
        # Only use positions within a corridor (0-200 cm range)
        # Use modulo to wrap positions to corridor space
        lick_pos_in_corridor = lick_pos % 200
        
        # Find which bin this position belongs to
        if 0 <= lick_pos_in_corridor <= 200:
            bin_idx = int(lick_pos_in_corridor / bin_size)
            if bin_idx >= len(bins) - 1:
                bin_idx = len(bins) - 2
            
            # Determine if this is during a rewarded or unrewarded trial
            # Find the most recent trial start before this lick
            trials_before = trials[trials['cue_outcome_ms'] <= lick_time]
            if len(trials_before) > 0:
                recent_trial = trials_before.iloc[-1]
                # Check if lick is within reasonable time of trial (within 60 seconds)
                if lick_time - recent_trial['cue_outcome_ms'] < 60000:
                    if recent_trial['is_rewarding']:
                        rewarded_lick_counts[bin_idx] += 1
                    else:
                        unrewarded_lick_counts[bin_idx] += 1

# Get cue positions
for _, cue in cue_state.iterrows():
    cue_pos_cm = cue['position_cm']
    if cue['isRewarding']:
        rewarded_cue_positions.append(cue_pos_cm)
    else:
        unrewarded_cue_positions.append(cue_pos_cm)

print(f"Processed {len(licks)} total licks")
print(f"Rewarded licks: {int(rewarded_lick_counts.sum())}")
print(f"Unrewarded licks: {int(unrewarded_lick_counts.sum())}")

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot rewarded trials
ax1.bar(bin_centers, rewarded_lick_counts, width=bin_size*0.9, 
        color='green', alpha=0.7, edgecolor='darkgreen')
ax1.set_ylabel('Number of Licks', fontsize=12)
ax1.set_title('Spatial Licking Distribution - Rewarded Trials', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add vertical lines for common cue positions
unique_rew_cues = list(set(rewarded_cue_positions))[:20]  # Show first 20 unique positions
for cue_pos in unique_rew_cues:
    ax1.axvline(x=cue_pos, color='red', linestyle='--', alpha=0.3, linewidth=0.5)

# Add legend
ax1.axvline(x=-1000, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Cue positions')
ax1.legend(loc='upper right')

# Plot unrewarded trials  
ax2.bar(bin_centers, unrewarded_lick_counts, width=bin_size*0.9,
        color='blue', alpha=0.7, edgecolor='darkblue')
ax2.set_xlabel('Position in Corridor (cm)', fontsize=12)
ax2.set_ylabel('Number of Licks', fontsize=12)
ax2.set_title('Spatial Licking Distribution - Unrewarded Trials', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add vertical lines for common cue positions
unique_unrew_cues = list(set(unrewarded_cue_positions))[:20]  # Show first 20 unique positions
for cue_pos in unique_unrew_cues:
    ax2.axvline(x=cue_pos, color='red', linestyle='--', alpha=0.3, linewidth=0.5)

ax2.axvline(x=-1000, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Cue positions')
ax2.legend(loc='upper right')

# Set x-axis limits
ax1.set_xlim(0, 200)
ax2.set_xlim(0, 200)

plt.tight_layout()
plt.savefig('spatial_licking_distribution.png', dpi=150, bbox_inches='tight')
print("\nSaved spatial_licking_distribution.png")

# Create normalized version (licks per cue at each position)
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Count cues at each position
rewarded_cue_counts = np.zeros(len(bins)-1)
unrewarded_cue_counts = np.zeros(len(bins)-1)

for cue_pos in rewarded_cue_positions:
    if 0 <= cue_pos <= 200:
        bin_idx = int(cue_pos / bin_size)
        if bin_idx < len(bins) - 1:
            rewarded_cue_counts[bin_idx] += 1

for cue_pos in unrewarded_cue_positions:
    if 0 <= cue_pos <= 200:
        bin_idx = int(cue_pos / bin_size)
        if bin_idx < len(bins) - 1:
            unrewarded_cue_counts[bin_idx] += 1

# Calculate licks per cue (avoid division by zero)
rewarded_licks_per_cue = np.divide(rewarded_lick_counts, rewarded_cue_counts, 
                                   where=rewarded_cue_counts!=0, out=np.zeros_like(rewarded_lick_counts))
unrewarded_licks_per_cue = np.divide(unrewarded_lick_counts, unrewarded_cue_counts,
                                     where=unrewarded_cue_counts!=0, out=np.zeros_like(unrewarded_lick_counts))

# Plot normalized data
ax3.bar(bin_centers, rewarded_licks_per_cue, width=bin_size*0.9,
        color='green', alpha=0.7, edgecolor='darkgreen')
ax3.set_ylabel('Licks per Cue', fontsize=12)
ax3.set_title('Spatial Licking Rate - Rewarded Cues (normalized by number of cues)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_xlim(0, 200)

ax4.bar(bin_centers, unrewarded_licks_per_cue, width=bin_size*0.9,
        color='blue', alpha=0.7, edgecolor='darkblue')
ax4.set_xlabel('Position in Corridor (cm)', fontsize=12)
ax4.set_ylabel('Licks per Cue', fontsize=12)
ax4.set_title('Spatial Licking Rate - Unrewarded Cues (normalized by number of cues)', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xlim(0, 200)

plt.tight_layout()
plt.savefig('spatial_licking_rate_normalized.png', dpi=150, bbox_inches='tight')
print("Saved spatial_licking_rate_normalized.png")

# Print summary statistics
print("\n=== SUMMARY ===")
print("Top 5 positions with most licking (Rewarded):")
top_rew = sorted(enumerate(rewarded_lick_counts), key=lambda x: x[1], reverse=True)[:5]
for bin_idx, count in top_rew:
    print(f"  {bins[bin_idx]:.0f}-{bins[bin_idx+1]:.0f} cm: {int(count)} licks")

print("\nTop 5 positions with most licking (Unrewarded):")
top_unrew = sorted(enumerate(unrewarded_lick_counts), key=lambda x: x[1], reverse=True)[:5]
for bin_idx, count in top_unrew:
    print(f"  {bins[bin_idx]:.0f}-{bins[bin_idx+1]:.0f} cm: {int(count)} licks")

print("\nTop 5 positions with highest lick rate per cue (Rewarded):")
top_rate_rew = sorted(enumerate(rewarded_licks_per_cue), key=lambda x: x[1], reverse=True)[:5]
for bin_idx, rate in top_rate_rew:
    if rate > 0:
        print(f"  {bins[bin_idx]:.0f}-{bins[bin_idx+1]:.0f} cm: {rate:.1f} licks/cue")

print("\nTop 5 positions with highest lick rate per cue (Unrewarded):")
top_rate_unrew = sorted(enumerate(unrewarded_licks_per_cue), key=lambda x: x[1], reverse=True)[:5]
for bin_idx, rate in top_rate_unrew:
    if rate > 0:
        print(f"  {bins[bin_idx]:.0f}-{bins[bin_idx+1]:.0f} cm: {rate:.1f} licks/cue")

plt.show()
