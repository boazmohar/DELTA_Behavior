#!/usr/bin/env python3
"""
Create cue-aligned licking rasters for rewarded and unrewarded trials.
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

# Add some useful columns if not present
trials['rewarded'] = trials['outcome'] == 'Hit'
trials['licked'] = trials['outcome'].isin(['Hit', 'FA'])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Time window around cue hit (in ms)
pre_time = 5000  # 5 seconds before
post_time = 10000  # 10 seconds after

# Process rewarded trials
print("\nProcessing rewarded trials...")
rewarded_trials = trials[trials['is_rewarding'] == True].copy()
rewarded_trials = rewarded_trials.sort_values('cue_outcome_ms').reset_index(drop=True)

print(f"Found {len(rewarded_trials)} rewarded trials")

# Plot raster for rewarded trials
trial_num = 0
for idx, trial in rewarded_trials.iterrows():
    cue_time = trial['cue_outcome_ms']
    
    # Find licks within window
    licks_in_window = licks[
        (licks['time'] >= cue_time - pre_time) & 
        (licks['time'] <= cue_time + post_time)
    ].copy()
    
    # Convert to relative time (ms relative to cue)
    licks_in_window['relative_time'] = licks_in_window['time'] - cue_time
    
    # Plot as raster
    for lick_time in licks_in_window['relative_time']:
        ax1.plot([lick_time/1000, lick_time/1000], [trial_num, trial_num + 0.8], 
                'b-', linewidth=1, alpha=0.7)
    
    # Color code by outcome
    if trial['outcome'] == 'Hit':
        ax1.axhspan(trial_num, trial_num + 1, alpha=0.1, color='green', zorder=0)
    else:  # Miss
        ax1.axhspan(trial_num, trial_num + 1, alpha=0.1, color='orange', zorder=0)
    
    trial_num += 1

ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Cue hit')
ax1.set_ylabel('Rewarded Trial #', fontsize=12)
ax1.set_title('Licking Raster - Rewarded Trials', fontsize=14, fontweight='bold')
ax1.set_xlim(-pre_time/1000, post_time/1000)
ax1.set_ylim(-1, len(rewarded_trials))

# Add outcome legend
green_patch = mpatches.Patch(color='green', alpha=0.3, label='Hit')
orange_patch = mpatches.Patch(color='orange', alpha=0.3, label='Miss')
ax1.legend(handles=[green_patch, orange_patch], loc='upper right')

# Process unrewarded trials
print(f"\nProcessing unrewarded trials...")
unrewarded_trials = trials[trials['is_rewarding'] == False].copy()
unrewarded_trials = unrewarded_trials.sort_values('cue_outcome_ms').reset_index(drop=True)

print(f"Found {len(unrewarded_trials)} unrewarded trials")

# Plot raster for unrewarded trials
trial_num = 0
for idx, trial in unrewarded_trials.iterrows():
    cue_time = trial['cue_outcome_ms']
    
    # Find licks within window
    licks_in_window = licks[
        (licks['time'] >= cue_time - pre_time) & 
        (licks['time'] <= cue_time + post_time)
    ].copy()
    
    # Convert to relative time
    licks_in_window['relative_time'] = licks_in_window['time'] - cue_time
    
    # Plot as raster
    for lick_time in licks_in_window['relative_time']:
        ax2.plot([lick_time/1000, lick_time/1000], [trial_num, trial_num + 0.8], 
                'b-', linewidth=1, alpha=0.7)
    
    # Color code by outcome
    if trial['outcome'] == 'FA':
        ax2.axhspan(trial_num, trial_num + 1, alpha=0.1, color='red', zorder=0)
    else:  # CR
        ax2.axhspan(trial_num, trial_num + 1, alpha=0.1, color='lightblue', zorder=0)
    
    trial_num += 1

ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Cue hit')
ax2.set_xlabel('Time from cue hit (seconds)', fontsize=12)
ax2.set_ylabel('Unrewarded Trial #', fontsize=12)
ax2.set_title('Licking Raster - Unrewarded Trials', fontsize=14, fontweight='bold')
ax2.set_xlim(-pre_time/1000, post_time/1000)
ax2.set_ylim(-1, len(unrewarded_trials))

# Add outcome legend
red_patch = mpatches.Patch(color='red', alpha=0.3, label='False Alarm')
blue_patch = mpatches.Patch(color='lightblue', alpha=0.3, label='Correct Rejection')
ax2.legend(handles=[red_patch, blue_patch], loc='upper right')

plt.tight_layout()
plt.savefig('licking_rasters_cue_aligned.png', dpi=150, bbox_inches='tight')
print("\nSaved licking_rasters_cue_aligned.png")

# Print summary statistics
print("\n=== SUMMARY ===")
print(f"Rewarded trials: {len(rewarded_trials)}")
print(f"  Hits: {(rewarded_trials['outcome'] == 'Hit').sum()} ({(rewarded_trials['outcome'] == 'Hit').sum()/len(rewarded_trials)*100:.1f}%)")
print(f"  Misses: {(rewarded_trials['outcome'] == 'Miss').sum()} ({(rewarded_trials['outcome'] == 'Miss').sum()/len(rewarded_trials)*100:.1f}%)")

print(f"\nUnrewarded trials: {len(unrewarded_trials)}")
print(f"  False Alarms: {(unrewarded_trials['outcome'] == 'FA').sum()} ({(unrewarded_trials['outcome'] == 'FA').sum()/len(unrewarded_trials)*100:.1f}%)")
print(f"  Correct Rejections: {(unrewarded_trials['outcome'] == 'CR').sum()} ({(unrewarded_trials['outcome'] == 'CR').sum()/len(unrewarded_trials)*100:.1f}%)")

# Calculate average lick rate
print("\n=== LICKING PATTERNS ===")
for trial_type, trial_df in [("Rewarded", rewarded_trials), ("Unrewarded", unrewarded_trials)]:
    total_licks_pre = 0
    total_licks_post = 0
    
    for idx, trial in trial_df.iterrows():
        cue_time = trial['cue_outcome_ms']
        
        licks_pre = licks[(licks['time'] >= cue_time - 2000) & (licks['time'] < cue_time)]
        licks_post = licks[(licks['time'] >= cue_time) & (licks['time'] <= cue_time + 2000)]
        
        total_licks_pre += len(licks_pre)
        total_licks_post += len(licks_post)
    
    avg_pre = total_licks_pre / len(trial_df) if len(trial_df) > 0 else 0
    avg_post = total_licks_post / len(trial_df) if len(trial_df) > 0 else 0
    
    print(f"\n{trial_type} trials:")
    print(f"  Avg licks 2s before cue: {avg_pre:.2f}")
    print(f"  Avg licks 2s after cue: {avg_post:.2f}")

plt.show()
