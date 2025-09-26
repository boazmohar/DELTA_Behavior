#!/usr/bin/env python3
"""
Test the final package with all functionality.
"""

import sys
sys.path.insert(0, '/groups/spruston/home/moharb/DELTA_Behavior')

from behavioral_analysis.processing.json_to_hdf5_processor import process_json_to_hdf5
import pandas as pd

# Input file
json_file = "/groups/spruston/sprustonlab/mesoscope-data/BM35/2025_09_17/1/Log BM35 2025-09-17 session 1.json"
output_file = "BM35_final.h5"

print("=== TESTING FINAL PACKAGE ===\n")
print(f"Processing: {json_file}")

# Process with all features enabled
result = process_json_to_hdf5(
    input_file=json_file,
    output_file=output_file,
    corridor_length_cm=200.0,
    include_combined=False,
    include_trials=True,  # Create trial dataframe
    enable_monotonic_position=True,
    overwrite=True,
    verbose=True
)

print(f"\n✓ Processing complete: {result}")

# Verify output
print("\n=== VERIFICATION ===")
with pd.HDFStore(output_file, 'r') as store:
    # Check available datasets
    print("\nAvailable datasets:")
    for key in store.keys():
        df = store[key]
        print(f"  {key}: {df.shape}")

    # Check trials specifically
    if '/events/Trials' in store:
        trials = store['/events/Trials']

        print(f"\n✓ Trials dataframe: {len(trials)} trials")
        print(f"  Columns: {list(trials.columns)}")

        # Check for required columns
        required = ['trial_id', 'corridor', 'outcome', 'is_rewarding', 'global_position_cm']
        missing = [col for col in required if col not in trials.columns]
        if missing:
            print(f"  ⚠ Missing columns: {missing}")
        else:
            print(f"  ✓ All required columns present")

        # Check outcomes
        print(f"\n  Outcome breakdown:")
        for outcome, count in trials['outcome'].value_counts().items():
            print(f"    {outcome}: {count} ({count/len(trials)*100:.1f}%)")

        # Check trial IDs
        if 'trial_id' in trials.columns:
            print(f"\n  ✓ Trial IDs: {trials['trial_id'].min()} to {trials['trial_id'].max()}")
            if (trials['trial_id'] == range(len(trials))).all():
                print(f"  ✓ Trial IDs are sequential (0 to {len(trials)-1})")

        # Save to CSV
        trials.to_csv('trials_BM35_package_final.csv', index=False)
        print(f"\n  ✓ Saved to: trials_BM35_package_final.csv")