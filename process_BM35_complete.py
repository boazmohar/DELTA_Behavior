#!/usr/bin/env python3
"""
Complete processing script for BM35 behavioral data using only package functions.
This script demonstrates the full pipeline from raw JSON to final visualizations.
"""

import sys
sys.path.insert(0, '/groups/spruston/home/moharb/DELTA_Behavior')

from behavioral_analysis.processing import process_json_to_hdf5
from behavioral_analysis.visualization.trial_visualizer import (
    load_data,
    plot_trial_outcomes_timeline,
    plot_performance_summary,
    plot_position_trace_with_events,
    plot_learning_curves,
    create_session_report
)
import matplotlib.pyplot as plt

# Input and output files
INPUT_JSON = '/groups/spruston/sprustonlab/mesoscope-data/BM35/2025_09_17/1/Log BM35 2025-09-17 session 1.json'
OUTPUT_HDF5 = 'BM35_complete_package.h5'
OUTPUT_CSV = 'BM35_trials_complete.csv'

print("=" * 60)
print("COMPLETE BM35 BEHAVIORAL DATA PROCESSING")
print("Using only package functions")
print("=" * 60)

# Step 1: Process JSON to HDF5 with all features
print("\nStep 1: Processing JSON to HDF5...")
output_file = process_json_to_hdf5(
    input_file=INPUT_JSON,
    output_file=OUTPUT_HDF5,
    corridor_length_cm=200.0,
    include_trials=True,
    verbose=True
)
print(f"\n✓ Saved to: {output_file}")

# Step 2: Load processed data for visualization
print("\n" + "=" * 60)
print("Step 2: Loading data and generating visualizations...")
hdf5_data, trials_df = load_data(OUTPUT_HDF5, None)

if trials_df is not None:
    # Add 'was_hit' column for compatibility
    if 'was_hit' not in trials_df.columns:
        trials_df['was_hit'] = trials_df['outcome'].isin(['Hit', 'FA'])
    
    # Save trials to CSV for inspection
    trials_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved trials to: {OUTPUT_CSV}")
    
    # Display summary statistics
    print(f"\nData Summary:")
    print(f"  - Total trials: {len(trials_df)}")
    print(f"  - Session duration: {trials_df['session_time_min'].max():.1f} minutes")
    print(f"  - Corridors traversed: {trials_df['corridor'].nunique()}")
    
    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  - Overall accuracy: {trials_df['correct'].mean():.1%}")
    print(f"  - Hit rate: {(trials_df['outcome'] == 'Hit').sum()}/{trials_df['is_rewarding'].sum()} = {(trials_df['outcome'] == 'Hit').sum()/trials_df['is_rewarding'].sum():.1%}")
    print(f"  - FA rate: {(trials_df['outcome'] == 'FA').sum()}/{(~trials_df['is_rewarding']).sum()} = {(trials_df['outcome'] == 'FA').sum()/(~trials_df['is_rewarding']).sum():.1%}")
    
    # Check if mouse positions are included
    if 'mouse_global_position_cm' in trials_df.columns:
        print("\n✓ Mouse positions at hit time are included!")
        print(f"  Mouse position range: {trials_df['mouse_global_position_cm'].min():.1f} to {trials_df['mouse_global_position_cm'].max():.1f} cm")
    else:
        print("\n⚠ Mouse positions not included (using cue positions)")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Trial outcomes timeline
    print("  1. Trial outcomes timeline...")
    fig1 = plot_trial_outcomes_timeline(trials_df)
    fig1.savefig('package_trial_outcomes.png', dpi=100, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Performance summary
    print("  2. Performance summary...")
    fig2 = plot_performance_summary(trials_df)
    fig2.savefig('package_performance_summary.png', dpi=100, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Position trace with events (first 5 minutes)
    print("  3. Position trace with events...")
    fig3 = plot_position_trace_with_events(hdf5_data, trials_df.copy(), time_range=(0, 5))
    fig3.savefig('package_position_trace.png', dpi=100, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Learning curves
    print("  4. Learning curves...")
    try:
        fig4 = plot_learning_curves(trials_df, bin_size=30)
        fig4.savefig('package_learning_curves.png', dpi=100, bbox_inches='tight')
        plt.close(fig4)
    except Exception as e:
        print(f"    Warning: Could not create learning curves: {e}")
    
    # 5. Generate comprehensive report
    print("  5. Generating PDF report...")
    create_session_report(OUTPUT_HDF5, OUTPUT_CSV, save_path='BM35_package_report.pdf')
    
    print("\n✓ All visualizations saved:")
    print("  - package_trial_outcomes.png")
    print("  - package_performance_summary.png")
    print("  - package_position_trace.png")
    print("  - package_learning_curves.png (if created)")
    print("  - BM35_package_report.pdf")
    
else:
    print("Error: Could not load trial data")

print("\n" + "=" * 60)
print("✓ PROCESSING COMPLETE!")
print("=" * 60)
