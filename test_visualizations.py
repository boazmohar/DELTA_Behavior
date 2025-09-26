#!/usr/bin/env python3
"""
Test visualization functions with the processed data.
"""

import sys
sys.path.insert(0, '/groups/spruston/home/moharb/DELTA_Behavior')

from behavioral_analysis.visualization.trial_visualizer import (
    load_data,
    plot_trial_outcomes_timeline,
    plot_performance_summary,
    plot_position_trace_with_events,
    plot_learning_curves,
    create_session_report
)
import matplotlib.pyplot as plt

# File paths
hdf5_file = 'BM35_final.h5'
csv_file = 'trials_BM35_package_final.csv'

print("=== TESTING VISUALIZATION FUNCTIONS ===\n")

# Load data
print("Loading data...")
hdf5_data, trials_df = load_data(hdf5_file, csv_file)

if trials_df is not None:
    print(f"Loaded {len(trials_df)} trials from {trials_df['corridor'].nunique()} corridors")
    print(f"Session duration: {trials_df['session_time_min'].max():.1f} minutes")

    # Add 'was_hit' column if missing (for compatibility)
    if 'was_hit' not in trials_df.columns:
        trials_df['was_hit'] = trials_df['outcome'].isin(['Hit', 'FA'])

    # Test individual plots
    print("\nGenerating visualizations...")

    # 1. Trial outcomes timeline
    print("  1. Trial outcomes timeline...")
    fig1 = plot_trial_outcomes_timeline(trials_df)
    fig1.savefig('viz_trial_outcomes.png', dpi=100, bbox_inches='tight')
    plt.close(fig1)

    # 2. Performance summary
    print("  2. Performance summary...")
    fig2 = plot_performance_summary(trials_df)
    fig2.savefig('viz_performance_summary.png', dpi=100, bbox_inches='tight')
    plt.close(fig2)

    # 3. Position trace with events (first 5 minutes)
    print("  3. Position trace with events...")
    fig3 = plot_position_trace_with_events(hdf5_data, trials_df.copy(), time_range=(0, 5))
    fig3.savefig('viz_position_trace.png', dpi=100, bbox_inches='tight')
    plt.close(fig3)

    # 4. Learning curves
    print("  4. Learning curves...")
    try:
        fig4 = plot_learning_curves(trials_df, bin_size=30)
        fig4.savefig('viz_learning_curves.png', dpi=100, bbox_inches='tight')
        plt.close(fig4)
    except Exception as e:
        print(f"    Warning: Could not create learning curves: {e}")

    print("\nVisualizations saved as PNG files:")
    print("  - viz_trial_outcomes.png")
    print("  - viz_performance_summary.png")
    print("  - viz_position_trace.png")
    print("  - viz_learning_curves.png")

    # Generate full PDF report
    print("\nGenerating comprehensive PDF report...")
    create_session_report(hdf5_file, csv_file, save_path='BM35_session_report.pdf')
    print("Report saved as: BM35_session_report.pdf")

else:
    print("Error: Could not load trial data")

print("\n✓ Visualization test complete!")