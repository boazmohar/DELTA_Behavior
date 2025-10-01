"""
Visualization tools for trial-based behavioral data analysis.

This module provides comprehensive visualization functions for:
- Trial outcomes over time
- Performance metrics
- Position tracking with behavioral markers
- Reaction time distributions
- Learning curves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Union
import warnings

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.facecolor'] = 'white'


def load_data(hdf5_path: str = None, csv_path: str = None) -> Tuple[Dict, pd.DataFrame]:
    """
    Load data from HDF5 file and/or CSV file.

    Args:
        hdf5_path: Path to HDF5 file
        csv_path: Path to trials CSV file

    Returns:
        Tuple of (hdf5_data_dict, trials_df)
    """
    hdf5_data = {}
    trials_df = None

    if hdf5_path:
        with pd.HDFStore(hdf5_path, 'r') as store:
            for key in store.keys():
                hdf5_data[key] = store[key]

    if csv_path:
        trials_df = pd.read_csv(csv_path)
    elif hdf5_path and '/events/Trials' in hdf5_data:
        trials_df = hdf5_data['/events/Trials']

    return hdf5_data, trials_df


def plot_trial_outcomes_timeline(trials_df: pd.DataFrame,
                                  figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
    """
    Plot trial outcomes over time with color coding.

    Args:
        trials_df: DataFrame with trial data
        figsize: Figure size

    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Define colors for outcomes
    colors = {
        'Hit': '#2ecc71',      # Green
        'Miss': '#f39c12',     # Orange
        'FA': '#e74c3c',       # Red
        'CR': '#3498db'        # Blue
    }

    # Plot 1: Trial outcomes over time
    # Use mouse position if available, otherwise use cue position
    position_col = 'mouse_global_position_cm' if 'mouse_global_position_cm' in trials_df.columns else 'global_position_cm'

    for outcome, color in colors.items():
        mask = trials_df['outcome'] == outcome
        if mask.any():
            ax1.scatter(trials_df[mask]['session_time_min'],
                       trials_df[mask][position_col],
                       c=color, label=outcome, alpha=0.6, s=30, edgecolors='white')

    ax1.set_ylabel('Global Position (cm)', fontsize=11)
    ax1.set_title('Trial Outcomes Over Session', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Running performance (sliding window)
    window = 20  # trials
    trials_df['correct_numeric'] = trials_df['correct'].astype(int)
    rolling_accuracy = trials_df['correct_numeric'].rolling(window, min_periods=1).mean() * 100

    ax2.plot(trials_df['session_time_min'], rolling_accuracy,
             color='#2c3e50', linewidth=2)
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(trials_df['session_time_min'], 50, rolling_accuracy,
                      where=(rolling_accuracy >= 50), alpha=0.3, color='green')
    ax2.fill_between(trials_df['session_time_min'], 50, rolling_accuracy,
                      where=(rolling_accuracy < 50), alpha=0.3, color='red')

    ax2.set_xlabel('Session Time (minutes)', fontsize=11)
    ax2.set_ylabel(f'Accuracy (%, {window}-trial window)', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_performance_summary(trials_df: pd.DataFrame,
                             figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Create a comprehensive performance summary with multiple panels.

    Args:
        trials_df: DataFrame with trial data
        figsize: Figure size

    Returns:
        Figure object
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Outcome distribution (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    outcome_counts = trials_df['outcome'].value_counts()
    colors = {'Hit': '#2ecc71', 'Miss': '#f39c12', 'FA': '#e74c3c', 'CR': '#3498db'}
    pie_colors = [colors[x] for x in outcome_counts.index]
    ax1.pie(outcome_counts.values, labels=outcome_counts.index, autopct='%1.1f%%',
            colors=pie_colors, startangle=90)
    ax1.set_title('Outcome Distribution', fontsize=11, fontweight='bold')

    # 2. Performance metrics (bar plot)
    ax2 = fig.add_subplot(gs[0, 1])
    hit_rate = (trials_df['outcome'] == 'Hit').sum() / trials_df['is_rewarding'].sum() * 100
    fa_rate = (trials_df['outcome'] == 'FA').sum() / (~trials_df['is_rewarding']).sum() * 100
    accuracy = trials_df['correct'].mean() * 100

    metrics = {'Hit Rate': hit_rate, 'FA Rate': fa_rate, 'Accuracy': accuracy}
    bars = ax2.bar(metrics.keys(), metrics.values(),
                   color=['#2ecc71', '#e74c3c', '#3498db'], edgecolor='black')
    ax2.set_ylabel('Percentage (%)', fontsize=10)
    ax2.set_title('Performance Metrics', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 100)

    # Add value labels on bars
    for bar, value in zip(bars, metrics.values()):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

    # 3. Reaction time distribution
    ax3 = fig.add_subplot(gs[0, 2])
    hit_trials = trials_df[trials_df['outcome'] == 'Hit']
    if len(hit_trials) > 0:
        ax3.hist(hit_trials['reaction_time_s'], bins=30, color='#2ecc71',
                edgecolor='black', alpha=0.7)
        ax3.axvline(hit_trials['reaction_time_s'].median(), color='red',
                   linestyle='--', linewidth=2, label=f'Median: {hit_trials["reaction_time_s"].median():.1f}s')
        ax3.set_xlabel('Reaction Time (s)', fontsize=10)
        ax3.set_ylabel('Count', fontsize=10)
        ax3.set_title('Hit Reaction Times', fontsize=11, fontweight='bold')
        ax3.legend()

    # 4. Performance by corridor
    ax4 = fig.add_subplot(gs[1, :])
    corridor_performance = trials_df.groupby('corridor')['correct'].mean() * 100
    ax4.plot(corridor_performance.index, corridor_performance.values,
             marker='o', linewidth=2, markersize=4, color='#2c3e50')
    ax4.fill_between(corridor_performance.index, corridor_performance.values,
                      alpha=0.3, color='#3498db')
    ax4.set_xlabel('Corridor', fontsize=10)
    ax4.set_ylabel('Accuracy (%)', fontsize=10)
    ax4.set_title('Performance by Corridor', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)

    # 5. Lick histogram for different outcomes
    ax5 = fig.add_subplot(gs[2, 0:2])
    for outcome in ['Hit', 'FA', 'Miss', 'CR']:
        outcome_data = trials_df[trials_df['outcome'] == outcome]['num_licks_reward']
        if len(outcome_data) > 0:
            ax5.hist(outcome_data, bins=20, alpha=0.5, label=outcome,
                    edgecolor='black', color=colors[outcome])
    ax5.set_xlabel('Number of Licks in Reward Window', fontsize=10)
    ax5.set_ylabel('Count', fontsize=10)
    ax5.set_title('Licking Behavior by Outcome', fontsize=11, fontweight='bold')
    ax5.legend()

    # 6. Reaction time by cue type
    ax6 = fig.add_subplot(gs[2, 2])
    rewarding_rt = trials_df[trials_df['is_rewarding'] & trials_df['was_hit']]['reaction_time_s']
    non_rewarding_rt = trials_df[~trials_df['is_rewarding'] & trials_df['was_hit']]['reaction_time_s']

    data_to_plot = []
    labels = []
    if len(rewarding_rt) > 0:
        data_to_plot.append(rewarding_rt)
        labels.append('Rewarding')
    if len(non_rewarding_rt) > 0:
        data_to_plot.append(non_rewarding_rt)
        labels.append('Non-rewarding')

    if data_to_plot:
        bp = ax6.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    ax6.set_ylabel('Reaction Time (s)', fontsize=10)
    ax6.set_title('RT by Cue Type', fontsize=11, fontweight='bold')

    plt.suptitle('Behavioral Performance Summary', fontsize=14, fontweight='bold', y=1.02)
    return fig


def plot_position_trace_with_events(hdf5_data: Dict, trials_df: pd.DataFrame,
                                    time_range: Optional[Tuple[float, float]] = None,
                                    figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
    """
    Plot position trace with behavioral events overlaid.

    Args:
        hdf5_data: Dictionary of HDF5 data
        trials_df: DataFrame with trial data
        time_range: Optional tuple of (start_time_min, end_time_min)
        figsize: Figure size

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]})

    # Get position data
    position_df = hdf5_data.get('/events/Position', pd.DataFrame())
    if position_df.empty:
        print("No position data found")
        return fig

    # Convert time to minutes
    position_df['time_min'] = position_df['time'] / 60000

    # Apply time range filter if specified
    if time_range:
        position_mask = (position_df['time_min'] >= time_range[0]) & \
                       (position_df['time_min'] <= time_range[1])
        position_df = position_df[position_mask]

        trials_mask = (trials_df['session_time_min'] >= time_range[0]) & \
                      (trials_df['session_time_min'] <= time_range[1])
        trials_df = trials_df[trials_mask]

    # Plot 1: Position trace with cue markers
    ax1 = axes[0]
    ax1.plot(position_df['time_min'], position_df['global_position_cm'],
             color='gray', alpha=0.5, linewidth=0.5)

    # Overlay trial outcomes
    # Use mouse position if available, otherwise use cue position
    position_col = 'mouse_global_position_cm' if 'mouse_global_position_cm' in trials_df.columns else 'global_position_cm'

    colors = {'Hit': '#2ecc71', 'Miss': '#f39c12', 'FA': '#e74c3c', 'CR': '#3498db'}
    for outcome, color in colors.items():
        mask = trials_df['outcome'] == outcome
        if mask.any():
            ax1.scatter(trials_df[mask]['session_time_min'],
                       trials_df[mask][position_col],
                       c=color, label=outcome, alpha=0.8, s=50,
                       edgecolors='black', linewidth=0.5, zorder=5)

    ax1.set_ylabel('Global Position (cm)', fontsize=11)
    ax1.set_title('Position Trace with Trial Outcomes', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Lick events
    ax2 = axes[1]
    lick_df = hdf5_data.get('/events/Lick', pd.DataFrame())
    if not lick_df.empty:
        lick_df['time_min'] = lick_df['time'] / 60000
        if time_range:
            lick_mask = (lick_df['time_min'] >= time_range[0]) & \
                       (lick_df['time_min'] <= time_range[1])
            lick_df = lick_df[lick_mask]

        ax2.eventplot(lick_df['time_min'], colors='blue', linewidths=0.5, alpha=0.5)
    ax2.set_ylabel('Licks', fontsize=11)
    ax2.set_ylim(-0.5, 1.5)
    ax2.set_yticks([])

    # Plot 3: Reward events
    ax3 = axes[2]
    reward_df = hdf5_data.get('/events/Reward', pd.DataFrame())
    if not reward_df.empty:
        reward_df['time_min'] = reward_df['time'] / 60000
        if time_range:
            reward_mask = (reward_df['time_min'] >= time_range[0]) & \
                         (reward_df['time_min'] <= time_range[1])
            reward_df = reward_df[reward_mask]

        ax3.eventplot(reward_df['time_min'], colors='green', linewidths=1, alpha=0.7)
    ax3.set_ylabel('Rewards', fontsize=11)
    ax3.set_xlabel('Time (minutes)', fontsize=11)
    ax3.set_ylim(-0.5, 1.5)
    ax3.set_yticks([])

    plt.tight_layout()
    return fig


def plot_learning_curves(trials_df: pd.DataFrame,
                        bin_size: int = 50,
                        figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot learning curves showing performance changes over the session.

    Args:
        trials_df: DataFrame with trial data
        bin_size: Number of trials per bin
        figsize: Figure size

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Calculate binned metrics
    n_bins = len(trials_df) // bin_size + 1
    trials_df['bin'] = pd.cut(trials_df['trial_id'], bins=n_bins, labels=False)

    # 1. Hit rate and FA rate over bins
    ax1 = axes[0, 0]
    bin_metrics = []
    for bin_id in range(n_bins):
        bin_data = trials_df[trials_df['bin'] == bin_id]
        if len(bin_data) > 0:
            n_rewarding = bin_data['is_rewarding'].sum()
            n_non_rewarding = (~bin_data['is_rewarding']).sum()

            hit_rate = (bin_data['outcome'] == 'Hit').sum() / n_rewarding * 100 if n_rewarding > 0 else 0
            fa_rate = (bin_data['outcome'] == 'FA').sum() / n_non_rewarding * 100 if n_non_rewarding > 0 else 0

            bin_metrics.append({'bin': bin_id, 'hit_rate': hit_rate, 'fa_rate': fa_rate})

    metrics_df = pd.DataFrame(bin_metrics)
    ax1.plot(metrics_df['bin'], metrics_df['hit_rate'], 'o-', color='#2ecc71',
             label='Hit Rate', linewidth=2, markersize=6)
    ax1.plot(metrics_df['bin'], metrics_df['fa_rate'], 's-', color='#e74c3c',
             label='FA Rate', linewidth=2, markersize=6)
    ax1.set_xlabel(f'Trial Bin ({bin_size} trials/bin)', fontsize=10)
    ax1.set_ylabel('Rate (%)', fontsize=10)
    ax1.set_title('Hit Rate vs FA Rate', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # 2. D-prime over bins
    ax2 = axes[0, 1]
    d_primes = []
    for _, row in metrics_df.iterrows():
        hit_rate = np.clip(row['hit_rate'] / 100, 0.01, 0.99)
        fa_rate = np.clip(row['fa_rate'] / 100, 0.01, 0.99)

        from scipy import stats
        d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
        d_primes.append(d_prime)

    ax2.plot(metrics_df['bin'], d_primes, 'o-', color='#9b59b6',
             linewidth=2, markersize=6)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel(f'Trial Bin ({bin_size} trials/bin)', fontsize=10)
    ax2.set_ylabel("D' (sensitivity)", fontsize=10)
    ax2.set_title("Discriminability (d')", fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Reaction time over bins
    ax3 = axes[1, 0]
    rt_by_bin = trials_df.groupby('bin')['reaction_time_s'].agg(['mean', 'sem'])
    ax3.errorbar(rt_by_bin.index, rt_by_bin['mean'], yerr=rt_by_bin['sem'],
                 fmt='o-', color='#3498db', linewidth=2, markersize=6, capsize=3)
    ax3.set_xlabel(f'Trial Bin ({bin_size} trials/bin)', fontsize=10)
    ax3.set_ylabel('Reaction Time (s)', fontsize=10)
    ax3.set_title('Reaction Time Trend', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Licking behavior over bins
    ax4 = axes[1, 1]
    lick_by_bin = trials_df.groupby('bin')['num_licks_reward'].mean()
    ax4.bar(lick_by_bin.index, lick_by_bin.values, color='#16a085',
            edgecolor='black', alpha=0.7)
    ax4.set_xlabel(f'Trial Bin ({bin_size} trials/bin)', fontsize=10)
    ax4.set_ylabel('Mean Licks in Reward Window', fontsize=10)
    ax4.set_title('Licking Behavior', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Learning Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_session_report(hdf5_path: str, csv_path: Optional[str] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Create a comprehensive session report with all visualizations.

    Args:
        hdf5_path: Path to HDF5 file
        csv_path: Optional path to trials CSV
        save_path: Optional path to save the report PDF
    """
    from matplotlib.backends.backend_pdf import PdfPages

    # Load data
    hdf5_data, trials_df = load_data(hdf5_path, csv_path)

    if trials_df is None or trials_df.empty:
        print("No trial data found")
        return

    # Add 'was_hit' column if it doesn't exist
    if 'was_hit' not in trials_df.columns:
        trials_df['was_hit'] = trials_df['outcome'].isin(['Hit', 'FA'])

    # Create figures
    figures = []

    # 1. Trial outcomes timeline
    fig1 = plot_trial_outcomes_timeline(trials_df)
    figures.append(('Trial Outcomes Timeline', fig1))

    # 2. Performance summary
    fig2 = plot_performance_summary(trials_df)
    figures.append(('Performance Summary', fig2))

    # 3. Position trace (first 10 minutes as example)
    fig3 = plot_position_trace_with_events(hdf5_data, trials_df,
                                           time_range=(0, 10))
    figures.append(('Position Trace (First 10 min)', fig3))

    # 4. Learning curves
    try:
        fig4 = plot_learning_curves(trials_df)
        figures.append(('Learning Curves', fig4))
    except Exception as e:
        print(f"Could not create learning curves: {e}")

    # Save to PDF if requested
    if save_path:
        with PdfPages(save_path) as pdf:
            for title, fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        print(f"Report saved to {save_path}")
    else:
        # Show all figures
        plt.show()

    # Print summary statistics
    print("\n" + "="*50)
    print("SESSION SUMMARY")
    print("="*50)
    print(f"Total trials: {len(trials_df)}")
    print(f"Session duration: {trials_df['session_time_min'].max():.1f} minutes")
    print(f"Corridors traversed: {trials_df['corridor'].nunique()}")
    print(f"\nOutcome distribution:")
    for outcome, count in trials_df['outcome'].value_counts().items():
        print(f"  {outcome}: {count} ({count/len(trials_df)*100:.1f}%)")

    hit_rate = (trials_df['outcome'] == 'Hit').sum() / trials_df['is_rewarding'].sum() * 100
    fa_rate = (trials_df['outcome'] == 'FA').sum() / (~trials_df['is_rewarding']).sum() * 100

    print(f"\nPerformance metrics:")
    print(f"  Hit rate: {hit_rate:.1f}%")
    print(f"  FA rate: {fa_rate:.1f}%")
    print(f"  Overall accuracy: {trials_df['correct'].mean()*100:.1f}%")
    print(f"  Mean reaction time: {trials_df['reaction_time_s'].mean():.2f} ± "
          f"{trials_df['reaction_time_s'].std():.2f} s")


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Visualize behavioral trial data')
    parser.add_argument('hdf5_file', help='Path to HDF5 file')
    parser.add_argument('--csv', help='Path to trials CSV file (optional)')
    parser.add_argument('--save', help='Path to save PDF report')

    args = parser.parse_args()

    create_session_report(args.hdf5_file, args.csv, args.save)