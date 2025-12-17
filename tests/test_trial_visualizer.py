from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from behavioral_analysis.visualization.trial_visualizer import plot_performance_summary


def test_plot_performance_summary_handles_unknown_outcomes() -> None:
    trials_df = pd.DataFrame(
        {
            "outcome": ["Hit", "Miss", "Unknown", "Abort", "CR"],
            "is_rewarding": [True, True, False, False, False],
            "correct": [True, False, False, False, True],
            "reaction_time_s": [1.0, 2.0, 1.5, 1.2, 0.8],
            "corridor": [1, 1, 2, 2, 2],
            "session_time_min": [0.1, 0.2, 0.3, 0.35, 0.4],
            "num_licks_reward": [3, 0, 1, 0, 0],
            "was_hit": [True, False, False, False, False],
        }
    )

    fig = plot_performance_summary(trials_df)
    try:
        fig.canvas.draw()
    finally:
        plt.close(fig)
