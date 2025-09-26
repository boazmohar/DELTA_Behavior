"""
Visualization module for DELTA behavioral data.

This module contains functions for visualizing behavioral data,
including position plots, trial rasters, global position plots,
and other visualizations.
"""

from behavioral_analysis.visualization.plotting import (
    plot_position_with_events,
    plot_lick_raster,
    plot_corridor_summary
)

from behavioral_analysis.visualization.global_position_plots import (
    plot_global_position_overview,
    plot_corridor_with_global_position,
    visualize_all_global_positions
)

__all__ = [
    # Standard plotting functions
    'plot_position_with_events',
    'plot_lick_raster',
    'plot_corridor_summary',

    # Global position plotting functions
    'plot_global_position_overview',
    'plot_corridor_with_global_position',
    'visualize_all_global_positions',
]