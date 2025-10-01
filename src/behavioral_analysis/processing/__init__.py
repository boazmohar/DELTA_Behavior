"""
Processing module for DELTA behavioral data.

This module contains functions for processing behavioral data, including:
- Event processing
- Corridor detection
- Trial building
- JSON to HDF5 processing with global position
"""

from behavioral_analysis.processing.corridor_detector_simple import (
    CorridorComputationArtifacts,
    add_corridor_info_to_events,
    annotate_cue_results_with_corridors,
    annotate_cue_states_with_corridors,
    compute_corridor_artifacts,
    detect_corridors_simple,
    detect_position_loops,
    match_cue_states_to_results,
    summarize_corridor_info,
)
from behavioral_analysis.processing.trial_matcher import create_trial_dataframe, calculate_performance_metrics
from behavioral_analysis.processing.json_to_hdf5_processor import process_json_to_hdf5

__all__ = [
    'CorridorComputationArtifacts',
    'detect_corridors_simple',
    'add_corridor_info_to_events',
    'annotate_cue_states_with_corridors',
    'annotate_cue_results_with_corridors',
    'compute_corridor_artifacts',
    'detect_position_loops',
    'match_cue_states_to_results',
    'summarize_corridor_info',
    'create_trial_dataframe',
    'calculate_performance_metrics',
    'process_json_to_hdf5',
]
