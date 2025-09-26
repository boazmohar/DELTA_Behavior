"""
Processing module for DELTA behavioral data.

This module contains functions for processing behavioral data, including:
- Event processing
- Corridor detection
- Trial building
- JSON to HDF5 processing with global position
"""

from behavioral_analysis.processing.corridor_detector_simple import detect_corridors_simple, add_corridor_info_to_events
from behavioral_analysis.processing.trial_matcher import create_trial_dataframe, calculate_performance_metrics
from behavioral_analysis.processing.json_to_hdf5_processor import process_json_to_hdf5

__all__ = [
    'detect_corridors_simple',
    'add_corridor_info_to_events',
    'create_trial_dataframe',
    'calculate_performance_metrics',
    'process_json_to_hdf5',
]