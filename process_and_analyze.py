#!/usr/bin/env python3
"""Process the data and create cue-aligned analysis"""

from behavioral_analysis.processing.json_to_hdf5_processor import process_json_to_hdf5

# Process the data
json_path = '/groups/spruston/sprustonlab/mesoscope-data/BM35/2025_09_17/1/Log BM35 2025-09-17 session 1.json'
output_dir = 'behavioral_analysis/output'

print("Processing JSON to HDF5...")
process_json_to_hdf5(json_path, output_dir)
print("Processing complete!")