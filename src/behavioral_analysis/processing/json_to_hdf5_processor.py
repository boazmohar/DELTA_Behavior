#!/usr/bin/env python3
"""
JSON to HDF5 Processor with Global Position

This module provides functions for processing JSON behavioral data to HDF5 format
with added corridor detection and global monotonic position calculation.

It integrates the entire pipeline:
1. Parse JSON data
2. Convert to DataFrames
3. Detect corridors and calculate global position
4. Save to HDF5

Usage:
    from behavioral_analysis.processing.json_to_hdf5_processor import process_json_to_hdf5

    # Process a file with one function call
    process_json_to_hdf5('input.json', 'output.h5')
"""

import os
import sys
import time
import argparse
from typing import Dict, Optional, Any

import pandas as pd

from behavioral_analysis.io.json_parser import parse_json_file
from behavioral_analysis.io.dataframe_builder import extract_events_by_type, create_combined_dataframe
from behavioral_analysis.io.hdf5_writer import save_to_hdf5
from behavioral_analysis.processing.corridor_detector_simple import detect_corridors_simple, add_corridor_info_to_events
from behavioral_analysis.processing.trial_matcher import create_trial_dataframe, calculate_performance_metrics


def process_json_to_hdf5(
    input_file: str,
    output_file: Optional[str] = None,
    corridor_length_cm: float = 500.0,
    include_combined: bool = False,
    include_trials: bool = True,
    enable_monotonic_position: bool = True,
    limit: Optional[int] = None,
    overwrite: bool = True,
    verbose: bool = True
) -> str:
    """
    Process JSON behavioral data to HDF5 with corridor detection and global position.

    This function integrates the entire processing pipeline in one call, including:
    - JSON parsing
    - DataFrame conversion
    - Corridor detection
    - Global position calculation
    - HDF5 export

    Args:
        input_file: Path to the input JSON file
        output_file: Path for the output HDF5 file (default: derived from input file)
        corridor_length_cm: Length of corridors in centimeters (default: 500 cm)
        include_combined: Whether to include combined events DataFrame (default: False)
        include_trials: Whether to create trial-based dataframe (default: True)
        enable_monotonic_position: Whether to add monotonic global position (default: True)
        limit: Optional limit on number of events to process (default: None)
        overwrite: Whether to overwrite existing output file (default: True)
        verbose: Whether to print progress information (default: True)

    Returns:
        Path to the saved HDF5 file

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If required data is missing or processing fails
    """
    if verbose:
        print(f"Processing JSON data with global position: {input_file}")
        start_time = time.time()

    # Validate input file
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Determine output path if not specified
    if not output_file:
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = f"{base_name}_with_global_position.h5"

    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if verbose:
            print(f"Created output directory: {output_dir}")

    # Check if output file exists
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(f"Output file exists and overwrite=False: {output_file}")

    # Step 1: Parse JSON file
    if verbose:
        print(f"Step 1: Parsing JSON file")

    events = parse_json_file(
        input_file,
        limit=limit,
        verbose=verbose
    )

    # Step 2: Extract events to DataFrames
    if verbose:
        print(f"Step 2: Extracting events to DataFrames")

    dataframes = extract_events_by_type(
        events,
        verbose=verbose
    )

    # Step 3: Detect corridors and calculate global position
    if verbose:
        print(f"Step 3: Detecting corridors and calculating global position")

    # Check if we have the necessary data for corridor detection (Cue State/Cue_State)
    cue_state_key = None
    for key in dataframes.keys():
        # Handle both formats (with space or underscore)
        if key == 'Cue_State' or key == 'Cue State':
            cue_state_key = key
            break

    if cue_state_key is None:
        raise ValueError("Input data must contain Cue State events for corridor detection")

    # Get cue, position, and result data
    cue_df = dataframes[cue_state_key]
    position_df = dataframes.get('Position')

    cue_result_df = None
    for candidate in ('Cue_Result', 'Cue Result'):
        if candidate in dataframes:
            cue_result_df = dataframes[candidate]
            break

    # Detect corridors using simple cue ID counting
    corridor_info, position_with_corridors = detect_corridors_simple(
        cue_df,
        position_df,
        corridor_length_cm=corridor_length_cm,
        verbose=verbose,
        cue_result_df=cue_result_df,
    )

    # Add corridor and global position information to all events
    if verbose:
        print("Adding global position to all events...")

    # Update position data first
    if position_with_corridors is not None:
        dataframes['Position'] = position_with_corridors

    updated_events = add_corridor_info_to_events(
        dataframes,
        corridor_info,
        corridor_length_cm=corridor_length_cm,
        verbose=verbose,
        position_df=position_with_corridors
    )

    # Add corridor info to the updated events
    updated_events['Corridor_Info'] = corridor_info

    # Step 3.5: Create trial-based dataframe if requested
    trials_df = None
    if include_trials:
        if verbose:
            print(f"Step 3.5: Creating trial-based dataframe")

        # Check for Cue_Result events
        cue_result_key = None
        for key in updated_events.keys():
            if key == 'Cue_Result' or key == 'Cue Result':
                cue_result_key = key
                break

        if cue_result_key:
            trials_df = create_trial_dataframe(
                updated_events[cue_state_key],
                updated_events[cue_result_key],
                corridor_length_cm=corridor_length_cm,
                verbose=verbose,
                position_df=updated_events.get('Position')
            )
            updated_events['Trials'] = trials_df

            # Calculate performance metrics
            if verbose:
                metrics = calculate_performance_metrics(trials_df)
                print(f"  Performance: Accuracy={metrics['accuracy']*100:.1f}%, "
                      f"Hit rate={metrics['hit_rate']*100:.1f}%, "
                      f"FA rate={metrics['fa_rate']*100:.1f}%")
        else:
            if verbose:
                print("  Warning: No Cue_Result events found, skipping trial creation")

    # Step 4: Create combined DataFrame if requested
    combined_df = None
    if include_combined:
        if verbose:
            print(f"Step 4: Creating combined DataFrame")

        combined_df = create_combined_dataframe(
            updated_events,
            verbose=verbose
        )
    elif verbose:
        print(f"Step 4: Skipping combined DataFrame creation (not requested)")

    # Step 5: Save to HDF5
    if verbose:
        print(f"Step 5: Saving to HDF5 file: {output_file}")

    # Prepare metadata
    metadata = {
        'source_file': input_file,
        'corridor_length_cm': corridor_length_cm,
        'enable_monotonic_position': enable_monotonic_position,
        'num_corridors': len(corridor_info),
        'include_combined': include_combined
    }

    # Save to HDF5
    save_to_hdf5(
        updated_events,
        output_file,
        combined_df=combined_df,
        metadata=metadata,
        overwrite=overwrite,
        verbose=verbose
    )

    if verbose:
        end_time = time.time()
        print(f"Processing completed in {end_time - start_time:.2f} seconds")
        print(f"Output file: {output_file}")

    return output_file


def main():
    """
    Command-line entry point for the processor.
    """
    parser = argparse.ArgumentParser(
        description='Process JSON behavioral data to HDF5 with global position',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/output options
    parser.add_argument('input_file',
                        help='Path to input JSON file with behavioral data')
    parser.add_argument('--output', '-o',
                        help='Output file path (default: derived from input filename)')
    parser.add_argument('--limit', '-l', type=int,
                        help='Limit number of events to process (for testing)')
    parser.add_argument('--no-overwrite',
                        action='store_true',
                        help='Do not overwrite existing output file')

    # Processing options
    parser.add_argument('--corridor-length', type=float, default=500.0,
                        help='Length of each corridor in centimeters')
    parser.add_argument('--disable-monotonic-position',
                        action='store_true',
                        help='Disable monotonic global position calculation')

    # Data options
    parser.add_argument('--include-combined',
                        action='store_true',
                        help='Include combined events DataFrame in output')

    # Verbosity options
    parser.add_argument('--quiet', '-q',
                        action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    try:
        # Process the file
        output_file = process_json_to_hdf5(
            input_file=args.input_file,
            output_file=args.output,
            corridor_length_cm=args.corridor_length,
            include_combined=args.include_combined,
            enable_monotonic_position=not args.disable_monotonic_position,
            limit=args.limit,
            overwrite=not args.no_overwrite,
            verbose=not args.quiet
        )

        if not args.quiet:
            # Print example usage
            print("\nTo use this file in Python:")
            print("```python")
            print("import pandas as pd")
            print(f"store = pd.HDFStore('{output_file}')")
            print("# List all available datasets")
            print("print(store.keys())")
            print("# Get metadata")
            print("metadata = store['metadata']")
            print("# Load position data with global position")
            print("position_df = store['events/Position']")
            print("# View global position")
            print("print(position_df[['time', 'position_cm', 'global_position_cm']].head())")
            print("# Close the store when done")
            print("store.close()")
            print("```")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
