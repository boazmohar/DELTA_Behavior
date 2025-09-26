"""
HDF5 Writer for DELTA Behavioral Data

This module provides functions for saving behavioral data to HDF5 format.
"""

import os
import time
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any, List, Tuple


def save_to_hdf5(dataframes: Dict[str, pd.DataFrame],
                output_file: str,
                combined_df: Optional[pd.DataFrame] = None,
                metadata: Optional[Dict[str, Any]] = None,
                overwrite: bool = True,
                verbose: bool = True) -> str:
    """
    Save DataFrames to HDF5 format.

    Args:
        dataframes: Dictionary of DataFrames by event type
        output_file: Path for output HDF5 file
        combined_df: Optional combined DataFrame to include
        metadata: Optional metadata dictionary to include
        overwrite: Whether to overwrite existing file
        verbose: Whether to print progress information

    Returns:
        Path to the saved file

    Raises:
        FileExistsError: If file exists and overwrite is False
        ValueError: If saving fails
    """
    if os.path.exists(output_file) and not overwrite:
        raise FileExistsError(f"Output file {output_file} already exists and overwrite=False")

    if verbose:
        print(f"Saving data to HDF5 file: {output_file}")
        start_time = time.time()

    try:
        with pd.HDFStore(output_file, mode='w') as store:
            # Save each individual DataFrame
            for event_type, df in dataframes.items():
                if not df.empty:
                    # Create a safe key name (no spaces)
                    safe_event_type = event_type.replace(' ', '_')
                    store[f'events/{safe_event_type}'] = df
                    if verbose:
                        print(f"  Saved {event_type} data ({len(df)} rows)")

            # Save combined DataFrame if provided
            if combined_df is not None and not combined_df.empty:
                store['combined'] = combined_df
                if verbose:
                    print(f"  Saved combined data ({len(combined_df)} rows)")

            # Save metadata
            if metadata is None:
                metadata = {}

            # Add standard metadata
            metadata.update({
                'timestamp': datetime.now().isoformat(),
                'event_types': ', '.join(dataframes.keys()),
                'event_counts': ', '.join([f"{k}: {len(v)}" for k, v in dataframes.items()]),
                'total_events': sum(len(df) for df in dataframes.values())
            })

            store['metadata'] = pd.Series(metadata)
            if verbose:
                print(f"  Saved metadata")

        if verbose:
            end_time = time.time()
            print(f"Successfully saved all data to {output_file} in {end_time - start_time:.2f} seconds")

        return output_file

    except Exception as e:
        raise ValueError(f"Error saving to HDF5: {e}")


def read_hdf5_metadata(file_path: str) -> pd.Series:
    """
    Read metadata from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file

    Returns:
        pandas Series containing metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If metadata can't be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with pd.HDFStore(file_path, mode='r') as store:
            if '/metadata' in store:
                return store['metadata']
            else:
                raise ValueError(f"No metadata found in {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading HDF5 metadata: {e}")


def list_hdf5_contents(file_path: str) -> List[Tuple[str, int, List[str]]]:
    """
    List contents of an HDF5 file.

    Args:
        file_path: Path to the HDF5 file

    Returns:
        List of tuples (dataset_name, row_count, column_names)

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If contents can't be read
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        contents = []
        with pd.HDFStore(file_path, mode='r') as store:
            for key in store.keys():
                if key == '/metadata':
                    # Skip metadata, handle separately
                    continue

                df = store[key]
                contents.append((key, len(df), list(df.columns)))

        return contents
    except Exception as e:
        raise ValueError(f"Error reading HDF5 contents: {e}")


if __name__ == "__main__":
    # Example usage if this module is run directly
    import argparse
    import sys
    from behavioral_analysis.io.json_parser import parse_json_file
    from behavioral_analysis.io.dataframe_builder import extract_events_by_type, create_combined_dataframe

    parser = argparse.ArgumentParser(description='Save behavioral data to HDF5')
    parser.add_argument('file_path', help='Path to input JSON file')
    parser.add_argument('--output', '-o', help='Output HDF5 file path')
    parser.add_argument('--limit', type=int, help='Limit number of events to process')
    parser.add_argument('--no-overwrite', action='store_true', help='Do not overwrite existing file')
    parser.add_argument('--include-combined', action='store_true', help='Include combined DataFrame')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    # Generate default output path if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.file_path))[0]
        args.output = f"{base_name}.h5"

    try:
        # Parse JSON
        events = parse_json_file(args.file_path, args.limit, verbose=not args.quiet)

        # Extract DataFrames
        dataframes = extract_events_by_type(events, verbose=not args.quiet)

        # Create combined DataFrame if requested
        combined_df = None
        if args.include_combined:
            combined_df = create_combined_dataframe(dataframes, verbose=not args.quiet)

        # Save to HDF5
        output_file = save_to_hdf5(
            dataframes,
            args.output,
            combined_df=combined_df,
            overwrite=not args.no_overwrite,
            verbose=not args.quiet
        )

        # Print success message
        if not args.quiet:
            print(f"\nHDF5 file created successfully: {output_file}")
            print("To explore this file, you can use:")
            print("```python")
            print("import pandas as pd")
            print(f"store = pd.HDFStore('{output_file}')")
            print("# List all available datasets")
            print("print(store.keys())")
            print("# Get metadata")
            print("metadata = store['metadata']")
            print("# Load event data")
            print("position_df = store['events/Position']")
            print("# Close the store when done")
            print("store.close()")
            print("```")

        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)