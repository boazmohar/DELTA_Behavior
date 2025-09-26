"""
HDF5 Reader for DELTA Behavioral Data

This module provides functions for loading behavioral data from HDF5 format.
"""

import os
import time
import pandas as pd
from typing import Dict, Optional, Any, List, Tuple


def load_from_hdf5(file_path: str,
                   event_types: Optional[List[str]] = None,
                   include_metadata: bool = True,
                   verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Load DataFrames from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file
        event_types: Optional list of event types to load (loads all if None)
        include_metadata: Whether to include metadata in the result
        verbose: Whether to print progress information

    Returns:
        Dictionary of DataFrames by event type, with optional metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If loading fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if verbose:
        print(f"Loading data from HDF5 file: {file_path}")
        start_time = time.time()

    try:
        with pd.HDFStore(file_path, mode='r') as store:
            # Get all keys that start with 'events/'
            all_events = [key for key in store.keys() if key.startswith('/events/')]

            # Parse event types from keys
            available_event_types = [key.split('/')[-1] for key in all_events]

            # Filter to requested event types if specified
            if event_types is not None:
                event_keys = [f'/events/{event_type}' for event_type in event_types
                              if event_type in available_event_types]
            else:
                event_keys = all_events

            # Load each event type
            result = {}
            for key in event_keys:
                event_type = key.split('/')[-1]
                df = store[key]
                result[event_type] = df
                if verbose:
                    print(f"  Loaded {event_type} data ({len(df)} rows)")

            # Load combined data if available
            if '/combined' in store.keys():
                result['combined'] = store['/combined']
                if verbose:
                    print(f"  Loaded combined data ({len(result['combined'])} rows)")

            # Load metadata if requested
            if include_metadata and '/metadata' in store.keys():
                metadata_series = store['/metadata']
                result['metadata'] = metadata_series
                if verbose:
                    print(f"  Loaded metadata")

            if verbose:
                end_time = time.time()
                print(f"Successfully loaded data from {file_path} in {end_time - start_time:.2f} seconds")

            return result

    except Exception as e:
        raise ValueError(f"Error loading from HDF5: {e}")


def get_available_event_types(file_path: str) -> List[str]:
    """
    Get list of available event types in an HDF5 file.

    Args:
        file_path: Path to the HDF5 file

    Returns:
        List of event type names

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If loading fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with pd.HDFStore(file_path, mode='r') as store:
            # Get all keys that start with 'events/'
            event_keys = [key for key in store.keys() if key.startswith('/events/')]
            # Extract just the event type names
            event_types = [key.split('/')[-1] for key in event_keys]
            return event_types
    except Exception as e:
        raise ValueError(f"Error reading event types from HDF5: {e}")


def load_single_event_type(file_path: str, event_type: str) -> pd.DataFrame:
    """
    Load a single event type DataFrame from an HDF5 file.

    Args:
        file_path: Path to the HDF5 file
        event_type: Event type name to load

    Returns:
        DataFrame for the specified event type

    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If event type doesn't exist
        ValueError: If loading fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        with pd.HDFStore(file_path, mode='r') as store:
            key = f'/events/{event_type}'
            if key not in store:
                raise KeyError(f"Event type '{event_type}' not found in {file_path}")
            return store[key]
    except Exception as e:
        if isinstance(e, KeyError):
            raise
        raise ValueError(f"Error loading event type from HDF5: {e}")


if __name__ == "__main__":
    # Example usage if this module is run directly
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Load behavioral data from HDF5')
    parser.add_argument('file_path', help='Path to input HDF5 file')
    parser.add_argument('--event-types', '-e', nargs='+', help='Specific event types to load')
    parser.add_argument('--list', action='store_true', help='List available event types and exit')
    parser.add_argument('--metadata', action='store_true', help='Display metadata and exit')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    try:
        if args.list:
            # Just list available event types
            event_types = get_available_event_types(args.file_path)
            print(f"Available event types in {args.file_path}:")
            for event_type in sorted(event_types):
                print(f"  {event_type}")
            sys.exit(0)

        if args.metadata:
            # Just show metadata
            with pd.HDFStore(args.file_path, mode='r') as store:
                if '/metadata' in store:
                    metadata = store['/metadata']
                    print(f"Metadata for {args.file_path}:")
                    for key, value in metadata.items():
                        print(f"  {key}: {value}")
                else:
                    print(f"No metadata found in {args.file_path}")
            sys.exit(0)

        # Load data
        data = load_from_hdf5(
            args.file_path,
            event_types=args.event_types,
            verbose=not args.quiet
        )

        # Print summary
        if not args.quiet:
            print("\nData loaded successfully:")
            for event_type, df in data.items():
                if event_type == 'metadata':
                    continue
                print(f"  {event_type}: {len(df)} rows, columns: {', '.join(df.columns)}")

        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)