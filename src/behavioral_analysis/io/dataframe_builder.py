"""
DataFrame Builder for DELTA Behavioral Data

This module provides functions for converting JSON behavioral events to pandas DataFrames.
"""

import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any


def extract_events_by_type(events: List[Dict[str, Any]], verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Extract events from JSON data and organize by type into pandas DataFrames.

    Args:
        events: List of event dictionaries from JSON file
        verbose: Whether to print progress information

    Returns:
        Dictionary of DataFrames, one for each event type
    """
    if verbose:
        print("Extracting events by type...")
        start_time = time.time()

    # Dictionary to store events by type
    events_by_type = {}

    # Process each event
    for event in events:
        event_type = event.get('msg', 'Unknown')

        # Add to events dictionary
        if event_type not in events_by_type:
            events_by_type[event_type] = []

        events_by_type[event_type].append(event)

    # Convert to DataFrames with appropriate structure
    dataframes = {}

    # Process each event type differently based on its structure
    for event_type, event_list in events_by_type.items():
        if not event_list:
            continue

        if event_type == 'Position':
            # Extract position data
            df = pd.DataFrame([{
                'time': event['time'],
                'name': event['data']['name'],
                'position': event['data']['position'][2] if len(event['data']['position']) > 2 else np.nan,
                'heading': event['data']['heading'][2] if len(event['data']['heading']) > 2 else np.nan
            } for event in event_list])

        elif event_type == 'Path Position':
            # Extract path position data
            df = pd.DataFrame([{
                'time': event['time'],
                'name': event['data']['name'],
                'pathName': event['data']['pathName'],
                'position': event['data']['position']
            } for event in event_list])

        elif event_type == 'Lick':
            # Extract lick data (typically just timestamp)
            df = pd.DataFrame([{
                'time': event['time']
            } for event in event_list])

        elif event_type == 'Info':
            # Extract info data
            df = pd.DataFrame([{
                'time': event['time'],
                'session_time': event['data'].get('time', ''),
                'project': event['data'].get('project', ''),
                'scene': event['data'].get('scene', '')
            } for event in event_list])

        elif event_type == 'Log':
            # Extract log data
            df = pd.DataFrame([{
                'time': event['time'],
                'source': event['data'].get('source', ''),
                'msg': str(event['data'].get('msg', ''))
            } for event in event_list])

        elif event_type == 'Cue Result':
            # Extract cue result data
            df = pd.DataFrame([{
                'time': event['time'],
                'id': event['data'].get('id'),
                'id2': event['data'].get('id2'),
                'position': event['data'].get('position'),
                'isRewarding': event['data'].get('isRewarding'),
                'hasGivenReward': event['data'].get('hasGivenReward'),
                'numLicksInReward': event['data'].get('numLicksInReward'),
                'numLicksInPre': event['data'].get('numLicksInPre')
            } for event in event_list])

        elif event_type == 'Cue State':
            # Extract cue state data
            df = pd.DataFrame([{
                'time': event['time'],
                'id': event['data'].get('id'),
                'id2': event['data'].get('id2'),
                'position': event['data'].get('position'),
                'isRewarding': event['data'].get('isRewarding')
            } for event in event_list])

        elif event_type == 'Reward':
            # Extract reward data (typically just timestamp)
            df = pd.DataFrame([{
                'time': event['time']
            } for event in event_list])

        elif event_type == 'Start Period':
            # Extract period data
            df = pd.DataFrame([{
                'time': event['time'],
                'periodType': event['data'].get('periodType'),
                'duration': event['data'].get('duration'),
                'cueSet': event['data'].get('cueSet'),
                'isGuided': event['data'].get('isGuided')
            } for event in event_list])

        elif event_type == 'End Period':
            # Extract period end data
            df = pd.DataFrame([{
                'time': event['time'],
                'periodType': event['data'].get('periodType')
            } for event in event_list])

        elif event_type == 'Linear Controller Settings':
            # Extract controller settings
            df = pd.DataFrame([{
                'time': event['time'],
                'name': event['data'].get('name'),
                'isActive': event['data'].get('isActive'),
                'loopPath': event['data'].get('loopPath'),
                'gain': event['data'].get('gain'),
                'inputSmooth': event['data'].get('inputSmooth')
            } for event in event_list])

        else:
            # Generic extraction for other event types
            # First, try to extract data fields directly
            try:
                records = []
                for event in event_list:
                    record = {'time': event['time']}
                    # Add all data fields
                    if 'data' in event and isinstance(event['data'], dict):
                        for key, value in event['data'].items():
                            record[key] = value
                    records.append(record)
                df = pd.DataFrame(records)
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not extract fields for {event_type}: {e}")
                # Fallback to simple time + data string
                df = pd.DataFrame([{
                    'time': event['time'],
                    'data': str(event['data']) if 'data' in event else ''
                } for event in event_list])

        # Store the DataFrame
        dataframes[event_type] = df

    if verbose:
        end_time = time.time()
        processing_time = end_time - start_time

        print(f"Event extraction completed in {processing_time:.2f} seconds")
        print(f"Extracted {len(dataframes)} event types")

        # Print summary of extracted data
        print("\nEvent types summary:")
        for event_type, df in dataframes.items():
            print(f"  {event_type}: {len(df)} events, {len(df.columns)} columns: {', '.join(df.columns)}")

    return dataframes


def create_combined_dataframe(dataframes: Dict[str, pd.DataFrame], verbose: bool = True) -> Optional[pd.DataFrame]:
    """
    Create a combined DataFrame containing all events with type information.

    Args:
        dataframes: Dictionary of DataFrames by event type
        verbose: Whether to print progress information

    Returns:
        Combined DataFrame or None if no data
    """
    if not dataframes:
        if verbose:
            print("No data to combine")
        return None

    if verbose:
        print("Creating combined DataFrame...")
        start_time = time.time()

    # Initialize empty combined DataFrame
    combined_df = pd.DataFrame()

    # Process each event type
    for event_type, df in dataframes.items():
        if df.empty:
            continue

        # Create a copy to avoid modifying original
        df_copy = df.copy()

        # Add event_type column
        df_copy['event_type'] = event_type

        # Ensure all DataFrames have 'time' column
        if 'time' not in df_copy.columns:
            if verbose:
                print(f"Warning: {event_type} has no time column")
            continue

        # Select columns for combined DataFrame: always include time and event_type
        # Then add any other columns that exist
        cols = ['time', 'event_type']

        # Add position column if it exists
        if 'position' in df_copy.columns:
            cols.append('position')

        # Add common important columns if they exist
        for col in ['id', 'isRewarding', 'hasGivenReward']:
            if col in df_copy.columns:
                cols.append(col)

        # Add all remaining columns
        cols.extend([c for c in df_copy.columns if c not in cols])

        # Append to combined DataFrame
        combined_df = pd.concat([combined_df, df_copy[cols]])

    # Sort by time
    if not combined_df.empty:
        combined_df = combined_df.sort_values('time').reset_index(drop=True)

        if verbose:
            end_time = time.time()
            print(f"Combined {len(combined_df)} events from {len(dataframes)} types in {end_time - start_time:.2f} seconds")

    return combined_df


if __name__ == "__main__":
    # Example usage if this module is run directly
    import argparse
    import sys
    from behavioral_analysis.io.json_parser import parse_json_file

    parser = argparse.ArgumentParser(description='Convert JSON behavioral data to pandas DataFrames')
    parser.add_argument('file_path', help='Path to the JSON file')
    parser.add_argument('--limit', type=int, help='Limit number of events to process')
    parser.add_argument('--sample', action='store_true', help='Print sample rows from each DataFrame')
    parser.add_argument('--combined', action='store_true', help='Create and show combined DataFrame')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    try:
        events = parse_json_file(args.file_path, args.limit, verbose=not args.quiet)
        dataframes = extract_events_by_type(events, verbose=not args.quiet)

        # Print sample data if requested
        if args.sample:
            print("\nSample data for each event type:")
            for event_type, df in dataframes.items():
                print(f"\n{event_type} (showing first 2 rows):")
                print(df.head(2))

        # Create combined DataFrame if requested
        if args.combined:
            combined_df = create_combined_dataframe(dataframes, verbose=not args.quiet)
            if combined_df is not None:
                print("\nCombined DataFrame (first 5 rows):")
                print(combined_df.head())

        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)