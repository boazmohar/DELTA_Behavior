"""
JSON Parser for DELTA Behavioral Data

This module provides functions for loading and parsing JSON behavioral data files.
"""

import json
import os
import time
from typing import Dict, List, Optional, Union, Any


def parse_json_file(file_path: str, limit: Optional[int] = None, verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Parse a JSON file containing behavioral data and return the events.

    Args:
        file_path: Path to the JSON file
        limit: Optional limit on number of events to process
        verbose: Whether to print progress information

    Returns:
        List of event dictionaries

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file can't be parsed as JSON
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if verbose:
        print(f"Parsing file: {file_path}")
        start_time = time.time()

        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"File size: {file_size / (1024 * 1024):.2f} MB")

    events = []
    event_count = 0
    event_types = {}

    try:
        with open(file_path, 'r') as f:
            # Skip first character if it's a [
            first_char = f.read(1)
            if first_char != '[':
                f.seek(0)  # Go back to start if not [

            # Read line by line
            for line in f:
                line = line.strip()

                # Skip empty lines
                if not line:
                    continue

                # Remove trailing comma if present
                if line.endswith(','):
                    line = line[:-1]

                # Skip if it's just a closing bracket
                if line == ']':
                    continue

                try:
                    # Try to parse this line as a complete JSON object
                    obj = json.loads(line)

                    # Count event types
                    event_type = obj.get('msg', 'Unknown')
                    event_types[event_type] = event_types.get(event_type, 0) + 1

                    # Add to events list
                    events.append(obj)
                    event_count += 1

                    # Print progress periodically
                    if verbose and event_count % 10000 == 0:
                        print(f"Processed {event_count} events...")

                    # Stop if limit reached
                    if limit and event_count >= limit:
                        if verbose:
                            print(f"Reached event limit ({limit})")
                        break

                except json.JSONDecodeError:
                    # If it fails, the line might be incomplete or malformed
                    if verbose:
                        print(f"Warning: Could not parse line: {line[:100]}...")
                    continue

    except Exception as e:
        raise ValueError(f"Error parsing JSON file: {e}")

    if verbose:
        end_time = time.time()
        processing_time = end_time - start_time

        print(f"\nProcessing completed in {processing_time:.2f} seconds")
        print(f"Total events: {event_count}")

        # Print event type summary
        print("\nEvent counts by type:")
        for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {event_type}: {count}")

    return events


def get_event_types(events: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Count the occurrences of each event type in a list of events.

    Args:
        events: List of event dictionaries

    Returns:
        Dictionary mapping event types to counts
    """
    event_types = {}
    for event in events:
        event_type = event.get('msg', 'Unknown')
        event_types[event_type] = event_types.get(event_type, 0) + 1

    return event_types


if __name__ == "__main__":
    # Example usage if this module is run directly
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='Parse JSON behavioral data')
    parser.add_argument('file_path', help='Path to the JSON file')
    parser.add_argument('--limit', type=int, help='Limit number of events to process')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')

    args = parser.parse_args()

    try:
        events = parse_json_file(args.file_path, args.limit, verbose=not args.quiet)

        # Print first event as example
        if events:
            print("\nFirst event example:")
            print(json.dumps(events[0], indent=2))

        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)