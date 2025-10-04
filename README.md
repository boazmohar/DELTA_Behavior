# DELTA Behavior Analysis

A Python package for processing, analyzing, and visualizing behavioral data from DELTA experiments. This package provides tools for converting JSON behavioral data to HDF5 format with corridor detection, trial identification, and behavioral outcome classification.

## Overview

This repository contains tools for handling behavioral data from the DELTA experimental setup. The main functionality includes:

- Parsing JSON behavioral data files
- Converting events to pandas DataFrames
- Detecting corridors based on Cue_State ID cycling
- Creating trial-based dataframes with behavioral outcomes (Hit/Miss/FA/CR)
- Calculating global position across corridors
- Saving structured data to HDF5 format

## Repository Structure

```
DELTA_Behavior/
├── src/
│   └── behavioral_analysis/      # Main package
│       ├── __init__.py
│       ├── io/                   # Input/Output functionality
│       ├── processing/           # Data processing pipeline
│       └── visualization/        # Plotting helpers
├── tests/                        # Integration-style scripts & checks
│   └── test_visualizations.py
├── notebooks/                    # (Add exploratory notebooks here)
├── outputs/                      # Generated artifacts (ignored by git)
├── run_one_animal.py             # Example processing script
├── process_and_analyze.py        # Minimal processing CLI
└── README.md                     # This file
```

## Quick Start

Add the `src/` directory to your `PYTHONPATH` (or install the package) before importing:

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

### Basic Usage

```python
from behavioral_analysis.processing import process_json_to_hdf5

# Process JSON file with all features
result = process_json_to_hdf5(
    input_file="path/to/data.json",
    output_file="output.h5",
    corridor_length_cm=500.0,
    include_trials=True,  # Create trial dataframe
    verbose=True
)
```

### Command-Line Helpers

Scripts in the repository expose the same functionality without hard-coded paths:

```bash
# Convert one or more JSON logs to HDF5 (outputs derive from each filename)
python run_one_animal.py /path/to/Log*.json --output-dir outputs/hdf5

# Lightweight converter (trials disabled by default for faster batch runs)
python process_and_analyze.py /path/to/Log*.json --output-dir outputs/raw

# Generate visual summaries for a processed session
python process_BM35_complete.py /path/to/Log.json --output-dir outputs/session_summary

# Build cue-aligned licking analysis from any processed HDF5
python cue_aligned_licking_analysis.py outputs/hdf5/Log.h5 --output-dir outputs/licking_analysis

# Produce figures/PDF from existing artifacts
python tests/test_visualizations.py outputs/hdf5/Log.h5 --trials-csv outputs/hdf5/Log_trials.csv
```

### Accessing the Processed Data

```python
import pandas as pd

# Open the HDF5 store
with pd.HDFStore('output.h5', 'r') as store:
    # Access trial data
    trials = store['/events/Trials']

    # Access corridor information
    corridors = store['/events/Corridor_Info']

    # Access raw event data with global positions
    position = store['/events/Position']
    cue_state = store['/events/Cue_State']
    cue_result = store['/events/Cue_Result']
```

## Key Features

### 1. Simplified Corridor Detection

Corridors are detected based on the cycling of Cue_State IDs (0-6):
- Each corridor contains 7 cues (IDs 0-6)
- When ID returns to 0, a new corridor begins
- No complex merging of teleports or position resets

### 2. Global Position Calculation

Global position is calculated using a simple formula:
```
global_position_cm = (corridor_id - 1) * 500 + position_cm
```
- Corridor 1 starts at global position 0 cm
- Each corridor adds 500 cm to the global position
- Consistent reference frame across all events

### 3. Robust Trial Matching

Cue_State and Cue_Result events are matched using:
- **Exact position matching**: Events must have the same position value
- **Time constraint**: Result must occur after State
- **Removal strategy**: Matched events are removed from pools to prevent duplicates
- Achieves 100% position accuracy for all matched trials

### 4. Trial-Based Analysis

Each trial includes:
- **Unique trial ID**: Sequential numbering from 0 to N-1
- **Behavioral outcome classification**:
  - **Hit**: Rewarding cue + licked (gave reward)
  - **Miss**: Rewarding cue + no lick
  - **FA (False Alarm)**: Non-rewarding cue + licked
  - **CR (Correct Rejection)**: Non-rewarding cue + no lick
- **Timing information**: Cue onset, hit time, reaction time
- **Response metrics**: Number of licks, reward delivery

## Data Structure

### Event Types

The behavioral data includes several event types:

1. **Position**: Mouse position data
   - Original columns: `time`, `name`, `position`, `heading`
   - Added columns: `corridor_id`, `position_cm`, `global_position_cm`

2. **Cue_State**: Cue information at corridor creation
   - Columns: `time`, `id`, `id2`, `position`, `isRewarding`
   - IDs cycle 0-6, with ID=0 marking new corridors

3. **Cue_Result**: Results when cues are hit
   - Columns: `time`, `id`, `id2`, `position`, `isRewarding`, `hasGivenReward`, `numLicksInPre`, `numLicksInReward`

4. **Lick**: Licking events
   - Columns: `time`

5. **Reward**: Reward delivery events
   - Columns: `time`

### HDF5 File Structure

The processed HDF5 file contains:

```
/events/
├── Position              # Position data with global coordinates
├── Cue_State            # Cue creation events
├── Cue_Result           # Cue hit events
├── Lick                 # Lick events
├── Reward               # Reward events
├── Corridor_Info        # Corridor timing and metadata
├── Trials               # Trial-based dataframe (if enabled)
└── [other event types]
/metadata                # Processing metadata
```

### Trials DataFrame Structure

The Trials dataframe contains the following columns:

| Column | Description |
|--------|-------------|
| `trial_id` | Unique trial identifier (0 to N-1) |
| `corridor` | Corridor number (0-based) |
| `cue_id_in_corridor` | Cue ID within corridor (0-6) |
| `cue_type` | 'Rewarding' or 'Non-rewarding' |
| `is_rewarding` | Boolean flag for rewarding cues |
| `cue_position` | Position value in arbitrary units |
| `cue_position_cm` | Position in centimeters |
| `global_position_cm` | Global position across all corridors |
| `cue_onset_ms` | Time when cue was created |
| `cue_hit_ms` | Time when cue was hit |
| `reaction_time_s` | Time between creation and hit (seconds) |
| `num_licks_pre` | Licks before reward window |
| `num_licks_reward` | Licks in reward window |
| `gave_reward` | Whether reward was delivered |
| `outcome` | 'Hit', 'Miss', 'FA', or 'CR' |
| `correct` | Boolean for correct response |
| `licked` | Boolean for any licking response |
| `session_time_min` | Time in session (minutes) |

## Example Analysis

### Basic Performance Metrics

```python
import pandas as pd

# Load trial data
with pd.HDFStore('output.h5', 'r') as store:
    trials = store['/events/Trials']

# Calculate performance metrics
hit_rate = (trials['outcome'] == 'Hit').sum() / trials['is_rewarding'].sum()
fa_rate = (trials['outcome'] == 'FA').sum() / (~trials['is_rewarding']).sum()
accuracy = trials['correct'].mean()

print(f"Hit rate: {hit_rate:.1%}")
print(f"False alarm rate: {fa_rate:.1%}")
print(f"Overall accuracy: {accuracy:.1%}")

# Performance by corridor
performance_by_corridor = trials.groupby('corridor').agg({
    'correct': 'mean',
    'reaction_time_s': 'mean',
    'num_licks_reward': 'mean'
})
```

### Visualizing Behavioral Data

The package includes comprehensive visualization tools in `behavioral_analysis.visualization.trial_visualizer`:

```python
from behavioral_analysis.visualization.trial_visualizer import (
    load_data,
    plot_trial_outcomes_timeline,
    plot_performance_summary,
    plot_position_trace_with_events,
    plot_learning_curves,
    create_session_report
)

# Load data from HDF5 and/or CSV
hdf5_data, trials_df = load_data('output.h5', 'trials.csv')

# 1. Trial outcomes over time
fig = plot_trial_outcomes_timeline(trials_df)

# 2. Comprehensive performance summary
fig = plot_performance_summary(trials_df)

# 3. Position trace with behavioral events
fig = plot_position_trace_with_events(hdf5_data, trials_df, time_range=(0, 10))

# 4. Learning curves
fig = plot_learning_curves(trials_df, bin_size=50)

# 5. Generate complete session report (PDF)
create_session_report('output.h5', 'trials.csv', save_path='session_report.pdf')
```

#### Available Visualizations

1. **Trial Outcomes Timeline**: Shows all trials color-coded by outcome with running accuracy
2. **Performance Summary**: Multi-panel display with:
   - Outcome distribution (pie chart)
   - Performance metrics (bar chart)
   - Reaction time distribution
   - Performance by corridor
   - Licking behavior analysis
   - Reaction time by cue type

3. **Position Trace with Events**: Synchronized display of:
   - Mouse position over time
   - Trial outcomes overlaid
   - Lick raster plot
   - Reward delivery markers

4. **Learning Curves**: Session progression analysis with:
   - Hit rate and FA rate trends
   - D-prime (sensitivity) over time
   - Reaction time changes
   - Licking behavior evolution

5. **Session Report**: Comprehensive PDF with all visualizations and statistics

#### Command-Line Usage

The visualization module can also be used from the command line:

```bash
# Generate a session report from HDF5 file
python -m behavioral_analysis.visualization.trial_visualizer data.h5 --save report.pdf

# Include CSV file for trials
python -m behavioral_analysis.visualization.trial_visualizer data.h5 --csv trials.csv --save report.pdf
```

## Position Units and Conversion

- **Arbitrary units**: Raw position values range from 0 to 50,000
- **Conversion factor**: 100 (50,000 AU = 500 cm)
- **Corridor length**: 500 cm
- **Position formula**: `position_cm = position_AU / 100`

## Processing Pipeline

1. **Parse JSON**: Extract events from JSON file
2. **Convert to DataFrames**: Organize events by type
3. **Detect Corridors**: Identify corridors from Cue_State ID cycling
4. **Calculate Global Positions**: Apply formula to all events with position
5. **Match Trials**: Robustly match Cue_State to Cue_Result events
6. **Classify Outcomes**: Determine Hit/Miss/FA/CR for each trial
7. **Save to HDF5**: Store all processed data

## API Reference

### Main Processing Function

```python
process_json_to_hdf5(
    input_file: str,
    output_file: Optional[str] = None,
    corridor_length_cm: float = 500.0,
    include_combined: bool = False,
    include_trials: bool = True,
    enable_monotonic_position: bool = True,
    limit: Optional[int] = None,
    overwrite: bool = True,
    verbose: bool = True
) -> str
```

**Parameters:**
- `input_file`: Path to input JSON file
- `output_file`: Path for output HDF5 (default: auto-generated)
- `corridor_length_cm`: Length of each corridor (default: 500)
- `include_combined`: Include combined DataFrame (default: False)
- `include_trials`: Create trial-based dataframe (default: True)
- `enable_monotonic_position`: Ensure monotonic global positions (default: True)
- `limit`: Limit number of events to process (default: None)
- `overwrite`: Overwrite existing output file (default: True)
- `verbose`: Print progress information (default: True)

**Returns:** Path to the saved HDF5 file

### Helper Functions

```python
from behavioral_analysis.processing import (
    detect_corridors_simple,
    create_trial_dataframe,
    calculate_performance_metrics
)

# Detect corridors from Cue_State events
corridor_info, position_with_corridors = detect_corridors_simple(
    cue_state_df, position_df, corridor_length_cm=500.0
)

# Create trial dataframe
trials_df = create_trial_dataframe(
    cue_state_df, cue_result_df, corridor_length_cm=500.0
)

# Calculate performance metrics
metrics = calculate_performance_metrics(trials_df)
```

## Dependencies

- Python 3.6+
- Runtime Python packages are listed in [`requirements.txt`](requirements.txt) and include core libraries such as pandas, numpy, scipy, matplotlib, seaborn, h5py, and tables (PyTables).

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/DELTA_Behavior.git
cd DELTA_Behavior

# Install dependencies
pip install -r requirements.txt
```

## Notes

- Only includes trials where cues were actually hit (used in experiment)
- Unhit cues at the end of sessions are excluded from trial analysis
- Position matching is exact - cues must have identical position values
- Global position provides consistent reference frame across entire session

## Version History

- **v2.0**: Simplified corridor detection, robust trial matching, behavioral outcome classification
- **v1.0**: Initial version with complex corridor detection
