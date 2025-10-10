# DELTA Behavior Agent Guide

This document targets automated agents interacting with the repository. It contains the minimal operational context needed to install dependencies, execute workflows, and contribute code without referring to the human-oriented `README.md`.

## Project Overview and Structure
- `src/behavioral_analysis/`: Installable package that exposes processing (`processing/`), analysis (`analysis/`), I/O helpers (`io/`), and visualization utilities (`visualization/`). Most automation should import from here rather than re-implementing logic.
- `run_one_animal.py`, `process_and_analyze.py`, `process_BM35_complete.py`: Command-line entry points for converting JSON logs to HDF5, running lightweight processing, and generating full session summaries.
- `tests/test_visualizations.py`: Integration-style script that regenerates plots from processed HDF5 files; it is not a traditional unit test.
- `notebooks/`: Exploratory workflows; avoid modifying them unless the task explicitly targets notebooks.
- `outputs/`: Default location for generated artifacts. It is git-ignored and safe for agents to write temporary files.
- Root-level scripts such as `create_licking_rasters.py` and `create_spatial_licking_distribution.py` generate additional derived figures once HDF5 data exists.

## Setup and Environment
- Requires Python 3.9+ (repository code uses modern typing and `pathlib` patterns; 3.8 may work but is unverified).
- Install dependencies:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Add the package to `PYTHONPATH` when running scripts without installing:
  ```bash
  export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
  ```
- Large JSON/HDF5 files are not tracked by git; ensure source logs are present in the workspace before running processing scripts.

## Execution and Build Commands
- Convert raw JSON logs to structured HDF5 (full pipeline, trials enabled):
  ```bash
  python run_one_animal.py /path/to/Log*.json --output-dir outputs/hdf5
  ```
- Lightweight batch conversion (trials disabled by default for speed):
  ```bash
  python process_and_analyze.py /path/to/Log*.json --output-dir outputs/raw
  ```
- Generate figures, PDFs, and lick-aligned analyses from an existing session:
  ```bash
  python process_BM35_complete.py /path/to/Log.json --output-dir outputs/session_summary
  python cue_aligned_licking_analysis.py outputs/hdf5/Log.h5 --output-dir outputs/licking_analysis
  ```
- Additional visualization helpers (`create_licking_rasters.py`, `create_spatial_licking_distribution.py`) assume matching HDF5 and trials CSV files.

## Testing and Quality Assurance
- Run all automated checks with:
  ```bash
  python -m pytest
  ```
  As of this writing pytest collects 9 tests covering corridor detection, corridor metrics, and visualization CLI integration helpers.
- Recommended integration check for visualization outputs (requires processed data with matching trials CSV):
  ```bash
  python tests/test_visualizations.py outputs/hdf5/Log.h5 --trials-csv outputs/hdf5/Log_trials.csv --output-dir outputs/visualization_check
  ```
  This script regenerates key figures and verifies the plotting pipeline. Inspect the console output for warnings and confirm generated files exist.
- When modifying processing logic, rerun `run_one_animal.py` on a representative JSON log and compare resulting HDF5 datasets (especially `/events/Trials` and derived metrics).
- Prefer deterministic operations; avoid randomness in automated workflows unless seeds are explicitly managed.

## Code Style and Conventions
- Follow conventional PEP 8 spacing and naming. Use descriptive, lowercase module names and snake_case functions.
- Add type hints to new public functions; existing modules (for example `src/behavioral_analysis/processing/__init__.py`) already expose typed interfaces.
- Prefer small, single-responsibility functions that can be unit-tested; introduce regression tests alongside new behavior before expanding the implementation.
- Use `pathlib.Path` for filesystem interactions and avoid hard-coded absolute paths.
- Favor pure functions that accept dataframes or arrays and return new objects; keep plotting and file I/O separated when possible.
- Populate docstrings on new modules and functions that agents may call. Inline comments should explain non-obvious transformations or domain-specific constants.
- Notebooks should import functionality from `src/behavioral_analysis` rather than duplicating processing code.

## Security and Data Handling
- JSON logs and derived HDF5/CSV files may contain subject identifiers and experimental metadata. Keep raw data within the repository workspace and avoid committing sensitive outputs.
- No API keys or environment secrets are required. Do not introduce network calls; processing should remain offline.
- Respect the `outputs/` directory convention so generated artifacts stay git-ignored. If alternative locations are required, document them in-code.
- Validate user-provided paths before writing to disk to avoid overwriting input data.

## Git Workflow and Deployment
- Default branch is `main`. Create feature branches for changes and raise pull requests targeting `main`.
- Commit messages should summarize intent (`module: short description` is a concise pattern).
- Keep commits focused; separate formatting-only changes from behavioral modifications.
- There is no automated deployment pipeline. Releases are produced manually by running the processing scripts on curated datasets and distributing the resulting artifacts.

## Agent-Specific Operations
- CI/CD is not configured. Agents should run any relevant processing or visualization commands locally and report resulting files.
- Non-interactive environments should export `PYTHONPATH` and execute the scripts listed above; none require user prompts.
- When generating outputs inside the sandbox, prefer writing to `outputs/` or a subdirectory you create there.
- If a workflow requires sizeable input data, check for its presence before execution and report missing dependencies rather than attempting downloads.
