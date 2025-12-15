# DELTA Behavior x Turnover Quick Reference

## Goal
- Relate behavioral learning metrics (d-prime slope, time to criterion, within-day gain, etc.) to synaptic turnover (tau/fraction pulse) across brain regions and animals.

## Behavioral Data
- Processed session data: `outputs/hdf5/<animal>/Log ... .h5` (trials, cue info).
- Behavioral summaries/notebooks:
  - `notebooks/dprime_learning_after_switch.ipynb` builds behavior metrics and saves `outputs/dprime_learning/dprime_learning_summary.csv` (and timestamped HDF5/CSV variants).
  - Metrics include: `slope`, `intercept`, `mean_accuracy`, `mean_dprime`, `time_to_criterion_trial`, `within_day_gain_sum`.
- Per-animal means used for centering behavioral-derived quantities are computed in the notebook; key dicts: `animal_all_tau` (global tau pool) and `animal_global_mean` / `animal_global_frac_mean` (per-animal global means for centering images/fractions).

## Turnover Data
- Parquet sources: `/nearline/spruston/Boaz/DELTA/GluA2_round_12.parquet` and `..._13.parquet` (fields: `ANM`, `Name`, `tau`, `tau_values`, `N`, `Round`, `File`, `CCF_ID`, etc.).
- Filtering: exclude white matter/fiber tracts/grooves via `structure_tree_safe_2017.csv` using `structure_id_path` (IDs 1009, 73, 1024) and `valid == 1`.
- Region grouping typically by `Name` (full CCF region), with region size `region_N_sum = sum(N)`.
- Bootstraps/robustness: stored in `boot_corr`/`boot_summary` (HDF5/CSV in `outputs/dprime_learning/`), with fields like `slope_mean`, `slope_abs_mean`, `frac_sig_p05`, `region_N_sum`.

## Image Data and Linking
- Raw images per round:
  - Round 12: `/nearline/spruston/Boaz/Delta/GluA2_Round12`
  - Round 13: `/nearline/spruston/Boaz/Delta/GluA2_Round13`
- Channel naming: base filename + `_CY5` (pulse), `_CY3` (chase); optional `_DAPI`. Atlas PNG: `_nl.png` with RGB codes from `labels.txt` (repo root).
- Fraction pulse (computed): `pulse / (pulse + chase)` images saved to `outputs/fraction_pulse_whole_slide/<ANM>_<base>_fraction.tif`.
- Lifetime conversion: `tau_days = -4.0 / ln(fraction)` for 4-day pulse-chase; beware values near 0/1.
- ROI overlay:
  - Decode `_nl.png` via `labels.txt` hex -> CCF ID -> mask (resize to fraction shape).
  - Overlay contour on fraction/tau; titles include behavior metrics for that animal.
- Per-animal global fraction mean: computed by pooling all fraction images per animal (used to center each slice: `frac_centered = frac - animal_global_frac_mean[ani]`).

## Notebooks and Code Style
- Working notebooks: `turnover_summary.ipynb` (filters, correlations, bootstraps, image overlays), `dprime_learning_after_switch.ipynb` (behavior metrics).
- Style: Pandas + NumPy, `tifffile` for images, `tqdm` for progress. Prefer explicit filtering up front; avoid silent `continue` unless necessary. Keep masks resized to image shape before overlaying.
- Saved artifacts: `outputs/dprime_learning/` (behavior summaries, volcano plots, bootstraps), `outputs/fraction_pulse_whole_slide/` (fraction TIFs, overlays).

## Useful Variables (notebook)
- `behavior`: DataFrame of behavioral metrics keyed by `ANM`.
- `animal_all_tau`: dict of per-animal global tau pools (for centering).
- `animal_global_frac_mean`: per-animal mean fraction (centering images).
- `region_animal_arrays`: per-region per-animal tau arrays for bootstraps.
- `region_pixels` / `region_N_sum`: region size (sum of `N`).

## Next Steps (co-registration angle)
- Use `labels.txt` and `_nl.png` to extract region masks and align across animals.
- Consider building a small loader that pairs fraction images with atlas masks and behavior rows by `ANM` and `Name` for pixelwise/voxelwise analyses.
