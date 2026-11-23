# Post-task change in resting EEG and depressive versus anxious symptoms

Analysis code for the study of how eyes-closed resting EEG changes between two resting sessions recorded about one hour apart, and how this change relates to depressive (Beck Depression Inventory) and anxious (Spielberger Trait Anxiety Inventory) symptoms.

## Data

OpenNeuro ds003478 (EEG: Depression rest), version 1.1.0, https://doi.org/10.18112/openneuro.ds003478.v1.1.0. Download the dataset and set `dataset_root` in `config.yaml` to its folder. No other data are needed.

## Requirements

Python 3.13 and the packages in `requirements.txt` (exact versions). Install with `pip install -r requirements.txt`. All computation runs on a CPU.

## Running

`bash "run all.sh"` runs the full pipeline:

1. `extract.py`: removes ocular and cardiac channels, average reference, 1 to 40 Hz filter, resampling to 128 Hz, 2 s epochs, amplitude-based epoch rejection; checks eye-state coding against occipital alpha; writes per-participant epochs to `work_dir` and an extraction log and exclusion list to `results_dir`.
2. `spectral.py`: Welch spectra for each eyes-closed and eyes-open minute of both sessions; band and regional power; individual alpha frequency.
3. `shift.py`: EEGNet trained, with subject-wise 10-fold cross-validation repeated with three fold assignments, to separate session 2 from session 1 eyes-closed epochs; no symptom score is used. Finished folds are saved in `work_dir` and reused if the run is interrupted.
4. `predict.py`: out-of-sample prediction of depression relative to anxiety from the spectral change (one band-power feature selected inside each training fold, ridge regression and a compact convolutional network on the full channel-by-frequency change), with a permutation test for the band model.
5. `analyse.py`: partial Spearman correlations, permutation tests (10,000) and bootstrap intervals (2,000), robustness and control analyses; writes `results summary.json`, `analysis table.csv` and `frequency resolved.csv`.
6. `figures.py`: Figures 1 to 3 (TIFF and PNG, 300 dpi).

All random seeds are fixed in `config.yaml`. Approximate run time on two CPU cores: extraction 10 minutes, spectra 2 minutes, session networks about 2.5 hours, prediction analysis about 30 minutes, statistics 1 minute.

## Participants

Participants marked invalid in the dataset metadata are excluded. Groups are defined by BDI (high: 13 or higher; low: 7 or lower); the dataset contains no scores between these cut-offs. Sex is coded 1 = female and 2 = male, as in `participants.json`.
