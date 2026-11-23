#!/usr/bin/env bash
# Reproduces every result from the raw dataset. Paths and settings: config.yaml.
set -euo pipefail
cd "$(dirname "$0")"
python3 -I extract.py      # 1. preprocessing and epoching of both resting sessions
python3 -I spectral.py     # 2. power spectra and change measures
python3 -I shift.py        # 3. deep networks trained to separate the sessions (subject-wise)
python3 -I predict.py      # 4. out-of-sample prediction from the spectral change
python3 -I analyse.py      # 5. statistics
python3 -I figures.py      # 6. figures
