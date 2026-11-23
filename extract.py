"""Extraction from the raw EEGLAB files of ds003478.

For every included participant and run: drop EOG/EKG, average reference, 1 to 40 Hz band pass,
resample, cut non-overlapping epochs inside each one-minute eyes-closed or eyes-open block
(trigger codes 1 to 6), reject epochs by amplitude, and save the arrays.
Also checks the eye-state coding of every run against occipital alpha power.
"""
import os
import sys
import glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import mne
from common import load_config, participants, epochs_path

mne.set_log_level('ERROR')


def block_spans(events_file, codes, spacing):
    ev = pd.read_csv(events_file, sep='\t')
    val = pd.to_numeric(ev['value'], errors='coerce')
    spans = []
    for c in codes:
        on = np.sort(ev.loc[val == c, 'onset'].astype(float).values)
        if len(on) < 2:
            continue
        # a block is the longest run of triggers without a gap over 4 spacings; isolated stray triggers are ignored
        cuts = np.where(np.diff(on) > 4 * spacing)[0] + 1
        runs = np.split(on, cuts)
        best = max(runs, key=len)
        if len(best) < 2:
            continue
        spans.append((c, best[0], best[-1] + spacing))
    return spans


def process_run(set_file, cfg):
    pp = cfg['preprocessing']
    raw = mne.io.read_raw_eeglab(set_file, preload=True)
    raw.drop_channels([c for c in pp['drop_channels'] if c in raw.ch_names])
    assert len(raw.ch_names) == 64, f'{set_file}: {len(raw.ch_names)} EEG channels'
    raw.set_eeg_reference('average', projection=False)
    raw.filter(pp['l_freq'], pp['h_freq'], fir_design='firwin')
    # occipital alpha check on the filtered, unresampled signal
    spans = block_spans(set_file.replace('_eeg.set', '_events.tsv'), pp['block_codes'], pp['trigger_spacing_seconds'])
    alpha = {}
    occ = raw.copy().pick(pp['alpha_channels'])
    for c, t0, t1 in spans:
        x = occ.get_data(tmin=t0, tmax=min(t1, occ.times[-1]))
        psd, fr = mne.time_frequency.psd_array_welch(x, raw.info['sfreq'], fmin=1, fmax=40, n_fft=int(raw.info['sfreq'] * 2))
        band = (fr >= pp['alpha_band'][0]) & (fr <= pp['alpha_band'][1])
        alpha[c] = psd[:, band].mean() / psd.mean()
    raw.resample(pp['sfreq'])
    sf = raw.info['sfreq']
    n = int(pp['epoch_seconds'] * sf)
    data = raw.get_data() * 1e6  # volts to microvolts
    X, eye, block, rej = [], [], [], 0
    for c, t0, t1 in spans:
        a, b = int(round(t0 * sf)), min(int(round(t1 * sf)), data.shape[1])
        for s in range(a, b - n + 1, n):
            seg = data[:, s:s + n]
            ptp = seg.max(1) - seg.min(1)
            if ptp.max() > pp['reject_peak_to_peak_uv'] or ptp.min() < pp['flat_uv']:
                rej += 1
                continue
            X.append(seg.astype(np.float32))
            eye.append('closed' if c % 2 == 1 else 'open')
            block.append(c)
    closed = np.mean([v for k, v in alpha.items() if k % 2 == 1]) if alpha else np.nan
    opened = np.mean([v for k, v in alpha.items() if k % 2 == 0]) if alpha else np.nan
    info = dict(blocks=len(spans), epochs_kept=len(X), epochs_rejected=rej,
                alpha_closed=closed, alpha_open=opened, alpha_ratio=closed / opened)
    X = np.stack(X) if X else np.zeros((0, 64, n), np.float32)
    return X, np.array(eye), np.array(block), list(raw.ch_names), info


def main():
    cfg = load_config()
    subs, excluded = participants(cfg)
    excluded.to_csv(os.path.join(cfg['results_dir'], 'excluded participants.csv'), index=False)
    log = []
    channels = None
    for sub in subs['participant_id']:
        out = {}
        for f in sorted(glob.glob(os.path.join(cfg['dataset_root'], sub, 'eeg', '*_eeg.set'))):
            run = 'run' + f.split('run-')[1][:2].lstrip('0')
            X, eye, block, chs, info = process_run(f, cfg)
            if channels is None:
                channels = chs
            assert chs == channels, f'{f}: channel order differs'
            out[f'X_{run}'], out[f'eye_{run}'], out[f'block_{run}'] = X, eye, block
            log.append(dict(participant_id=sub, run=run, **info))
            print(sub, run, info, flush=True)
        np.savez(epochs_path(cfg, sub), channels=np.array(channels), **out)
    log = pd.DataFrame(log)
    log['eye_coding_flag'] = log['alpha_ratio'] < 1.0
    log.to_csv(os.path.join(cfg['results_dir'], 'extraction log.csv'), index=False)
    print('runs', len(log), 'flagged eye coding', int(log.eye_coding_flag.sum()))


if __name__ == '__main__':
    main()
