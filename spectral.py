"""Per-participant power spectra for each eyes-closed and eyes-open minute of both resting sessions.

Session 1 was recorded immediately after electrode application; session 2 about one hour later,
after task performance. The primary comparison uses the first eyes-closed minute of each session
so that the two estimates rest on the same amount of data.
Writes Results/spectra.npz and Results/change measures.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import welch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, participants, epochs_path


def spectrum(X, sf, nper):
    f, p = welch(X, fs=sf, nperseg=nper, axis=-1)
    return f, np.log10(p.mean(0))


def iaf(f, logp, idx, lo, hi):
    m = (f >= lo) & (f <= hi)
    p = (10 ** logp[idx][:, m]).mean(0)
    return float((f[m] * p).sum() / p.sum())   # centre of gravity of posterior alpha


def main():
    cfg = load_config()
    sp = cfg['spectral']
    sf = cfg['preprocessing']['sfreq']
    nper = int(sp['welch_seconds'] * sf)
    P, _ = participants(cfg)
    rows, spec = [], {}
    for sub in P.participant_id:
        z = np.load(epochs_path(cfg, sub))
        ch = [str(c) for c in z['channels']]
        reg = {k: [ch.index(c) for c in v] for k, v in sp['regions'].items()}
        post = [ch.index(c) for c in sp['iaf_channels']]
        r = {'participant_id': sub}
        for run, sess in (('run1', 1), ('run2', 2)):
            if f'X_{run}' not in z:
                continue
            X, eye, blk = z[f'X_{run}'], z[f'eye_{run}'], z[f'block_{run}']
            for state, tag in (('closed', 'EC'), ('open', 'EO')):
                m = eye == state
                if m.sum() == 0:
                    continue
                first = m & (blk == blk[m].min())
                for sel, name in ((first, 'first'), (m, 'all')):
                    if sel.sum() < 5:
                        continue
                    f, lp = spectrum(X[sel], sf, nper)
                    keep = (f >= sp['fmin']) & (f <= sp['fmax'])
                    key = f'{tag}{sess}{name}'
                    spec[f'{sub}|{key}'] = lp[:, keep]
                    r[f'{key}_epochs'] = int(sel.sum())
                    for b, (lo, hi) in sp['bands'].items():
                        bm = (f >= lo) & (f < hi)
                        r[f'{key}_{b}'] = float(lp[:, bm].mean())
                        for rn, idx in reg.items():
                            r[f'{key}_{b}_{rn}'] = float(lp[idx][:, bm].mean())
                    tot = (f >= sp['fmin']) & (f <= sp['fmax'])
                    r[f'{key}_total'] = float(lp[:, tot].mean())
                    r[f'{key}_iaf'] = iaf(f, lp, post, *sp['iaf_range'])
                    if name == 'first':   # split-half (odd and even epochs) for internal reliability
                        idx_sel = np.where(sel)[0]
                        for half, ii in (('odd', idx_sel[0::2]), ('even', idx_sel[1::2])):
                            fh, lh = spectrum(X[ii], sf, nper)
                            bm = (fh >= sp['bands']['alpha'][0]) & (fh < sp['bands']['alpha'][1])
                            r[f'{key}_alpha_{half}'] = float(lh[:, bm].mean())
        rows.append(r)
    D = pd.DataFrame(rows)
    out = cfg['results_dir']
    D.to_csv(os.path.join(out, 'change measures.csv'), index=False)
    freqs = f[(f >= sp['fmin']) & (f <= sp['fmax'])]
    np.savez_compressed(os.path.join(out, 'spectra.npz'), freqs=freqs, channels=np.array(ch),
                        keys=np.array(list(spec)), spectra=np.stack(list(spec.values())))
    print('participants', len(D), 'with session 2 eyes closed', D['EC2first_alpha'].notna().sum())


if __name__ == '__main__':
    main()
