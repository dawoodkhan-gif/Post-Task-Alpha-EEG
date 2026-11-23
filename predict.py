"""Out-of-sample prediction of depression relative to anxiety from the post-task spectral change.

Target: the contrast between rank-standardised BDI and STAI (computed within the analysed sample).
Input: each participant's change in log10 power (session 2 minus session 1, first eyes-closed minute),
64 channels by 1 to 40 Hz. Models, all evaluated with repeated subject-wise 10-fold cross-validation:
  band:  one band-power change feature, the band chosen inside each training fold as the band whose change
         correlates most strongly (absolute Spearman rho) with the target in the training participants;
  ridge: ridge regression on the full channel-by-frequency change (penalty chosen by inner cross-validation);
  cnn:   a compact convolutional network (spatial filters, then spectral convolution) on the full change,
         an ensemble of five seeds per fold.
Writes Results/prediction.csv (per-participant predictions) and Results/prediction summary.json.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, rankdata
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, participants

def zr(v):
    r = rankdata(v)
    return (r - r.mean()) / r.std()


class SpectralNet(nn.Module):
    def __init__(self, n_channels, n_freqs, k=4):
        super().__init__()
        self.spat = nn.Conv1d(n_channels, k, 1)
        self.spec = nn.Conv1d(k, k, 9, padding=4, groups=k)
        self.pool = nn.AdaptiveAvgPool1d(8)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(k * 8, 1)

    def forward(self, x):
        z = torch.tanh(self.spec(self.spat(x)))
        return self.out(self.drop(torch.flatten(self.pool(z), 1))).squeeze(1)


def cnn_predict(Xtr, ytr, Xte, seeds, epochs):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    a, b, t = torch.tensor((Xtr - mu) / sd), torch.tensor(ytr), torch.tensor((Xte - mu) / sd)
    preds = []
    for s in seeds:
        torch.manual_seed(s)
        m = SpectralNet(Xtr.shape[1], Xtr.shape[2])
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-2)
        for _ in range(epochs):
            m.train()
            opt.zero_grad()
            loss = ((m(a) - b) ** 2).mean()
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            preds.append(m(t).numpy())
    return np.mean(preds, 0)


def band_predict(B, ytr_idx, yte_idx, y):
    rs = [abs(spearmanr(B[ytr_idx, j], y[ytr_idx])[0]) for j in range(B.shape[1])]
    j = int(np.argmax(rs))
    k = np.polyfit(B[ytr_idx, j], y[ytr_idx], 1)
    return np.polyval(k, B[yte_idx, j]), j


def main():
    cfg = load_config()
    pc = cfg['prediction']
    torch.set_num_threads(cfg['training']['threads'])
    res = cfg['results_dir']
    P, _ = participants(cfg)
    P = P.set_index('participant_id')
    Z = np.load(os.path.join(res, 'spectra.npz'))
    idx = {k: i for i, k in enumerate(Z['keys'])}
    fr = Z['freqs']
    subs = [s for s in P.index if f'{s}|EC1first' in idx and f'{s}|EC2first' in idx]
    X = np.stack([Z['spectra'][idx[f'{s}|EC2first']] - Z['spectra'][idx[f'{s}|EC1first']] for s in subs]).astype(np.float32)
    y = (zr(P.loc[subs, 'BDI'].values) - zr(P.loc[subs, 'STAI'].values)).astype(np.float32)
    bands = cfg['spectral']['bands']
    names = list(bands)
    B = np.stack([X[:, :, (fr >= lo) & (fr < hi)].mean((1, 2)) for lo, hi in bands.values()], 1)
    n = len(y)
    rows, chosen = [], []
    for rep in range(pc['repeats']):
        pr = {m: np.zeros(n) for m in ('band', 'ridge', 'cnn')}
        for k, (tr, te) in enumerate(KFold(10, shuffle=True, random_state=cfg['training']['seed'] + rep).split(X)):
            pr['band'][te], j = band_predict(B, tr, te, y)
            chosen.append(names[j])
            Xf = X.reshape(n, -1)
            mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-6
            r = RidgeCV(alphas=np.logspace(0, 6, 25)).fit((Xf[tr] - mu) / sd, y[tr])
            pr['ridge'][te] = r.predict((Xf[te] - mu) / sd)
            pr['cnn'][te] = cnn_predict(X[tr], y[tr], X[te], [cfg['training']['seed'] + s for s in range(pc['cnn_seeds'])], pc['cnn_epochs'])
        for m, v in pr.items():
            rows += [{'participant_id': s, 'repeat': rep, 'model': m, 'prediction': float(p), 'target': float(t)} for s, p, t in zip(subs, v, y)]
        print('repeat', rep, {m: round(spearmanr(v, y)[0], 3) for m, v in pr.items()}, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(res, 'prediction.csv'), index=False)
    rng = np.random.RandomState(cfg['analysis']['seed'])
    out = {'n': n, 'band_chosen': {b: chosen.count(b) for b in set(chosen)}, 'models': {}}
    for m in ('band', 'ridge', 'cnn'):
        per_rep = [spearmanr(g.prediction, g.target)[0] for _, g in D[D.model == m].groupby('repeat')]
        avg = D[D.model == m].groupby('participant_id')[['prediction', 'target']].mean()
        r = spearmanr(avg.prediction, avg.target)[0]
        bs = []
        for _ in range(cfg['analysis']['bootstrap']):
            i = rng.randint(0, len(avg), len(avg))
            bs.append(spearmanr(avg.prediction.values[i], avg.target.values[i])[0])
        out['models'][m] = {'rho_mean_of_repeats': float(np.mean(per_rep)), 'rho_repeats': [float(v) for v in per_rep],
                            'rho_averaged': float(r), 'ci': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    # permutation test for the band model: the whole selection and fitting is repeated with shuffled targets
    obs = out['models']['band']['rho_repeats'][0]   # same fold assignment as the permutations
    null = []
    for i in range(pc['permutations']):
        yp = rng.permutation(y)
        p = np.zeros(n)
        for tr, te in KFold(10, shuffle=True, random_state=cfg['training']['seed']).split(X):
            p[te], _ = band_predict(B, tr, te, yp)
        null.append(spearmanr(p, yp)[0])
    out['models']['band']['p_perm'] = float((1 + np.sum(np.array(null) >= obs)) / (len(null) + 1))
    out['models']['band']['null_95'] = float(np.percentile(null, 95))
    json.dump(out, open(os.path.join(res, 'prediction summary.json'), 'w'), indent=1)
    print(json.dumps(out['models'], indent=1), out['band_chosen'])


if __name__ == '__main__':
    main()
