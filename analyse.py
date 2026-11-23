"""Statistics for the post-task change in resting EEG and its relation to depressive and anxious symptoms.

Reads Results/change measures.csv, Results/spectra.npz and Results/shift/*.csv.
Writes Results/results summary.json, Results/frequency resolved.csv and the table CSV files.
Partial Spearman correlations are Pearson correlations between rank residuals after regressing
the ranks of both variables on the ranks of the covariates.
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, participants

R = stats.rankdata


def resid(v, C):
    v = R(v)
    if C is None or len(C) == 0:
        return v - v.mean()
    A = np.column_stack([np.ones(len(v))] + [R(c) for c in C])
    return v - A @ np.linalg.lstsq(A, v, rcond=None)[0]


def pcor(x, y, C=(), rng=None, nperm=0, nboot=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    C = [np.asarray(c, float) for c in C]
    ok = np.isfinite(x) & np.isfinite(y) & np.all([np.isfinite(c) for c in C] or [np.ones(len(x), bool)], axis=0)
    x, y, C = x[ok], y[ok], [c[ok] for c in C]
    rx, ry = resid(x, C), resid(y, C)
    r = float(np.corrcoef(rx, ry)[0, 1])
    n, k = len(x), len(C)
    df = n - 2 - k
    t = r * np.sqrt(df / max(1e-12, 1 - r * r))
    out = {'rho': r, 'n': int(n), 'df': int(df), 'p': float(2 * stats.t.sf(abs(t), df))}
    if nperm:
        null = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(nperm)])
        out['p_perm'] = float((1 + (np.abs(null) >= abs(r)).sum()) / (nperm + 1))
    if nboot:
        bs = []
        for _ in range(nboot):
            i = rng.randint(0, n, n)
            bx, by = resid(x[i], [c[i] for c in C]), resid(y[i], [c[i] for c in C])
            bs.append(np.corrcoef(bx, by)[0, 1])
        out['ci'] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    return out


def both(df, ch, rng, C=(), **kw):
    """Change measure against BDI controlling STAI, and against STAI controlling BDI."""
    return {'bdi_given_stai': pcor(df[ch], df.BDI, [df.STAI] + [df[c] for c in C], rng, **kw),
            'stai_given_bdi': pcor(df[ch], df.STAI, [df.BDI] + [df[c] for c in C], rng, **kw),
            'bdi': pcor(df[ch], df.BDI, [df[c] for c in C], rng),
            'stai': pcor(df[ch], df.STAI, [df[c] for c in C], rng)}


def std_ols(df, y, X, rng, nboot):
    """OLS on ranks (all variables rank transformed and standardised); bootstrap CI for coefficients."""
    def fit(d):
        Z = lambda v: (R(v) - R(v).mean()) / R(v).std()
        A = np.column_stack([np.ones(len(d))] + [Z(d[c]) for c in X])
        return np.linalg.lstsq(A, Z(d[y]), rcond=None)[0][1:]
    d = df[[y] + X].dropna()
    b = fit(d)
    bs = np.array([fit(d.iloc[rng.randint(0, len(d), len(d))]) for _ in range(nboot)])
    A = np.column_stack([np.ones(len(d))] + [(R(d[c]) - R(d[c]).mean()) / R(d[c]).std() for c in X])
    yy = (R(d[y]) - R(d[y]).mean()) / R(d[y]).std()
    res = yy - A @ np.r_[0, b]
    s2 = res @ res / (len(d) - A.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A.T @ A)))[1:]
    p = 2 * stats.t.sf(np.abs(b / se), len(d) - A.shape[1])
    return {'n': int(len(d)), 'terms': {c: {'beta': float(b[i]), 'p': float(p[i]),
            'ci': [float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))]} for i, c in enumerate(X)}}


def holm(ps):
    ps = np.asarray(ps)
    o = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1, run)
    return adj.tolist()


def main():
    cfg = load_config()
    a = cfg['analysis']
    rng = np.random.RandomState(a['seed'])
    res_dir = cfg['results_dir']
    P, _ = participants(cfg)
    D = pd.read_csv(os.path.join(res_dir, 'change measures.csv')).merge(
        P[['participant_id', 'BDI', 'STAI', 'age', 'sex', 'group', 'SCID', 'HamD']], on='participant_id')
    import warnings
    warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
    bands = list(cfg['spectral']['bands'])
    for key in ('first', 'all'):
        for b in bands + ['total', 'iaf']:
            D[f'd_{key}_{b}'] = D[f'EC2{key}_{b}'] - D[f'EC1{key}_{b}']
        D[f'd_{key}_relalpha'] = D[f'd_{key}_alpha'] - D[f'd_{key}_total']
        for rn in cfg['spectral']['regions']:
            D[f'd_{key}_alpha_{rn}'] = D[f'EC2{key}_alpha_{rn}'] - D[f'EC1{key}_alpha_{rn}']
    D['d_EO_alpha'] = D['EO2first_alpha'] - D['EO1first_alpha']
    D['d_theta_alpha_ratio'] = (D['EC2first_theta'] - D['EC2first_alpha']) - (D['EC1first_theta'] - D['EC1first_alpha'])
    D['d_epochs'] = D['EC2first_epochs'] - D['EC1first_epochs']
    S = D.dropna(subset=['d_first_alpha']).copy()
    out = {}

    # sample
    out['sample'] = {'included': int(len(D)), 'both_sessions': int(len(S)), 'high': int(S.group.sum()),
                     'low': int((S.group == 0).sum()), 'women': int((S.sex == 1).sum()),
                     'epochs_s1_median': float(S.EC1first_epochs.median()), 'epochs_s2_median': float(S.EC2first_epochs.median()),
                     'rho_bdi_stai': float(stats.spearmanr(S.BDI, S.STAI)[0])}

    demo = []
    for g, name in ((0, 'Low BDI'), (1, 'High BDI')):
        d = S[S.group == g]
        sc = d.SCID.astype(str)
        demo.append({'group': name, 'n': int(len(d)), 'female': int((d.sex == 1).sum()), 'male': int((d.sex == 2).sum()),
                     'age_mean': float(d.age.mean()), 'age_sd': float(d.age.std()), 'age_missing': int(d.age.isna().sum()),
                     'bdi_mean': float(d.BDI.mean()), 'bdi_sd': float(d.BDI.std()), 'bdi_min': float(d.BDI.min()), 'bdi_max': float(d.BDI.max()),
                     'stai_mean': float(d.STAI.mean()), 'stai_sd': float(d.STAI.std()),
                     'interviewed': int((~sc.str.contains('No Interview')).sum()),
                     'scid_values': sc.value_counts().to_dict(),
                     'alpha_s1_mean': float(d.EC1first_alpha.mean()), 'alpha_s1_sd': float(d.EC1first_alpha.std()),
                     'alpha_s2_mean': float(d.EC2first_alpha.mean()), 'alpha_s2_sd': float(d.EC2first_alpha.std()),
                     'change_mean': float(d.d_first_alpha.mean()), 'change_sd': float(d.d_first_alpha.std())})
    out['demographics'] = demo
    # internal (split-half) reliability of the alpha measures, Spearman-Brown corrected
    sb = lambda r: 2 * r / (1 + r)
    rel = {}
    for k, (o, e) in {'session1': ('EC1first_alpha_odd', 'EC1first_alpha_even'), 'session2': ('EC2first_alpha_odd', 'EC2first_alpha_even')}.items():
        r = stats.pearsonr(S[o], S[e])[0]
        rel[k] = float(sb(r))
    r = stats.pearsonr(S.EC2first_alpha_odd - S.EC1first_alpha_odd, S.EC2first_alpha_even - S.EC1first_alpha_even)[0]
    rel['change'] = float(sb(r))
    out['reliability'] = rel
    # sensitivity of the sample: smallest correlation detectable with 80% power, two-sided alpha 0.05
    n = len(S)
    zc = stats.norm.ppf(0.975) + stats.norm.ppf(0.80)
    out['sensitivity_r80'] = float(np.tanh(zc / np.sqrt(n - 4)))   # one covariate partialled out
    # within-person change and group comparison
    w = stats.wilcoxon(S.d_first_alpha)
    mw = stats.mannwhitneyu(S.loc[S.group == 1, 'd_first_alpha'], S.loc[S.group == 0, 'd_first_alpha'])
    out['change'] = {'alpha_mean': float(S.d_first_alpha.mean()), 'alpha_sd': float(S.d_first_alpha.std()),
                     'wilcoxon_W': float(w.statistic), 'wilcoxon_p': float(w.pvalue),
                     'high_mean': float(S.loc[S.group == 1, 'd_first_alpha'].mean()),
                     'low_mean': float(S.loc[S.group == 0, 'd_first_alpha'].mean()),
                     'group_U': float(mw.statistic), 'group_p': float(mw.pvalue),
                     'group_rank_biserial': float(1 - 2 * mw.statistic / (S.group.sum() * (S.group == 0).sum())),
                     'abs_change_bdi': pcor(S.d_first_alpha.abs(), S.BDI, [S.STAI]),
                     }

    # primary analysis
    out['primary'] = both(S, 'd_first_alpha', rng, nperm=a['permutations'], nboot=a['bootstrap'])
    zr = lambda v: (R(v) - R(v).mean()) / R(v).std()
    S['distress_shared'] = (zr(S.BDI) + zr(S.STAI)) / 2
    S['depression_over_anxiety'] = zr(S.BDI) - zr(S.STAI)
    out['components'] = {'shared': pcor(S.d_first_alpha, S.distress_shared, rng=rng, nperm=a['permutations'], nboot=a['bootstrap']),
                         'contrast': pcor(S.d_first_alpha, S.depression_over_anxiety, rng=rng, nperm=a['permutations'], nboot=a['bootstrap']),
                         'contrast_high': pcor(S.loc[S.group == 1, 'd_first_alpha'], S.loc[S.group == 1, 'depression_over_anxiety']),
                         'contrast_low': pcor(S.loc[S.group == 0, 'd_first_alpha'], S.loc[S.group == 0, 'depression_over_anxiety']),
                         'contrast_sd': float(S.depression_over_anxiety.std())}
    out['primary_baseline'] = both(S, 'd_first_alpha', rng, C=['EC1first_alpha'])
    S['female'] = (S.sex == 1).astype(float)  # dataset coding: 1 female, 2 male
    out['model_ranks'] = std_ols(S, 'd_first_alpha', ['BDI', 'STAI', 'age', 'female', 'EC1first_alpha'], rng, a['bootstrap'])

    S['d_anterior'] = (S.EC2first_alpha_frontal - S.EC2first_alpha_occipital) - (S.EC1first_alpha_frontal - S.EC1first_alpha_occipital)
    out['levels'] = {'session1': both(S, 'EC1first_alpha', rng), 'session2': both(S, 'EC2first_alpha', rng),
                     'anteriorisation_change': both(S, 'd_anterior', rng),
                     'anteriorisation_mean': float(S.d_anterior.mean()), 'anteriorisation_p': float(stats.wilcoxon(S.d_anterior).pvalue)}
    # band-wise
    bw = {b: both(S, f'd_first_{b}', rng) for b in bands}
    pb = holm([bw[b]['bdi_given_stai']['p'] for b in bands] + [bw[b]['stai_given_bdi']['p'] for b in bands])
    for i, b in enumerate(bands):
        bw[b]['bdi_given_stai']['p_holm'] = pb[i]
        bw[b]['stai_given_bdi']['p_holm'] = pb[i + len(bands)]
    out['bands'] = bw
    out['regions'] = {rn: both(S, f'd_first_alpha_{rn}', rng) for rn in cfg['spectral']['regions']}

    # vigilance and data-quality controls
    out['controls'] = {
        'theta_iaf': both(S, 'd_first_alpha', rng, C=['d_first_theta', 'd_first_iaf']),
        'delta_theta': both(S, 'd_first_alpha', rng, C=['d_first_delta', 'd_first_theta']),
        'epochs': both(S, 'd_first_alpha', rng, C=['EC1first_epochs', 'EC2first_epochs']),
        'total_power': both(S, 'd_first_alpha', rng, C=['d_first_total']),
        'iaf_change': both(S, 'd_first_iaf', rng),
        'theta_alpha_ratio': both(S, 'd_theta_alpha_ratio', rng),
        'eyes_open_alpha': both(S.dropna(subset=['d_EO_alpha']), 'd_EO_alpha', rng),
        'iaf_change_mean': float(S.d_first_iaf.mean()),
        'iaf_wilcoxon_p': float(stats.wilcoxon(S.d_first_iaf.dropna()).pvalue),
    }

    # sensitivity
    flagged = set()
    lp = os.path.join(res_dir, 'extraction log.csv')
    if os.path.exists(lp):
        L = pd.read_csv(lp)
        flagged = set(L.loc[L.eye_coding_flag.astype(bool), 'participant_id'])
    out['sensitivity'] = {
        'all_epochs': both(S.dropna(subset=['d_all_alpha']), 'd_all_alpha', rng),
        'relative_alpha': both(S, 'd_first_relalpha', rng),
        'without_flagged': both(S[~S.participant_id.isin(flagged)], 'd_first_alpha', rng),
        'n_flagged_participants': len(flagged & set(S.participant_id)),
        'women': both(S[S.sex == 1], 'd_first_alpha', rng),
        'men': both(S[S.sex == 2], 'd_first_alpha', rng),
        'within_high': {'bdi': pcor(S.loc[S.group == 1, 'd_first_alpha'], S.loc[S.group == 1, 'BDI']),
                        'stai': pcor(S.loc[S.group == 1, 'd_first_alpha'], S.loc[S.group == 1, 'STAI'])},
        'within_low': {'stai': pcor(S.loc[S.group == 0, 'd_first_alpha'], S.loc[S.group == 0, 'STAI'])},
    }
    H = S.dropna(subset=['HamD'])
    out['hamd'] = {'n': int(len(H)), 'rho': pcor(H.d_first_alpha, H.HamD) if len(H) > 5 else None,
                   'rho_given_stai': pcor(H.d_first_alpha, H.HamD, [H.STAI]) if len(H) > 5 else None}

    # frequency-resolved partial correlations (channel mean)
    Z = np.load(os.path.join(res_dir, 'spectra.npz'))
    keys = list(Z['keys'])
    idx = {k: i for i, k in enumerate(keys)}
    fr = Z['freqs']
    sub = [s for s in S.participant_id if f'{s}|EC1first' in idx and f'{s}|EC2first' in idx]
    Sd = S.set_index('participant_id').loc[sub]
    dm = np.array([Z['spectra'][idx[f'{s}|EC2first']].mean(0) - Z['spectra'][idx[f'{s}|EC1first']].mean(0) for s in sub])
    rows = []
    for j, f in enumerate(fr):
        x = dm[:, j]
        rows.append({'freq': float(f), 'mean_change': float(x.mean()),
                     'bdi_given_stai': pcor(x, Sd.BDI, [Sd.STAI])['rho'], 'stai_given_bdi': pcor(x, Sd.STAI, [Sd.BDI])['rho']})
    pd.DataFrame(rows).to_csv(os.path.join(res_dir, 'frequency resolved.csv'), index=False)

    # deep networks: learned post-task shift index
    dl = {}
    for m in cfg['training']['models']:
        files = sorted(glob.glob(os.path.join(res_dir, 'shift', f'{m} repeat *.csv')))
        if not files:
            continue
        T = pd.concat([pd.read_csv(f) for f in files])
        W = T.pivot_table(index='participant_id', columns='repeat', values='shift_index')
        A = T.groupby('participant_id')[['shift_index', 'session_auc']].mean()
        M = S.merge(A, left_on='participant_id', right_index=True)
        rep_r = W.corr(method='spearman').values[np.triu_indices(W.shape[1], 1)] if W.shape[1] > 1 else []
        dl[m] = {'n': int(len(M)), 'repeats': int(W.shape[1]),
                 'session_auc_mean': float(M.session_auc.mean()), 'session_auc_median': float(M.session_auc.median()),
                 'session_auc_above_half': float((M.session_auc > 0.5).mean()),
                 'shift_mean': float(M.shift_index.mean()), 'shift_wilcoxon_p': float(stats.wilcoxon(M.shift_index).pvalue),
                 'repeat_rho_min': float(np.min(rep_r)) if len(rep_r) else None,
                 'epochs_trained_median': float(T.epochs_trained.median()),
                 'symptoms': both(M, 'shift_index', rng, nperm=a['permutations'], nboot=a['bootstrap']),
                 'contrast': pcor(M.shift_index, M.depression_over_anxiety, rng=rng, nperm=a['permutations'], nboot=a['bootstrap']),
                 'shared': pcor(M.shift_index, M.distress_shared),
                 'symptoms_given_alpha': both(M, 'shift_index', rng, C=['d_first_alpha']),
                 'alpha_given_shift': both(M, 'd_first_alpha', rng, C=['shift_index']),
                 'rho_with_band_change': {b: float(stats.spearmanr(M.shift_index, M[f'd_first_{b}'])[0]) for b in bands + ['total', 'iaf']},
                 'group_p': float(stats.mannwhitneyu(M.loc[M.group == 1, 'shift_index'], M.loc[M.group == 0, 'shift_index']).pvalue)}
        S = S.merge(A[['shift_index']].rename(columns={'shift_index': f'shift_{m}'}), left_on='participant_id', right_index=True, how='left')
    out['deep'] = dl
    pp = os.path.join(res_dir, 'prediction summary.json')
    if os.path.exists(pp):
        out['prediction'] = json.load(open(pp))
    S.to_csv(os.path.join(res_dir, 'analysis table.csv'), index=False)
    json.dump(out, open(os.path.join(res_dir, 'results summary.json'), 'w'), indent=1)
    p = out['primary']
    print('n', p['bdi_given_stai']['n'], 'alpha change: BDI|STAI', round(p['bdi_given_stai']['rho'], 3), p['bdi_given_stai']['p_perm'],
          'STAI|BDI', round(p['stai_given_bdi']['rho'], 3), p['stai_given_bdi']['p_perm'])
    for m, v in dl.items():
        print(m, 'session AUC', round(v['session_auc_mean'], 3), 'BDI|STAI', round(v['symptoms']['bdi_given_stai']['rho'], 3),
              round(v['symptoms']['bdi_given_stai']['p'], 4), 'STAI|BDI', round(v['symptoms']['stai_given_bdi']['rho'], 3))


if __name__ == '__main__':
    main()
