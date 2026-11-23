"""Figures for the manuscript, drawn from Results/analysis table.csv, Results/frequency resolved.csv
and Results/results summary.json. Saved as TIFF and PNG at 300 dpi in Results/figures."""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
HIGH, LOW, DEP, ANX = '#B03A2E', '#2E86C1', '#B03A2E', '#2E86C1'


def save(fig, out, name):
    for ext in ('tiff', 'png'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=300, bbox_inches='tight',
                    **({'pil_kwargs': {'compression': 'tiff_lzw'}} if ext == 'tiff' else {}))
    plt.close(fig)


def label(ax, s):
    ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontweight='bold', fontsize=11)


def main():
    cfg = load_config()
    res = cfg['results_dir']
    out = os.path.join(res, 'figures')
    os.makedirs(out, exist_ok=True)
    S = pd.read_csv(os.path.join(res, 'analysis table.csv'))
    F = pd.read_csv(os.path.join(res, 'frequency resolved.csv'))
    J = json.load(open(os.path.join(res, 'results summary.json')))

    # Figure 1
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [1, 1.15]})
    for g, c, n in ((0, LOW, 'Low BDI'), (1, HIGH, 'High BDI')):
        d = S[S.group == g]
        ax[0].scatter(d.depression_over_anxiety, d.d_first_alpha, s=14, c=c, alpha=0.75, label=n, edgecolors='none')
    b = np.polyfit(S.depression_over_anxiety, S.d_first_alpha, 1)
    xs = np.linspace(S.depression_over_anxiety.min(), S.depression_over_anxiety.max(), 10)
    ax[0].plot(xs, np.polyval(b, xs), 'k-', lw=1)
    ax[0].axhline(0, color='grey', lw=0.6, ls=':')
    ax[0].set_xlabel('Depression relative to anxiety\n(rank z BDI minus rank z STAI)')
    ax[0].set_ylabel('Change in eyes-closed alpha power\n(session 2 minus session 1, log$_{10}$)')
    ax[0].legend(frameon=False, fontsize=8, loc='upper right')
    label(ax[0], 'A')
    ax[1].axvspan(8, 13, color='0.92')
    ax[1].plot(F.freq, F.bdi_given_stai, color=DEP, lw=1.5, label='BDI controlling for STAI')
    ax[1].plot(F.freq, F.stai_given_bdi, color=ANX, lw=1.5, ls='--', label='STAI controlling for BDI')
    n = J['primary']['bdi_given_stai']['n']
    crit = stats.t.ppf(0.975, n - 3)
    r = crit / np.sqrt(n - 3 + crit ** 2)
    for v in (r, -r):
        ax[1].axhline(v, color='grey', lw=0.6, ls=':')
    ax[1].axhline(0, color='k', lw=0.6)
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Partial Spearman rho with power change')
    ax[1].set_xlim(1, 40)
    ax[1].legend(frameon=False, fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=1)
    label(ax[1], 'B')
    fig.tight_layout()
    save(fig, out, 'Figure 1')

    # Figure 2: robustness
    rows = [('Primary analysis', J['primary']),
            ('Adjusted for session 1 alpha', J['primary_baseline']),
            ('Adjusted for theta and alpha frequency change', J['controls']['theta_iaf']),
            ('Adjusted for delta and theta change', J['controls']['delta_theta']),
            ('Adjusted for broadband power change', J['controls']['total_power']),
            ('Adjusted for epoch counts', J['controls']['epochs']),
            ('All eyes-closed epochs', J['sensitivity']['all_epochs']),
            ('Relative alpha', J['sensitivity']['relative_alpha']),
            ('Without flagged recordings', J['sensitivity']['without_flagged']),
            ('Women only', J['sensitivity']['women']),
            ('Men only', J['sensitivity']['men']),
            ('Eyes-open alpha (control)', J['controls']['eyes_open_alpha'])]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    y = np.arange(len(rows))[::-1]
    for k, (key, c, m, off) in enumerate((('bdi_given_stai', DEP, 'o', 0.15), ('stai_given_bdi', ANX, 's', -0.15))):
        rr = np.array([v[key]['rho'] for _, v in rows])
        se = 1 / np.sqrt(np.array([v[key]['df'] for _, v in rows]) - 1)
        lo, hi = np.tanh(np.arctanh(rr) - 1.96 * se), np.tanh(np.arctanh(rr) + 1.96 * se)
        ax.errorbar(rr, y + off, xerr=[rr - lo, hi - rr], fmt=m, color=c, ms=4, capsize=2, lw=1,
                    label='BDI controlling for STAI' if k == 0 else 'STAI controlling for BDI')
    ax.axvline(0, color='k', lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel('Partial Spearman rho with change in eyes-closed alpha power (95% CI)')
    ax.legend(frameon=False, fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()
    save(fig, out, 'Figure 2')

    # Figure 3: deep networks
    dl = J.get('deep', {})
    if dl:
        T = []
        for m in dl:
            for f in sorted(os.listdir(os.path.join(res, 'shift'))):
                if f.startswith(m + ' repeat'):
                    T.append(pd.read_csv(os.path.join(res, 'shift', f)))
        T = pd.concat(T).groupby(['model', 'participant_id'])[['session_auc', 'shift_index']].mean().reset_index()
        fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7), gridspec_kw={'width_ratios': [0.8, 1, 1.1]})
        ms = list(dl)
        ax[0].boxplot([T.loc[T.model == m, 'session_auc'] for m in ms], widths=0.5, showfliers=False)
        for i, m in enumerate(ms):
            v = T.loc[T.model == m, 'session_auc']
            ax[0].scatter(np.random.RandomState(0).normal(i + 1, 0.05, len(v)), v, s=5, c='0.4', alpha=0.6)
        ax[0].axhline(0.5, color='grey', ls=':', lw=0.8)
        ax[0].set_xticks(range(1, len(ms) + 1))
        ax[0].set_xticklabels(ms, fontsize=8)
        ax[0].set_ylabel('Session 2 vs session 1 AUC\n(held-out participant)')
        label(ax[0], 'A')
        m0 = ms[0]
        col = f'shift_{m0}'
        for g, c, n in ((0, LOW, 'Low BDI'), (1, HIGH, 'High BDI')):
            d = S[(S.group == g) & S[col].notna()]
            ax[1].scatter(d.depression_over_anxiety, d[col], s=12, c=c, alpha=0.75, label=n, edgecolors='none')
        d = S[S[col].notna()]
        b = np.polyfit(d.depression_over_anxiety, d[col], 1)
        ax[1].plot(xs, np.polyval(b, xs), 'k-', lw=1)
        ax[1].set_xlabel('Depression relative to anxiety')
        ax[1].set_ylabel(f'{m0} post-task shift index')
        ax[1].legend(frameon=False, fontsize=7)
        label(ax[1], 'B')
        pr = J.get('prediction', {}).get('models', {})
        labs = [('band', 'Band\n1 feature'), ('ridge', 'Ridge\nfull'), ('cnn', 'CNN\nfull')]
        for i, (k, lab) in enumerate(labs):
            if k not in pr:
                continue
            v = pr[k]
            ax[2].bar(i, v['rho_averaged'], color=['0.3', '0.6', '0.6'][i], width=0.6)
            ax[2].errorbar(i, v['rho_averaged'], yerr=[[v['rho_averaged'] - v['ci'][0]], [v['ci'][1] - v['rho_averaged']]], color='k', capsize=3, lw=1)
        if 'band' in pr:
            ax[2].axhline(pr['band']['null_95'], color='grey', ls=':', lw=0.8)
        ax[2].axhline(0, color='k', lw=0.6)
        ax[2].set_xticks(range(len(labs)))
        ax[2].set_xticklabels([l for _, l in labs], fontsize=7.5)
        ax[2].set_ylabel('Out-of-sample rho with\ndepression relative to anxiety')
        label(ax[2], 'C')
        fig.tight_layout()
        save(fig, out, 'Figure 3')
    print('figures written to', out)


if __name__ == '__main__':
    main()
