"""Deep networks that learn the post-task change in resting EEG without seeing any symptom score.

Within each outer fold, a network is trained on the training participants to tell their
eyes-closed epochs from session 2 (after about one hour of tasks) from those of session 1
(immediately after electrode application). Each participant's epochs are scaled per channel by
that participant's own pooled standard deviation, so the network must learn a within-person
change rather than who the person is. For each held-out participant the shift index is the mean
logit of their session 2 epochs minus the mean logit of their session 1 epochs.
Usage: python shift.py [model ...]
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, participants, epochs_path
from models import EEGNet, ShallowConvNet

MODELS = {'EEGNet': EEGNet, 'ShallowConvNet': ShallowConvNet}


def first_closed(z, run):
    eye, blk = z[f'eye_{run}'], z[f'block_{run}']
    m = eye == 'closed'
    return z[f'X_{run}'][m & (blk == blk[m].min())]


def load(cfg, subs):
    data = {}
    for s in subs:
        z = np.load(epochs_path(cfg, s))
        if 'X_run2' not in z:
            continue
        a, b = first_closed(z, 'run1'), first_closed(z, 'run2')
        if len(a) < 5 or len(b) < 5:
            continue
        sd = np.concatenate([a, b]).std(axis=(0, 2), keepdims=True)[0]
        data[s] = ((a / sd).astype(np.float32), (b / sd).astype(np.float32))
    return data


def stack(data, subs):
    X = np.concatenate([np.concatenate(data[s]) for s in subs])
    y = np.concatenate([np.r_[np.zeros(len(data[s][0])), np.ones(len(data[s][1]))] for s in subs]).astype(np.float32)
    return X, y


def fit(model_name, Xtr, ytr, Xva, yva, t, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = MODELS[model_name](n_channels=Xtr.shape[1], n_times=Xtr.shape[2])
    opt = torch.optim.Adam(net.parameters(), lr=t['learning_rate'])
    lossf = nn.BCEWithLogitsLoss()
    Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    g = torch.Generator().manual_seed(seed)
    best, best_state, wait = np.inf, None, 0
    for ep in range(t['max_epochs']):
        net.train()
        order = torch.randperm(len(Xtr_t), generator=g)
        for i in range(0, len(order), t['batch_size']):
            b = order[i:i + t['batch_size']]
            opt.zero_grad()
            loss = lossf(net(Xtr_t[b])[0], ytr_t[b])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(torch.cat([net(Xva_t[i:i + 512])[0] for i in range(0, len(Xva_t), 512)]), yva_t).item()
        if vl < best - 1e-4:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= t['patience']:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, ep + 1


def logits(net, X):
    with torch.no_grad():
        return torch.cat([net(torch.from_numpy(X[i:i + 512]))[0] for i in range(0, len(X), 512)]).numpy()


def run(cfg, data, strat, model_name, repeat, out):
    t = cfg['training']
    subs = np.array(sorted(data))
    seed = t['seed'] + 1000 * repeat
    skf = StratifiedKFold(t['outer_folds'], shuffle=True, random_state=seed)
    rows = []
    for k, (tr, te) in enumerate(skf.split(subs, strat.loc[subs].values)):
        part = os.path.join(cfg['work_dir'], f'shift {model_name} repeat {repeat} fold {k}.csv')
        if os.path.exists(part):
            rows.append(pd.read_csv(part))
            continue
        t0 = time.time()
        rng = np.random.RandomState(seed + k)
        trs = subs[tr].copy()
        rng.shuffle(trs)
        nva = max(2, int(round(t['inner_validation_fraction'] * len(trs))))
        va, fit_s = trs[:nva], trs[nva:]
        net, n_ep = fit(model_name, *stack(data, fit_s), *stack(data, va), t, seed + k)
        fr = []
        for s in subs[te]:
            l1, l2 = logits(net, data[s][0]), logits(net, data[s][1])
            y = np.r_[np.zeros(len(l1)), np.ones(len(l2))]
            fr.append({'participant_id': s, 'model': model_name, 'repeat': repeat, 'fold': k,
                       'shift_index': float(l2.mean() - l1.mean()),
                       'session_auc': float(roc_auc_score(y, np.r_[l1, l2])),
                       'epochs_trained': n_ep})
        fr = pd.DataFrame(fr)
        fr.to_csv(part, index=False)
        rows.append(fr)
        print(f'{model_name} repeat {repeat} fold {k} epochs {n_ep} {time.time() - t0:.0f}s', flush=True)
    pd.concat(rows).to_csv(out, index=False)


def main():
    cfg = load_config()
    t = cfg['training']
    torch.set_num_threads(t['threads'])
    P, _ = participants(cfg)
    P = P.set_index('participant_id')
    data = load(cfg, P.index)
    strat = P['group']
    models = sys.argv[1:] or t['models']
    od = os.path.join(cfg['results_dir'], 'shift')
    os.makedirs(od, exist_ok=True)
    print('participants with both sessions', len(data), flush=True)
    for m in models:
        for r in t.get('model_repeats', {}).get(m, t['repeats']):
            out = os.path.join(od, f'{m} repeat {r}.csv')
            if not os.path.exists(out):
                run(cfg, data, strat, m, r, out)


if __name__ == '__main__':
    main()
