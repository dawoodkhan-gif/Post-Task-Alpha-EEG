"""Shared helpers: configuration, participant table, paths."""
import os
import yaml
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(HERE, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    for key in ('dataset_root', 'work_dir', 'results_dir'):
        p = cfg[key]
        cfg[key] = p if os.path.isabs(p) else os.path.normpath(os.path.join(HERE, p))
    os.makedirs(cfg['work_dir'], exist_ok=True)
    os.makedirs(cfg['results_dir'], exist_ok=True)
    return cfg


def participants(cfg):
    """Participant table with group labels read from the metadata column named in config.

    Returns included participants and a table of exclusions with reasons.
    """
    p = cfg['participants']
    d = pd.read_csv(os.path.join(cfg['dataset_root'], 'participants.tsv'), sep='\t')
    assert len(d) == 122, f'expected 122 rows in participants.tsv, found {len(d)}'
    invalid = d[p['invalid_column']].astype(str).str.strip() == p['invalid_value']
    missing = d[p['group_column']].isna()
    excluded = d[invalid | missing][['participant_id', 'Original_ID']].copy()
    excluded['reason'] = np.where(invalid[invalid | missing], 'marked invalid in metadata', 'no BDI score')
    d = d[~(invalid | missing)].copy()
    bdi = d[p['group_column']]
    high = bdi >= p['high_min']
    low = bdi <= p['low_max']
    assert (high | low).all(), 'participants with BDI between the group cut-offs found'
    d['group'] = high.astype(int)
    return d.reset_index(drop=True), excluded.reset_index(drop=True)


def epochs_path(cfg, sub):
    return os.path.join(cfg['work_dir'], f'{sub}.npz')
