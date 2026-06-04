"""
thunderbird_features.py

Extracts feature vector per Thunderbird window session.
"""

import re
import numpy as np
import pandas as pd

# Thunderbird-specific severity/component patterns
FATAL_PAT    = re.compile(r'\bfatal\b',   re.IGNORECASE)
ERROR_PAT    = re.compile(r'\b(error|fail|fault|failed|cannot|denied|refused)\b', re.IGNORECASE)
WARN_PAT     = re.compile(r'\b(warn|warning)\b', re.IGNORECASE)
LINK_PAT     = re.compile(r'\blink\b',    re.IGNORECASE)
PROC_PAT     = re.compile(r'\b(crond|sshd|ntpd|sendmail)\b', re.IGNORECASE)
TIMEOUT_PAT  = re.compile(r'\b(timeout|not answer|lost)\b', re.IGNORECASE)
SWITCH_PAT   = re.compile(r'\b(ib_sm|ganglia|gmetad)\b', re.IGNORECASE)
MEM_PAT      = re.compile(r'\b(memory|ecc|parity)\b', re.IGNORECASE)


def extract_window_features(win_id: str, events: list) -> dict:
    n = max(len(events), 1)

    fatal_count   = sum(1 for e in events if FATAL_PAT.search(e))
    error_count   = sum(1 for e in events if ERROR_PAT.search(e))
    warn_count    = sum(1 for e in events if WARN_PAT.search(e))
    link_count    = sum(1 for e in events if LINK_PAT.search(e))
    proc_count    = sum(1 for e in events if PROC_PAT.search(e))
    timeout_count = sum(1 for e in events if TIMEOUT_PAT.search(e))
    switch_count  = sum(1 for e in events if SWITCH_PAT.search(e))
    mem_count     = sum(1 for e in events if MEM_PAT.search(e))

    error_ratio   = error_count / n
    unique_ratio  = len(set(events)) / n

    return {
        'win_id':         win_id,
        'seq_len':        n,
        'fatal_count':    fatal_count,
        'error_count':    error_count,
        'warn_count':     warn_count,
        'link_count':     link_count,
        'proc_count':     proc_count,
        'timeout_count':  timeout_count,
        'switch_count':   switch_count,
        'mem_count':      mem_count,
        'error_ratio':    error_ratio,
        'unique_ratio':   unique_ratio,
    }


def build_feature_matrix(windows: dict) -> pd.DataFrame:
    print(f"[Thunderbird Features] Extracting features for {len(windows):,} windows...")
    rows = []
    for i, (win_id, w) in enumerate(windows.items()):
        feat = extract_window_features(win_id, w['events'])
        feat['label'] = w['label']
        rows.append(feat)
        if (i + 1) % 50_000 == 0:
            print(f"  Processed {i+1:,} windows...")

    df = pd.DataFrame(rows)
    print(f"[Thunderbird Features] Matrix: {df.shape[0]:,} windows × {df.shape[1]} cols")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in {'win_id', 'label'}
            and df[c].dtype in [np.float64, np.int64, int, float]]
