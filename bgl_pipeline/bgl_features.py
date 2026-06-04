"""
bgl_features.py
Extracts feature vector per BGL window session.
"""

import re
import numpy as np
import pandas as pd

# BGL-specific error/severity patterns
FATAL_PAT    = re.compile(r'\bFATAL\b',    re.IGNORECASE)
ERROR_PAT    = re.compile(r'\b(error|fail|fault|exception|corrupt|crash|abort)\b', re.IGNORECASE)
WARN_PAT     = re.compile(r'\b(warn|WARNING|SEVERE)\b', re.IGNORECASE)
KERN_PAT     = re.compile(r'\bKERNEL\b',  re.IGNORECASE)
APP_PAT      = re.compile(r'\bAPP\b',     re.IGNORECASE)
CIOD_PAT     = re.compile(r'\bciod\b',    re.IGNORECASE)
TIMEOUT_PAT  = re.compile(r'\btimeout\b', re.IGNORECASE)
MEM_PAT      = re.compile(r'\b(memory|ddr|parity|ecc|corrected)\b', re.IGNORECASE)
FATAL_TYPES  = re.compile(r'\b(TLB|storage\s+interrupt|machine\s+check|program\s+interrupt)\b', re.IGNORECASE)


def extract_window_features(win_id: str, events: list) -> dict:
    n = max(len(events), 1)
    joined = ' '.join(events)

    fatal_count   = sum(1 for e in events if FATAL_PAT.search(e))
    error_count   = sum(1 for e in events if ERROR_PAT.search(e))
    warn_count    = sum(1 for e in events if WARN_PAT.search(e))
    kern_count    = sum(1 for e in events if KERN_PAT.search(e))
    app_count     = sum(1 for e in events if APP_PAT.search(e))
    ciod_count    = sum(1 for e in events if CIOD_PAT.search(e))
    timeout_count = sum(1 for e in events if TIMEOUT_PAT.search(e))
    mem_count     = sum(1 for e in events if MEM_PAT.search(e))
    fatal_type_ct = sum(1 for e in events if FATAL_TYPES.search(e))

    fatal_ratio   = fatal_count / n
    error_ratio   = error_count / n
    unique_ratio  = len(set(events)) / n

    # Unique nodes in the window (via node-like pattern)
    node_pat = re.compile(r'R\d+-M\d+-N')
    unique_nodes = len(set(m.group() for e in events for m in [node_pat.search(e)] if m))

    return {
        'win_id':         win_id,
        'seq_len':        n,
        'fatal_count':    fatal_count,
        'error_count':    error_count,
        'warn_count':     warn_count,
        'kern_count':     kern_count,
        'app_count':      app_count,
        'ciod_count':     ciod_count,
        'timeout_count':  timeout_count,
        'mem_count':      mem_count,
        'fatal_type_ct':  fatal_type_ct,
        'fatal_ratio':    fatal_ratio,
        'error_ratio':    error_ratio,
        'unique_ratio':   unique_ratio,
        'unique_nodes':   unique_nodes,
    }


def build_feature_matrix(windows: dict) -> pd.DataFrame:
    """Build feature DataFrame from all windows."""
    print(f"[BGL Features] Extracting features for {len(windows):,} windows...")
    rows = []
    for i, (win_id, w) in enumerate(windows.items()):
        feat = extract_window_features(win_id, w['events'])
        feat['label'] = w['label']
        rows.append(feat)
        if (i + 1) % 50_000 == 0:
            print(f"  Processed {i+1:,} windows...")

    df = pd.DataFrame(rows)
    print(f"[BGL Features] Matrix: {df.shape[0]:,} windows × {df.shape[1]} cols")
    if 'label' in df.columns:
        print(f"  Anomaly: {df['label'].sum():,} | Normal: {(df['label']==0).sum():,}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {'win_id', 'label'}
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in [np.float64, np.int64, int, float]]
