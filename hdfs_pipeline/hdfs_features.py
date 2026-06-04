"""
hdfs_features.py
Extracts a feature vector per HDFS block session.
Each block = one sequence of log events = one sample for the model.
"""

import re
import numpy as np
import pandas as pd
from collections import Counter


# HDFS event keywords mapped to numeric event IDs
EVENT_KEYWORDS = {
    'E1':  r'Receiving block',
    'E2':  r'Served block',
    'E3':  r'writeBlock',
    'E4':  r'Deleting block',
    'E5':  r'received block',
    'E6':  r'PacketResponder.*Exception',
    'E7':  r'Exception|Error|Failed|failed',
    'E8':  r'heartbeat',
    'E9':  r'allocate',
    'E10': r'replicate|Replicat',
    'E11': r'WARN|WARNING',
    'E12': r'INFO',
}

ERROR_PATTERN  = re.compile(r'Exception|Error|Failed|failed|WARN|WARNING', re.IGNORECASE)
REPLICATE_PAT  = re.compile(r'replicate|Replicat', re.IGNORECASE)
BLOCK_SIZE_PAT = re.compile(r'size\s+(\d+)')


def extract_block_features(block_id: str, events: list) -> dict:
    """
    Given a list of log line contents for one block,
    return a feature dict representing that block's behavior.
    """
    n = len(events)
    joined = ' '.join(events)

    # Event type counts
    event_counts = {}
    for eid, pattern in EVENT_KEYWORDS.items():
        event_counts[f'count_{eid}'] = sum(
            1 for e in events if re.search(pattern, e)
        )

    # Error indicators
    error_count    = sum(1 for e in events if ERROR_PATTERN.search(e))
    replicate_count = sum(1 for e in events if REPLICATE_PAT.search(e))

    # Block size (take max mentioned size)
    sizes = [int(m.group(1)) for e in events for m in [BLOCK_SIZE_PAT.search(e)] if m]
    max_size = max(sizes) if sizes else 0

    # Unique components
    comp_pat = re.compile(r'dfs\.(\w+)')
    components = set()
    for e in events:
        for m in comp_pat.finditer(e):
            components.add(m.group(1))

    # Sequence length features
    unique_event_ratio = len(set(events)) / max(n, 1)
    error_ratio        = error_count / max(n, 1)

    features = {
        'block_id':           block_id,
        'seq_len':            n,
        'error_count':        error_count,
        'error_ratio':        error_ratio,
        'replicate_count':    replicate_count,
        'max_block_size':     max_size,
        'unique_event_ratio': unique_event_ratio,
        'unique_components':  len(components),
        **event_counts,
    }
    return features


def build_feature_matrix(block_groups: dict, label_map: dict = None) -> pd.DataFrame:
    """
    Build feature DataFrame from all block groups.
    Optionally attach ground truth labels.
    """
    print(f"[HDFS Features] Extracting features for {len(block_groups):,} blocks...")
    rows = []
    for i, (block_id, events) in enumerate(block_groups.items()):
        feat = extract_block_features(block_id, events)
        if label_map:
            raw_label = label_map.get(block_id, 'Normal')
            feat['label'] = 1 if raw_label == 'Anomaly' else 0
        rows.append(feat)

        if (i + 1) % 50_000 == 0:
            print(f"  Processed {i+1:,} blocks...")

    df = pd.DataFrame(rows)
    print(f"[HDFS Features] Feature matrix: {df.shape[0]:,} blocks × {df.shape[1]} columns")

    if label_map and 'label' in df.columns:
        print(f"  Anomaly blocks: {df['label'].sum():,} | "
              f"Normal blocks: {(df['label']==0).sum():,}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return only numeric feature columns (exclude block_id and label)."""
    exclude = {'block_id', 'label'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, int, float]]
