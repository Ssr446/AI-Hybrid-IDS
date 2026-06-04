"""
hdfs_parser.py
Parses HDFS.log from Loghub and groups log lines by block ID (blk_XXXX).
Each block becomes one "session" — the unit of anomaly detection.
"""

import re
import pandas as pd
from collections import defaultdict
from pathlib import Path


# HDFS log format:
# 081109 203615 148 INFO dfs.DataNode$PacketResponder: ... blk_-1608999687919862906 ...
LOG_PATTERN = re.compile(
    r'(\d{6})\s+(\d{6})\s+(\d+)\s+(\w+)\s+(\S+):\s+(.*)'
)
BLOCK_PATTERN = re.compile(r'(blk_-?\d+)')


def parse_hdfs_log(log_path: str, max_lines: int = None) -> pd.DataFrame:
    """
    Parse HDFS.log into a DataFrame with one row per log line.
    Extracts block ID from each line for session grouping.
    """
    print(f"[HDFS Parser] Reading {log_path} ...")
    records = []
    
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue

            m = LOG_PATTERN.match(line)
            if not m:
                continue

            date, time_, pid, level, component, content = m.groups()
            blocks = BLOCK_PATTERN.findall(content)
            block_id = blocks[0] if blocks else None

            records.append({
                'date':      date,
                'time':      time_,
                'pid':       int(pid),
                'level':     level,
                'component': component,
                'content':   content,
                'block_id':  block_id,
                'raw':       line,
            })

            if (i + 1) % 1_000_000 == 0:
                print(f"  Parsed {i+1:,} lines...")

    df = pd.DataFrame(records)
    print(f"[HDFS Parser] Done — {len(df):,} lines, {df['block_id'].nunique():,} unique blocks")
    return df


def load_labels(label_path: str) -> dict:
    """
    Load anomaly_label.csv → dict {block_id: label}
    Label is 'Anomaly' or 'Normal'
    """
    print(f"[HDFS Parser] Loading labels from {label_path} ...")
    df = pd.read_csv(label_path)
    # columns: BlockId, Label
    label_map = dict(zip(df['BlockId'], df['Label']))
    anomaly_count = sum(1 for v in label_map.values() if v == 'Anomaly')
    print(f"[HDFS Parser] {len(label_map):,} blocks — "
          f"Anomaly: {anomaly_count:,}, Normal: {len(label_map)-anomaly_count:,}")
    return label_map


def group_by_block(df: pd.DataFrame) -> dict:
    """
    Group log lines by block_id → dict {block_id: [list of content strings]}
    Uses vectorized groupby for performance on large files.
    """
    df_valid = df[df['block_id'].notna() & (df['block_id'] != '')]
    groups = df_valid.groupby('block_id')['content'].apply(list).to_dict()
    print(f"[HDFS Parser] Grouped into {len(groups):,} blocks")
    return groups
