"""
bgl_parser.py
Parses BGL.log from Loghub.
BGL log format (space-separated):
  Col 0: Label ('-' = normal, else alert-type string)
  Col 1: Timestamp (unix)
  Col 2: Date (YYYY.MM.DD)
  Col 3: Node
  Col 4: DateTime (YYYY-MM-DD-HH.MM.SS.ffffff)
  Col 5: Node (repeat)
  Col 6: System type
  Col 7: Component
  Col 8: Level (INFO/FATAL/WARNING/SEVERE/etc.)
  Col 9+: Content

Unit of detection: fixed-size sliding window of N consecutive log lines,
grouped by node (Col 3) — matches standard BGL evaluation methodology.
"""

import re
import pandas as pd
from collections import defaultdict
from pathlib import Path

WINDOW_SIZE = 20   # lines per session window
STEP_SIZE   = 10   # step size (overlap 50%)


def parse_bgl_log(log_path: str, max_lines: int = None) -> pd.DataFrame:
    """Parse BGL.log → DataFrame with one row per line."""
    print(f"[BGL Parser] Reading {log_path} ...")
    records = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 9)
            if len(parts) < 9:
                continue
            label    = parts[0]                   # '-' or alert tag
            node     = parts[3]                   # e.g. R02-M1-N0-C:J12-U11
            level    = parts[8] if len(parts) > 8 else 'INFO'
            content  = parts[9] if len(parts) > 9 else ''
            is_anomaly = 0 if label == '-' else 1

            records.append({
                'label':      is_anomaly,
                'node':       node,
                'level':      level,
                'content':    content,
                'raw_label':  label,
            })

            if (i + 1) % 1_000_000 == 0:
                print(f"  Parsed {i+1:,} lines...")

    df = pd.DataFrame(records)
    total    = len(df)
    anomaly  = df['label'].sum()
    print(f"[BGL Parser] Done — {total:,} lines | Anomaly: {anomaly:,} | Normal: {total-anomaly:,}")
    return df


def make_windows(df: pd.DataFrame, window: int = WINDOW_SIZE, step: int = STEP_SIZE) -> dict:
    """
    Create fixed-size sliding windows PER NODE.
    Returns dict: {window_id: {'events': [...], 'label': 0/1}}
    Window is anomalous if ANY line in it is anomalous.
    """
    print(f"[BGL Parser] Creating windows (size={window}, step={step})...")
    windows  = {}
    win_id   = 0

    for node, group in df.groupby('node'):
        lines   = group['content'].tolist()
        labels  = group['label'].tolist()
        n       = len(lines)

        for start in range(0, n - window + 1, step):
            end     = start + window
            events  = lines[start:end]
            lbl     = 1 if any(labels[start:end]) else 0
            windows[f"win_{win_id}"] = {'events': events, 'label': lbl, 'node': node}
            win_id += 1

    total_wins = len(windows)
    anom_wins  = sum(1 for v in windows.values() if v['label'] == 1)
    print(f"[BGL Parser] {total_wins:,} windows | Anomaly: {anom_wins:,} | Normal: {total_wins-anom_wins:,}")
    return windows
