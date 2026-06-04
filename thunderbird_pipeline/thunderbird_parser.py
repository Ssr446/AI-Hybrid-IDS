"""
thunderbird_parser.py

Parses Thunderbird.log from Loghub.
Thunderbird log format (space-separated):
  Col 0: Label ('-' = normal, else alert-type string)
  Col 1: Timestamp (unix)
  Col 2: Date (YYYY.MM.DD)
  Col 3: Node
  Col 4: Month
  Col 5: Day
  Col 6: Time
  Col 7: User/Node context
  Col 8+: Content (daemon: message)

Unit of detection: fixed-size sliding window of consecutive log lines,
grouped by node (Col 3).
"""

import pandas as pd

WINDOW_SIZE = 20
STEP_SIZE   = 10


def parse_thunderbird_log(log_path: str, max_lines: int = None) -> pd.DataFrame:
    print(f"[Thunderbird Parser] Reading {log_path} ...")
    records = []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 8)
            if len(parts) < 9:
                continue
            
            label   = parts[0]
            node    = parts[3]
            content = parts[8]
            is_anomaly = 0 if label == '-' else 1

            records.append({
                'label':     is_anomaly,
                'node':      node,
                'content':   content,
                'raw_label': label,
            })

            if (i + 1) % 5_000_000 == 0:
                print(f"  Parsed {i+1:,} lines...")

    df = pd.DataFrame(records)
    total   = len(df)
    anomaly = df['label'].sum()
    print(f"[Thunderbird Parser] Done — {total:,} lines | Anomaly: {anomaly:,} | Normal: {total-anomaly:,}")
    return df


def make_windows(df: pd.DataFrame, window: int = WINDOW_SIZE, step: int = STEP_SIZE) -> dict:
    print(f"[Thunderbird Parser] Creating windows (size={window}, step={step})...")
    windows = {}
    win_id  = 0

    for node, group in df.groupby('node'):
        lines  = group['content'].tolist()
        labels = group['label'].tolist()
        n      = len(lines)

        for start in range(0, n - window + 1, step):
            end    = start + window
            events = lines[start:end]
            lbl    = 1 if any(labels[start:end]) else 0
            windows[f"tb_{win_id}"] = {'events': events, 'label': lbl, 'node': node}
            win_id += 1

    total_wins = len(windows)
    anom_wins  = sum(1 for v in windows.values() if v['label'] == 1)
    print(f"[Thunderbird Parser] {total_wins:,} windows | Anomaly: {anom_wins:,}")
    return windows
