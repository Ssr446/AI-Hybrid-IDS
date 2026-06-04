"""
pipeline/labels.py

Loads ground-truth labels from Loghub datasets.

Loghub provides labels in two ways:
  1. HDFS_v1: anomaly_label.csv  (BlockId, Label)
              Label: "Anomaly" or "Normal"
  2. OpenSSH: no separate label file — labels are inferred from
              known attack patterns in the logs themselves
              (Failed password bursts, Invalid user sweeps, etc.)
"""

import re
import csv
import numpy as np
from collections import defaultdict


# ── HDFS LABELS ────────────────────────────────────────────────────────────────

def load_hdfs_labels(label_csv_path: str) -> dict:
    """
    Loads HDFS anomaly_label.csv.
    Returns dict: {block_id: 1 (anomaly) or 0 (normal)}

    CSV format:
        BlockId,Label
        blk_-1608999687919862906,Anomaly
        blk_7503483334202473044,Normal
    """
    labels = {}
    with open(label_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_id = row["BlockId"].strip().replace("blk_", "")
            is_anomaly = 1 if row["Label"].strip().lower() == "anomaly" else 0
            labels[block_id] = is_anomaly
    return labels


def assign_hdfs_labels(logs: list, label_dict: dict) -> np.ndarray:
    """
    Maps each parsed HDFS log line to its label using block_id.
    Lines with no block_id → labeled 0 (normal) by default.
    """
    y = []
    for log in logs:
        block_id = log.extras.get("block_id", None)
        if block_id and block_id in label_dict:
            y.append(label_dict[block_id])
        else:
            y.append(0)
    return np.array(y)


# ── SSH LABELS (inferred from log content) ─────────────────────────────────────

def infer_ssh_labels(logs: list) -> np.ndarray:
    """
    Infers attack labels from real OpenSSH log patterns.

    Attack heuristics (based on Loghub SSH research and security literature):
      - IP with >= 6 failed_password events      → all its lines = attack
      - IP with >= 4 invalid_user events         → all its lines = attack
      - Any break_in_attempt event               → attack
      - Any no_identification event from ext IP  → attack
      - SSH errors from external IPs             → attack
      - Root login accepted from external IP     → attack

    Everything else → normal
    """
    from pipeline.features import _is_external
    from collections import Counter

    # Count per-IP events
    ip_fail  = Counter()
    ip_inval = Counter()

    for log in logs:
        if log.event_type == "failed_password" and log.ip:
            ip_fail[log.ip]  += 1
        if log.event_type == "invalid_user" and log.ip:
            ip_inval[log.ip] += 1

    # IPs above threshold are attackers
    attacker_ips = set()
    for ip, count in ip_fail.items():
        if count >= 6:
            attacker_ips.add(ip)
    for ip, count in ip_inval.items():
        if count >= 4:
            attacker_ips.add(ip)

    y = []
    for log in logs:
        is_attack = 0

        # IP-level attack
        if log.ip and log.ip in attacker_ips:
            is_attack = 1

        # Event-level attacks (regardless of IP threshold)
        elif log.event_type == "break_in_attempt":
            is_attack = 1
        elif log.event_type == "no_identification" and _is_external(log.ip or ""):
            is_attack = 1
        elif (log.event_type == "accepted_login" and
              (log.user or "") == "root" and
              _is_external(log.ip or "")):
            is_attack = 1
        elif log.event_type == "ssh_error" and _is_external(log.ip or ""):
            is_attack = 1

        y.append(is_attack)

    return np.array(y)


# ── APACHE LABELS (inferred from log content) ─────────────────────────────────

def infer_apache_labels(logs: list) -> np.ndarray:
    """
    Infers attack labels from Apache log events.
    """
    SQLI_TOKENS  = ["union+select", "'+or+", "1=1", "drop+table", "select+*", "'--"]
    XSS_TOKENS   = ["<script", "onerror=", "onload=", "javascript:", "alert("]
    SUSP_PATHS   = ["passwd", "shadow", ".env", ".git", "wp-config", "shell.php"]
    SCANNER_UAS  = ["nikto", "sqlmap", "nmap", "masscan", "dirbuster", "hydra"]

    from collections import Counter
    ip_counts = Counter(log.ip for log in logs if log.ip)

    y = []
    for log in logs:
        is_attack = 0
        path = log.path or ""
        ua   = (log.user_agent or "").lower()

        if any(t in path for t in SQLI_TOKENS):
            is_attack = 1
        elif any(t in path for t in XSS_TOKENS):
            is_attack = 1
        elif any(s in path for s in SUSP_PATHS):
            is_attack = 1
        elif any(s in ua for s in SCANNER_UAS):
            is_attack = 1
        elif ip_counts.get(log.ip, 0) >= 100:  # flood
            is_attack = 1

        y.append(is_attack)

    return np.array(y)


# ── LINUX LABELS (inferred) ────────────────────────────────────────────────────

def infer_linux_labels(logs: list) -> np.ndarray:
    """
    Infers labels from Linux syslog.
    Treats ERROR/CRIT level events and sshd attack patterns as anomalies.
    """
    y = []
    for log in logs:
        raw_l = log.raw.lower()
        is_attack = int(
            log.event_type in ("failed_password", "invalid_user",
                               "break_in_attempt", "linux_error") or
            any(k in raw_l for k in ["segfault", "kernel panic", "oom-killer",
                                      "authentication failure", "possible break"])
        )
        y.append(is_attack)
    return np.array(y)
