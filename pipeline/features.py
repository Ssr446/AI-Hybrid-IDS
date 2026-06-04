"""
pipeline/features.py

Per-log feature extraction for real Loghub datasets.
Works with OpenSSH, HDFS, Apache, and Linux log formats.
Produces one feature vector per log line for Isolation Forest.
"""

import re
import numpy as np
import pandas as pd
from pipeline.parser import ParsedLog

# ── KNOWN ATTACK SIGNATURES ────────────────────────────────────────────────────
SENSITIVE_CMDS = [
    "/bin/bash", "/bin/sh", "passwd root", "sudoers",
    "chmod 777", "/etc/shadow", "/bin/su", "chown root", "/etc/"
]

SENSITIVE_PATHS = [
    "passwd", "shadow", "sudoers", ".env", ".git", "wp-config",
    "phpmyadmin", "shell.php", "backup.zip", ".htaccess", "db.sql"
]

SQLI_TOKENS = [
    "union+select", "'+or+", "1=1", "drop+table", "select+*",
    "waitfor+delay", "sleep(", "'--", "or '1'='1", "benchmark("
]

XSS_TOKENS = [
    "<script", "onerror=", "onload=", "javascript:", "alert(",
    "<svg", "document.cookie"
]

SCANNER_UAS = [
    "nikto", "sqlmap", "nmap", "masscan", "dirbuster",
    "gobuster", "flood-bot", "zgrab", "hydra", "medusa"
]

HDFS_ANOMALY_KEYWORDS = [
    "exception", "error", "failed", "failure", "timeout",
    "lost", "corrupt", "missing", "refused", "denied", "warn"
]

# Known scanner / attacker patterns in SSH
BRUTE_FORCE_INDICATORS = [
    "failed password", "invalid user", "authentication failure",
    "possible break-in", "did not receive identification"
]


def extract_per_log_features(logs: list) -> pd.DataFrame:
    """
    Extracts one 20-dimensional feature vector per log line.
    All features are binary or normalized numeric — ready for Isolation Forest.
    """
    rows = []
    for log in logs:
        row = _extract_one(log)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Fill any missing cols with 0
    df = df.fillna(0)
    return df


def _extract_one(log: ParsedLog) -> dict:
    source = log.source
    et     = log.event_type or ""
    raw_l  = log.raw.lower()

    # ── Source encoding ──────────────────────────────────────────────────────
    is_ssh    = int(source == "ssh")
    is_hdfs   = int(source == "hdfs")
    is_apache = int(source == "apache")
    is_linux  = int(source == "linux")

    # ── SSH features ─────────────────────────────────────────────────────────
    is_failed_pw     = int(et == "failed_password")
    is_invalid_user  = int(et == "invalid_user")
    is_accepted      = int(et == "accepted_login")
    is_sudo          = int(et == "sudo_command")
    is_session       = int(et == "session_event")
    is_break_in      = int(et == "break_in_attempt")
    is_no_ident      = int(et == "no_identification")
    is_ssh_error     = int(et == "ssh_error")

    is_root_user     = int((log.user or "") in ("root", "admin", "administrator"))
    is_external_ip   = int(_is_external(log.ip or ""))

    susp_sudo        = int(
        et == "sudo_command" and
        any(s in log.extras.get("command", "").lower() for s in SENSITIVE_CMDS)
    )

    # ── HDFS features ─────────────────────────────────────────────────────────
    is_hdfs_anomaly  = int(et == "hdfs_anomaly")
    is_hdfs_error    = int((log.log_level or "") in ("ERROR", "WARN"))
    hdfs_kw_hit      = int(any(k in raw_l for k in HDFS_ANOMALY_KEYWORDS) and is_hdfs)

    # ── Apache features ───────────────────────────────────────────────────────
    path             = log.path or ""
    ua               = (log.user_agent or "").lower()
    status           = log.status_code or 0

    is_4xx           = int(400 <= status < 500)
    is_5xx           = int(status >= 500)
    is_sqli          = int(any(t in path for t in SQLI_TOKENS))
    is_xss           = int(any(t in path for t in XSS_TOKENS))
    is_susp_path     = int(any(s in path for s in SENSITIVE_PATHS))
    is_scanner_ua    = int(any(s in ua for s in SCANNER_UAS))

    # ── Generic anomaly signals ────────────────────────────────────────────────
    is_brute_signal  = int(any(k in raw_l for k in BRUTE_FORCE_INDICATORS))
    is_unknown_event = int(et == "unknown")

    return {
        # Source
        "is_ssh":           is_ssh,
        "is_hdfs":          is_hdfs,
        "is_apache":        is_apache,
        "is_linux":         is_linux,
        # SSH
        "is_failed_pw":     is_failed_pw,
        "is_invalid_user":  is_invalid_user,
        "is_accepted":      is_accepted,
        "is_sudo":          is_sudo,
        "is_session":       is_session,
        "is_break_in":      is_break_in,
        "is_no_ident":      is_no_ident,
        "is_ssh_error":     is_ssh_error,
        "is_root_user":     is_root_user,
        "is_external_ip":   is_external_ip,
        "is_susp_sudo":     susp_sudo,
        # HDFS
        "is_hdfs_anomaly":  is_hdfs_anomaly,
        "is_hdfs_error":    is_hdfs_error,
        "hdfs_kw_hit":      hdfs_kw_hit,
        # Apache
        "is_4xx":           is_4xx,
        "is_5xx":           is_5xx,
        "is_sqli":          is_sqli,
        "is_xss":           is_xss,
        "is_susp_path":     is_susp_path,
        "is_scanner_ua":    is_scanner_ua,
        # Generic
        "is_brute_signal":  is_brute_signal,
        "is_unknown":       is_unknown_event,
    }


def build_ip_burst_features(logs: list) -> pd.DataFrame:
    """
    Adds burst/frequency features per IP address.
    Useful for detecting floods and brute-force sweeps.
    Returns a DataFrame aligned with logs (one row per log).
    """
    from collections import Counter
    ip_counts = Counter(log.ip for log in logs if log.ip)
    ip_fail_counts = Counter(
        log.ip for log in logs
        if log.event_type in ("failed_password", "invalid_user") and log.ip
    )
    total = max(len(logs), 1)

    rows = []
    for log in logs:
        ip = log.ip or "unknown"
        rows.append({
            "ip_total_count":  ip_counts.get(ip, 0),
            "ip_fail_count":   ip_fail_counts.get(ip, 0),
            "ip_freq_norm":    round(ip_counts.get(ip, 0) / total, 6),
            "ip_fail_norm":    round(ip_fail_counts.get(ip, 0) / total, 6),
        })
    return pd.DataFrame(rows)


def get_combined_features(logs: list) -> pd.DataFrame:
    """
    Combines per-log features with IP burst features.
    This is the full feature matrix used for training.
    """
    base    = extract_per_log_features(logs)
    burst   = build_ip_burst_features(logs)
    return pd.concat([base, burst], axis=1).fillna(0)


def _is_external(ip: str) -> bool:
    if not ip or ip == "unknown":
        return False
    return not (
        ip.startswith("192.168.") or
        ip.startswith("10.")      or
        ip.startswith("172.16.")  or
        ip == "127.0.0.1"
    )
