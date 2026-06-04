"""
pipeline/rules.py
Rule-based detection engine.
Each rule returns a (triggered: bool, rule_id: str, description: str, severity: str)
"""

import re
from collections import defaultdict
from pipeline.parser import ParsedLog


# ── RULE DEFINITIONS ──────────────────────────────────────────────────────────

def evaluate_rules(logs: list) -> list:
    """
    Evaluate all rules against a list of ParsedLog objects.
    Returns list of alert dicts.
    """
    alerts = []
    alerts += _rule_brute_force_ssh(logs)
    alerts += _rule_invalid_user_scan(logs)
    alerts += _rule_root_login_accepted(logs)
    alerts += _rule_suspicious_sudo(logs)
    alerts += _rule_multiple_users_same_ip(logs)
    alerts += _rule_sql_injection(logs)
    alerts += _rule_xss_attempt(logs)
    alerts += _rule_directory_traversal(logs)
    alerts += _rule_scanner_user_agent(logs)
    alerts += _rule_http_flood(logs)
    alerts += _rule_sensitive_file_access(logs)
    alerts += _rule_repeated_4xx(logs)
    return alerts


# ── AUTH RULES ─────────────────────────────────────────────────────────────────

def _is_ssh_source(log):
    return log.source in ("auth", "ssh", "linux")

def _rule_brute_force_ssh(logs):
    """Rule #1: More than 6 failed SSH passwords from same IP"""
    alerts = []
    ip_fails = defaultdict(list)
    for log in logs:
        if _is_ssh_source(log) and log.event_type == "failed_password":
            ip_fails[log.ip].append(log)
    for ip, entries in ip_fails.items():
        if len(entries) >= 10:
            alerts.append({
                "rule_id":     "AUTH-001",
                "rule_name":   "SSH Brute Force",
                "description": f"IP {ip} made {len(entries)} failed SSH login attempts",
                "severity":    "CRITICAL",
                "source_ip":   ip,
                "count":       len(entries),
                "rule_score":  min(1.0, 0.8 + (len(entries) - 10) * 0.02),
                "sample_raw":  entries[0].raw,
            })
    return alerts


def _rule_invalid_user_scan(logs):
    """Rule #2: More than 5 invalid user attempts from same IP"""
    alerts = []
    ip_inv = defaultdict(list)
    for log in logs:
        if _is_ssh_source(log) and log.event_type == "invalid_user":
            ip_inv[log.ip].append(log)
    for ip, entries in ip_inv.items():
        if len(entries) >= 5:
            alerts.append({
                "rule_id":     "AUTH-002",
                "rule_name":   "User Enumeration",
                "description": f"IP {ip} probed {len(entries)} invalid usernames",
                "severity":    "HIGH",
                "source_ip":   ip,
                "count":       len(entries),
                "rule_score":  min(1.0, 0.75 + (len(entries) - 5) * 0.05),
                "sample_raw":  entries[0].raw,
            })
    return alerts


def _rule_root_login_accepted(logs):
    """Rule #3: Successful root login from external IP"""
    alerts = []
    for log in logs:
        if (_is_ssh_source(log) and
                log.event_type == "accepted_password" and
                log.user == "root" and
                _is_external(log.ip)):
            alerts.append({
                "rule_id":     "AUTH-003",
                "rule_name":   "External Root Login",
                "description": f"Root login accepted from external IP {log.ip}",
                "severity":    "CRITICAL",
                "source_ip":   log.ip,
                "count":       1,
                "rule_score":  1.0,
                "sample_raw":  log.raw,
            })
    return alerts


def _rule_suspicious_sudo(logs):
    """Rule #4: sudo command accessing sensitive system paths"""
    alerts = []
    sensitive = ["/bin/bash", "/bin/sh", "passwd", "sudoers",
                 "chmod 777", "/etc/shadow", "/bin/su", "chown root"]
    for log in logs:
        if _is_ssh_source(log) and log.event_type == "sudo_command":
            cmd = log.extras.get("command", "")
            if any(s in cmd for s in sensitive):
                alerts.append({
                    "rule_id":     "AUTH-004",
                    "rule_name":   "Suspicious Sudo Command",
                    "description": f"User {log.user} ran sensitive sudo: {cmd[:60]}",
                    "severity":    "HIGH",
                    "source_ip":   log.ip or "localhost",
                    "count":       1,
                    "rule_score":  0.85,
                    "sample_raw":  log.raw,
                })
    return alerts


def _rule_multiple_users_same_ip(logs):
    """Rule #5: Single IP attempting login as 4+ different users"""
    alerts = []
    ip_users = defaultdict(set)
    for log in logs:
        if _is_ssh_source(log) and log.event_type in ("failed_password", "invalid_user"):
            if log.user and log.ip:
                ip_users[log.ip].add(log.user)
    for ip, users in ip_users.items():
        if len(users) >= 4:
            alerts.append({
                "rule_id":     "AUTH-005",
                "rule_name":   "Credential Stuffing",
                "description": f"IP {ip} tried {len(users)} different usernames: {list(users)[:5]}",
                "severity":    "HIGH",
                "source_ip":   ip,
                "count":       len(users),
                "rule_score":  min(1.0, 0.7 + (len(users) - 4) * 0.05),
                "sample_raw":  "",
            })
    return alerts


# ── APACHE RULES ───────────────────────────────────────────────────────────────

def _rule_sql_injection(logs):
    """Rule #6: SQL injection patterns in request path"""
    alerts = []
    sqli_ips = defaultdict(list)
    for log in logs:
        if log.source == "apache" and log.event_type == "sql_injection_attempt":
            sqli_ips[log.ip].append(log)
    for ip, entries in sqli_ips.items():
        alerts.append({
            "rule_id":     "HTTP-001",
            "rule_name":   "SQL Injection Attempt",
            "description": f"IP {ip} made {len(entries)} SQLi attempts. Sample: {entries[0].path[:60]}",
            "severity":    "CRITICAL" if len(entries) > 5 else "HIGH",
            "source_ip":   ip,
            "count":       len(entries),
            "rule_score":  min(1.0, 0.7 + len(entries) * 0.01),
            "sample_raw":  entries[0].raw,
        })
    return alerts


def _rule_xss_attempt(logs):
    """Rule #7: XSS patterns in request path"""
    alerts = []
    xss_ips = defaultdict(list)
    for log in logs:
        if log.source == "apache" and log.event_type == "xss_attempt":
            xss_ips[log.ip].append(log)
    for ip, entries in xss_ips.items():
        alerts.append({
            "rule_id":     "HTTP-002",
            "rule_name":   "XSS Attempt",
            "description": f"IP {ip} injected XSS payload. Sample: {entries[0].path[:60]}",
            "severity":    "HIGH",
            "source_ip":   ip,
            "count":       len(entries),
            "rule_score":  0.82,
            "sample_raw":  entries[0].raw,
        })
    return alerts


def _rule_directory_traversal(logs):
    """Rule #8: Directory traversal patterns"""
    alerts = []
    trav_tokens = ["../", "%2e%2e", "..%2f", "%2f.."]
    for log in logs:
        if log.source == "apache":
            path = log.path or ""
            if any(t in path for t in trav_tokens):
                alerts.append({
                    "rule_id":     "HTTP-003",
                    "rule_name":   "Directory Traversal",
                    "description": f"IP {log.ip} attempted path traversal: {path[:60]}",
                    "severity":    "HIGH",
                    "source_ip":   log.ip,
                    "count":       1,
                    "rule_score":  0.88,
                    "sample_raw":  log.raw,
                })
    return alerts[:10]  # cap for display


def _rule_scanner_user_agent(logs):
    """Rule #9: Known scanner/exploit tool user agents"""
    alerts = []
    scanners = ["nikto", "sqlmap", "nmap", "masscan", "dirbuster", "gobuster", "flood-bot"]
    scanner_ips = defaultdict(list)
    for log in logs:
        if log.source == "apache":
            ua = (log.user_agent or "").lower()
            if any(s in ua for s in scanners):
                scanner_ips[log.ip].append(log)
    for ip, entries in scanner_ips.items():
        scanner_name = next((s for s in scanners if s in (entries[0].user_agent or "").lower()), "scanner")
        alerts.append({
            "rule_id":     "HTTP-004",
            "rule_name":   "Scanner Detected",
            "description": f"IP {ip} using {scanner_name} ({len(entries)} requests)",
            "severity":    "HIGH",
            "source_ip":   ip,
            "count":       len(entries),
            "rule_score":  0.9,
            "sample_raw":  entries[0].raw,
        })
    return alerts


def _rule_http_flood(logs):
    """Rule #10: More than 100 requests from same IP"""
    alerts = []
    ip_counts = defaultdict(int)
    for log in logs:
        if log.source == "apache":
            ip_counts[log.ip] += 1
    for ip, count in ip_counts.items():
        if count >= 100:
            alerts.append({
                "rule_id":     "HTTP-005",
                "rule_name":   "HTTP Flood",
                "description": f"IP {ip} sent {count} requests (possible DDoS)",
                "severity":    "CRITICAL",
                "source_ip":   ip,
                "count":       count,
                "rule_score":  min(1.0, count / 200),
                "sample_raw":  "",
            })
    return alerts


def _rule_sensitive_file_access(logs):
    """Rule #11: Access attempts to sensitive files"""
    alerts = []
    sensitive = [".env", "wp-config", "phpmyadmin", ".git/config",
                 "backup.zip", "db.sql", "/etc/passwd", "shell.php"]
    for log in logs:
        if log.source == "apache":
            path = log.path or ""
            for s in sensitive:
                if s in path:
                    alerts.append({
                        "rule_id":     "HTTP-006",
                        "rule_name":   "Sensitive File Access",
                        "description": f"IP {log.ip} accessed sensitive path: {path[:60]}",
                        "severity":    "HIGH",
                        "source_ip":   log.ip,
                        "count":       1,
                        "rule_score":  0.87,
                        "sample_raw":  log.raw,
                    })
                    break
    return alerts[:10]


def _rule_repeated_4xx(logs):
    """Rule #12: More than 20 consecutive 4xx errors from same IP"""
    alerts = []
    ip_4xx = defaultdict(int)
    for log in logs:
        if log.source == "apache" and log.status_code and 400 <= log.status_code < 500:
            ip_4xx[log.ip] += 1
    for ip, count in ip_4xx.items():
        if count >= 20:
            alerts.append({
                "rule_id":     "HTTP-007",
                "rule_name":   "Repeated 4xx Errors",
                "description": f"IP {ip} triggered {count} client errors (scanning?)",
                "severity":    "MEDIUM",
                "source_ip":   ip,
                "count":       count,
                "rule_score":  min(0.8, count / 50),
                "sample_raw":  "",
            })
    return alerts


def _is_external(ip):
    if not ip:
        return False
    return not (ip.startswith("192.168.") or ip.startswith("10.") or
                ip.startswith("172.16.") or ip == "127.0.0.1")
