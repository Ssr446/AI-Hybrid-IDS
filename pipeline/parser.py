"""
pipeline/parser.py

Parses real Loghub datasets:
  - OpenSSH (SSH_2k.log / SSH.log)      → auth-style SSH logs
  - HDFS    (HDFS.log)                  → distributed system logs  
  - Apache  (Apache_2k.log)             → web server logs
  - Linux   (Linux_2k.log)              → generic syslog

Loghub OpenSSH line format:
  Dec 10 06:55:46 LabSZ sshd[24200]: Failed password for invalid user webmaster from 173.234.31.186 port 38926 ssh2

Loghub HDFS line format:
  081109 203615 148 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_38865049064139660 terminating

Apache_2k.log format:
  [Sun Dec 04 04:47:44 2005] [notice] workerEnv.init() ok /etc/httpd/conf/workers2.properties
"""

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedLog:
    source: str              # "ssh", "hdfs", "apache", "linux"
    raw: str
    timestamp: Optional[str] = None
    ip: Optional[str] = None
    user: Optional[str] = None
    event_type: Optional[str] = None
    path: Optional[str] = None
    status_code: Optional[int] = None
    method: Optional[str] = None
    user_agent: Optional[str] = None
    log_level: Optional[str] = None   # for HDFS: INFO/WARN/ERROR
    component: Optional[str] = None   # for HDFS: component name
    template: str = ""
    extras: dict = field(default_factory=dict)


# ── SSH / OpenSSH PATTERNS (Loghub OpenSSH format) ────────────────────────────
SSH_PATTERNS = [
    (re.compile(
        r"(?P<ts>\w+\s+\d+\s[\d:]+).*sshd\[.*?\]:\s+Failed password for (?:invalid user )?(?P<user>\S+) from (?P<ip>[\d.]+)"
    ), "failed_password", "Failed password for <USER> from <IP>"),

    (re.compile(
        r"(?P<ts>\w+\s+\d+\s[\d:]+).*sshd\[.*?\]:\s+Accepted (?:password|publickey) for (?P<user>\S+) from (?P<ip>[\d.]+)"
    ), "accepted_login", "Accepted login for <USER> from <IP>"),

    (re.compile(
        r"(?P<ts>\w+\s+\d+\s[\d:]+).*sshd\[.*?\]:\s+Invalid user (?P<user>\S+) from (?P<ip>[\d.]+)"
    ), "invalid_user", "Invalid user <USER> from <IP>"),

    (re.compile(
        r"(?P<ts>\w+\s+\d+\s[\d:]+).*sshd\[.*?\]:\s+(?:Connection closed|Received disconnect|Disconnecting).*?(?P<ip>[\d.]+)"
    ), "disconnect", "Disconnect from <IP>"),

    (re.compile(
        r"(?P<ts>\w+\s+\d+\s[\d:]+).*sshd\[.*?\]:\s+reverse mapping checking.*?(?P<ip>[\d.]+).*POSSIBLE BREAK-IN"
    ), "break_in_attempt", "POSSIBLE BREAK-IN from <IP>"),

    (re.compile(
        r"(?P<ts>\w+\s+\d+\s[\d:]+).*sshd\[.*?\]:\s+Did not receive identification.*?(?P<ip>[\d.]+)"
    ), "no_identification", "No identification from <IP>"),

    (re.compile(
        r"(?P<ts>\w+\s+\d+\s[\d:]+).*sudo.*?:\s+(?P<user>\S+).*?COMMAND=(?P<cmd>.+)"
    ), "sudo_command", "sudo <USER> COMMAND=<CMD>"),

    (re.compile(
        r"(?P<ts>\w+\s+\d+\s[\d:]+).*pam_unix.*?session (?P<action>opened|closed) for user (?P<user>\S+)"
    ), "session_event", "session <ACTION> for user <USER>"),

    (re.compile(
        r"(?P<ts>\w+\s+\d+\s[\d:]+).*sshd\[.*?\]:\s+(?:error|fatal):\s+(?P<msg>.+)"
    ), "ssh_error", "SSH error: <MSG>"),

    (re.compile(
        r"(?P<ts>\w+\s+\d+\s[\d:]+).*sshd\[.*?\]:\s+Postponed.*?for (?P<user>\S+) from (?P<ip>[\d.]+)"
    ), "postponed_auth", "Postponed auth for <USER> from <IP>"),
]

# ── HDFS PATTERNS ──────────────────────────────────────────────────────────────
# Format: 081109 203615 148 INFO dfs.DataNode$PacketResponder: message
HDFS_PATTERN = re.compile(
    r"(?P<date>\d{6})\s+(?P<time>\d{6})\s+(?P<pid>\d+)\s+(?P<level>\w+)\s+(?P<component>[\w.$]+):\s+(?P<msg>.+)"
)

# ── APACHE PATTERNS (Loghub Apache format) ─────────────────────────────────────
# [Sun Dec 04 04:47:44 2005] [notice] message
APACHE_LOGHUB_PATTERN = re.compile(
    r"\[(?P<ts>[^\]]+)\]\s+\[(?P<level>\w+)\]\s+(?P<msg>.+)"
)
# Standard Apache access log
APACHE_ACCESS_PATTERN = re.compile(
    r'(?P<ip>[\d.]+).*?\[(?P<ts>[^\]]+)\]\s+"(?P<method>\w+)\s+(?P<path>\S+)[^"]*"\s+(?P<status>\d+)'
)

# ── LINUX SYSLOG PATTERNS ──────────────────────────────────────────────────────
LINUX_SYSLOG_PATTERN = re.compile(
    r"(?P<ts>\w+\s+\d+\s+[\d:]+)\s+(?P<host>\S+)\s+(?P<process>\S+?)(?:\[(?P<pid>\d+)\])?:\s+(?P<msg>.+)"
)

# Attack-indicator keywords for HDFS
HDFS_ERROR_KEYWORDS = ["exception", "error", "failed", "failure", "timeout",
                        "lost", "corrupt", "missing", "refused", "denied"]


def parse_ssh_line(line: str) -> ParsedLog:
    log = ParsedLog(source="ssh", raw=line)
    for pattern, event_type, template in SSH_PATTERNS:
        m = pattern.search(line)
        if m:
            log.event_type = event_type
            log.template   = template
            groups = m.groupdict()
            log.timestamp  = groups.get("ts")
            log.ip         = groups.get("ip")
            log.user       = groups.get("user")
            if "cmd" in groups:
                log.extras["command"] = groups.get("cmd", "")
            if "action" in groups:
                log.extras["action"] = groups.get("action", "")
            if "msg" in groups:
                log.extras["message"] = groups.get("msg", "")
            return log
    log.event_type = "unknown"
    log.template   = "UNKNOWN_SSH_EVENT"
    return log


def parse_hdfs_line(line: str) -> ParsedLog:
    log = ParsedLog(source="hdfs", raw=line)
    m = HDFS_PATTERN.match(line.strip())
    if not m:
        log.event_type = "unknown"
        log.template   = "UNKNOWN_HDFS_LINE"
        return log

    log.log_level  = m.group("level")
    log.component  = m.group("component")
    log.timestamp  = f"{m.group('date')} {m.group('time')}"
    msg = m.group("msg")

    # Extract block ID if present (used for session grouping)
    blk_match = re.search(r"blk_(-?\d+)", msg)
    if blk_match:
        log.extras["block_id"] = blk_match.group(1)

    # Classify event type by component and message
    level = log.log_level.upper()
    comp  = log.component.lower()
    msg_l = msg.lower()

    if level in ("ERROR", "WARN") or any(k in msg_l for k in HDFS_ERROR_KEYWORDS):
        log.event_type = "hdfs_anomaly"
        log.template   = f"HDFS {level} <COMPONENT>: <ANOMALY_MSG>"
    elif "namenode" in comp:
        log.event_type = "hdfs_namenode"
        log.template   = "HDFS NameNode: <MSG>"
    elif "datanode" in comp:
        log.event_type = "hdfs_datanode"
        log.template   = "HDFS DataNode: <MSG>"
    elif "client" in comp or "dfsClient" in comp.lower():
        log.event_type = "hdfs_client"
        log.template   = "HDFS Client: <MSG>"
    else:
        log.event_type = "hdfs_general"
        log.template   = "HDFS <COMPONENT>: <MSG>"

    return log


def parse_apache_line(line: str) -> ParsedLog:
    log = ParsedLog(source="apache", raw=line)

    # Try Loghub Apache format first
    m = APACHE_LOGHUB_PATTERN.match(line.strip())
    if m:
        log.timestamp  = m.group("ts")
        log.log_level  = m.group("level")
        msg = m.group("msg")
        msg_l = msg.lower()
        if log.log_level in ("error", "crit", "alert", "emerg"):
            log.event_type = "apache_error"
        elif "access" in msg_l or "denied" in msg_l:
            log.event_type = "apache_access_denied"
        else:
            log.event_type = f"apache_{log.log_level}"
        log.template = f"Apache [{log.log_level.upper()}] <MSG>"
        return log

    # Try standard access log
    m = APACHE_ACCESS_PATTERN.match(line.strip())
    if m:
        log.ip          = m.group("ip")
        log.timestamp   = m.group("ts")
        log.method      = m.group("method")
        log.path        = m.group("path").lower()
        log.status_code = int(m.group("status"))
        log.event_type  = f"http_{log.method.lower()}"
        log.template    = f"HTTP {log.method} <PATH> {log.status_code}"
        return log

    log.event_type = "unknown"
    log.template   = "UNKNOWN_APACHE_LINE"
    return log


def parse_linux_line(line: str) -> ParsedLog:
    log = ParsedLog(source="linux", raw=line)
    m = LINUX_SYSLOG_PATTERN.match(line.strip())
    if not m:
        log.event_type = "unknown"
        log.template   = "UNKNOWN_LINUX_EVENT"
        return log

    log.timestamp = m.group("ts")
    process = m.group("process").lower()
    msg     = m.group("msg")
    msg_l   = msg.lower()

    # Re-use SSH parser if it's an sshd line
    if "sshd" in process:
        return parse_ssh_line(line)

    if any(k in msg_l for k in ["error", "failed", "failure", "critical", "denied"]):
        log.event_type = "linux_error"
    elif "kernel" in process:
        log.event_type = "kernel_event"
    else:
        log.event_type = "linux_general"

    log.template = f"Linux {process}: <MSG>"
    return log


def parse_file(filepath: str, source: str) -> list:
    """
    Parse a Loghub log file.
    source: "ssh" | "hdfs" | "apache" | "linux"
    """
    parsers = {
        "ssh":    parse_ssh_line,
        "hdfs":   parse_hdfs_line,
        "apache": parse_apache_line,
        "linux":  parse_linux_line,
    }
    parser_fn = parsers.get(source, parse_linux_line)

    results = []
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(parser_fn(line))
    return results


def detect_log_type(filepath: str) -> str:
    """Auto-detect which parser to use based on file content."""
    with open(filepath, "r", errors="ignore") as f:
        sample = [f.readline().strip() for _ in range(10)]

    joined = " ".join(sample).lower()

    if "sshd" in joined or "failed password" in joined or "invalid user" in joined:
        return "ssh"
    if re.search(r"\d{6}\s+\d{6}\s+\d+\s+(INFO|WARN|ERROR)", " ".join(sample)):
        return "hdfs"
    if re.search(r"\[.+?\]\s+\[(notice|error|warn|info)\]", " ".join(sample)):
        return "apache"
    return "linux"
