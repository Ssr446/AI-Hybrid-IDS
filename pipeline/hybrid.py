"""
pipeline/hybrid.py
Hybrid Decision Engine.
ThreatScore = alpha * rule_score + (1 - alpha) * anomaly_score
"""

SEVERITY_THRESHOLDS = {
    "CRITICAL": 0.80,
    "HIGH":     0.60,
    "MEDIUM":   0.40,
    "LOW":      0.20,
}

def classify_severity(threat_score: float) -> str:
    if threat_score >= SEVERITY_THRESHOLDS["CRITICAL"]:
        return "CRITICAL"
    elif threat_score >= SEVERITY_THRESHOLDS["HIGH"]:
        return "HIGH"
    elif threat_score >= SEVERITY_THRESHOLDS["MEDIUM"]:
        return "MEDIUM"
    elif threat_score >= SEVERITY_THRESHOLDS["LOW"]:
        return "LOW"
    return "NORMAL"


def fuse(rule_score: float, anomaly_score: float, alpha: float = 0.55) -> dict:
    """
    Fuse rule score and anomaly score into a ThreatScore.
    alpha: weight given to rule engine (0.5 = equal, higher = trust rules more)
    """
    threat_score = alpha * rule_score + (1 - alpha) * anomaly_score
    severity     = classify_severity(threat_score)
    return {
        "rule_score":    round(rule_score, 4),
        "anomaly_score": round(anomaly_score, 4),
        "threat_score":  round(threat_score, 4),
        "severity":      severity,
        "alpha":         alpha,
    }


def build_alert(rule_alert: dict, anomaly_score: float, alpha: float = 0.55) -> dict:
    """Build a unified alert from a rule match + anomaly score."""
    fusion = fuse(rule_alert["rule_score"], anomaly_score, alpha)
    return {
        **rule_alert,
        **fusion,
        "detection_method": "hybrid" if anomaly_score > 0.3 else "rule",
    }


def build_ai_only_alert(log, anomaly_score: float, alpha: float = 0.55) -> dict:
    """Build an alert for AI-detected anomaly with no rule match."""
    fusion = fuse(0.0, anomaly_score, alpha)
    return {
        "rule_id":          "AI-001",
        "rule_name":        "Anomaly — Unknown Pattern",
        "description":      f"Isolation Forest flagged anomalous behavior from {log.ip or 'unknown'}",
        "source_ip":        log.ip or "unknown",
        "count":            1,
        "sample_raw":       log.raw[:120],
        "detection_method": "isolation_forest",
        **fusion,
    }
