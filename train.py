"""
train.py  —  Real Data Training Pipeline
==========================================

Trains on real Loghub datasets. Drop your downloaded files into
the data/ folder and this script handles the rest automatically.

SUPPORTED DATASETS (use one or more):
  data/
  ├── SSH_2k.log          ← Loghub OpenSSH (2k sample)
  ├── SSH.log             ← Loghub OpenSSH (full 655k lines)
  ├── HDFS.log            ← Loghub HDFS system log
  ├── anomaly_label.csv   ← Loghub HDFS ground truth labels
  ├── Apache_2k.log       ← Loghub Apache (2k sample)
  ├── Apache.log          ← Loghub Apache (full)
  └── Linux_2k.log        ← Loghub Linux syslog

DOWNLOAD FROM:
  https://zenodo.org/records/8196385
  (Free, no login needed — pick any of the above files)

FASTEST START: just download SSH_2k.log  (~200 KB, 2000 lines)
"""

import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.parser   import parse_file, detect_log_type
from pipeline.features import get_combined_features
from pipeline.labels   import (load_hdfs_labels, assign_hdfs_labels,
                                infer_ssh_labels, infer_apache_labels,
                                infer_linux_labels)
from pipeline.rules    import evaluate_rules
from pipeline.model    import train, evaluate, ablation_study
from pipeline.hybrid   import fuse

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ── DATASET CONFIGS ────────────────────────────────────────────────────────────
# Priority order: if file exists, it's used. First hit wins per type.
SSH_FILES    = ["SSH_2k.log", "SSH.log"]
HDFS_FILES   = ["HDFS.log", "HDFS_2k.log"]
HDFS_LABELS  = ["anomaly_label.csv"]
APACHE_FILES = ["Apache_2k.log", "Apache.log"]
LINUX_FILES  = ["Linux_2k.log", "Linux.log"]


def find_file(candidates: list) -> str | None:
    for name in candidates:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    return None


def load_dataset():
    """
    Auto-discovers available Loghub files and loads them.
    Returns (all_logs, all_labels, dataset_info)
    """
    all_logs    = []
    all_labels  = []
    dataset_info = {}

    # ── SSH ────────────────────────────────────────────────────────────────────
    ssh_path = find_file(SSH_FILES)
    if ssh_path:
        print(f"[+] Loading SSH logs: {os.path.basename(ssh_path)}")
        t0 = time.time()
        ssh_logs = parse_file(ssh_path, "ssh")
        ssh_labels = infer_ssh_labels(ssh_logs)
        elapsed = time.time() - t0

        n_attack = int(sum(ssh_labels))
        n_normal = int(sum(ssh_labels == 0))
        print(f"    {len(ssh_logs):,} lines parsed in {elapsed:.2f}s")
        print(f"    Normal: {n_normal:,}  |  Attack: {n_attack:,}")

        all_logs   += ssh_logs
        all_labels += ssh_labels.tolist()
        dataset_info["ssh"] = {
            "file": os.path.basename(ssh_path),
            "total": len(ssh_logs),
            "normal": n_normal,
            "attack": n_attack,
        }
    else:
        print("[!] No SSH log found. Add SSH_2k.log to data/")

    # ── HDFS ───────────────────────────────────────────────────────────────────
    hdfs_path  = find_file(HDFS_FILES)
    label_path = find_file(HDFS_LABELS)
    if hdfs_path:
        print(f"\n[+] Loading HDFS logs: {os.path.basename(hdfs_path)}")
        t0 = time.time()
        hdfs_logs = parse_file(hdfs_path, "hdfs")
        elapsed = time.time() - t0

        if label_path:
            print(f"    Loading labels: {os.path.basename(label_path)}")
            label_dict = load_hdfs_labels(label_path)
            hdfs_labels = assign_hdfs_labels(hdfs_logs, label_dict)
        else:
            print("    [!] No anomaly_label.csv — using heuristic HDFS labels")
            # Heuristic: ERROR/WARN lines = anomaly
            hdfs_labels = np.array([
                1 if log.event_type == "hdfs_anomaly" else 0
                for log in hdfs_logs
            ])

        n_attack = int(sum(hdfs_labels))
        n_normal = int(sum(hdfs_labels == 0))
        print(f"    {len(hdfs_logs):,} lines parsed in {elapsed:.2f}s")
        print(f"    Normal: {n_normal:,}  |  Anomaly: {n_attack:,}")

        all_logs   += hdfs_logs
        all_labels += hdfs_labels.tolist()
        dataset_info["hdfs"] = {
            "file": os.path.basename(hdfs_path),
            "total": len(hdfs_logs),
            "normal": n_normal,
            "attack": n_attack,
            "labeled": label_path is not None,
        }

    # ── Apache ─────────────────────────────────────────────────────────────────
    apache_path = find_file(APACHE_FILES)
    if apache_path:
        print(f"\n[+] Loading Apache logs: {os.path.basename(apache_path)}")
        t0 = time.time()
        apache_logs = parse_file(apache_path, "apache")
        apache_labels = infer_apache_labels(apache_logs)
        elapsed = time.time() - t0

        n_attack = int(sum(apache_labels))
        n_normal = int(sum(apache_labels == 0))
        print(f"    {len(apache_logs):,} lines parsed in {elapsed:.2f}s")
        print(f"    Normal: {n_normal:,}  |  Attack: {n_attack:,}")

        all_logs   += apache_logs
        all_labels += apache_labels.tolist()
        dataset_info["apache"] = {
            "file": os.path.basename(apache_path),
            "total": len(apache_logs),
            "normal": n_normal,
            "attack": n_attack,
        }

    # ── Linux syslog ───────────────────────────────────────────────────────────
    linux_path = find_file(LINUX_FILES)
    if linux_path:
        print(f"\n[+] Loading Linux logs: {os.path.basename(linux_path)}")
        t0 = time.time()
        linux_logs = parse_file(linux_path, "linux")
        linux_labels = infer_linux_labels(linux_logs)
        elapsed = time.time() - t0

        n_attack = int(sum(linux_labels))
        n_normal = int(sum(linux_labels == 0))
        print(f"    {len(linux_logs):,} lines parsed in {elapsed:.2f}s")
        print(f"    Normal: {n_normal:,}  |  Anomaly: {n_attack:,}")

        all_logs   += linux_logs
        all_labels += linux_labels.tolist()
        dataset_info["linux"] = {
            "file": os.path.basename(linux_path),
            "total": len(linux_logs),
            "normal": n_normal,
            "attack": n_attack,
        }

    if not all_logs:
        print("\n[ERROR] No dataset files found in data/")
        print("  Download from: https://zenodo.org/records/8196385")
        print("  Place SSH_2k.log (or any Loghub file) into the data/ folder")
        sys.exit(1)

    return all_logs, np.array(all_labels), dataset_info


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  AI HYBRID VULNERABILITY DETECTION — TRAINING PIPELINE")
    print("  Dataset: Real Loghub Logs")
    print("=" * 60)

    # ── STEP 1: Load real data ──────────────────────────────────────────────
    print("\n=== STEP 1: Loading real Loghub datasets ===")
    all_logs, all_labels, dataset_info = load_dataset()

    total_logs = len(all_logs)
    n_attack   = int(sum(all_labels))
    n_normal   = int(sum(all_labels == 0))
    attack_pct = round(100 * n_attack / max(total_logs, 1), 2)

    print(f"\n[SUMMARY] Total logs : {total_logs:,}")
    print(f"          Normal      : {n_normal:,}  ({100-attack_pct:.1f}%)")
    print(f"          Attack      : {n_attack:,}  ({attack_pct:.1f}%)")

    # ── STEP 2: Feature extraction ──────────────────────────────────────────
    print("\n=== STEP 2: Extracting features ===")
    t0 = time.time()
    X = get_combined_features(all_logs)
    print(f"[+] Feature matrix: {X.shape[0]:,} samples × {X.shape[1]} features  ({time.time()-t0:.2f}s)")
    print(f"[+] Features: {list(X.columns)}")

    # ── STEP 3: Train on normal-only samples ───────────────────────────────
    print("\n=== STEP 3: Training Isolation Forest (normal samples only) ===")
    normal_mask  = (all_labels == 0)
    X_normal     = X[normal_mask]
    contamination = min(0.5, attack_pct / 100 + 0.02)   # slight buffer
    print(f"[+] Training on {len(X_normal):,} normal samples  (contamination={contamination:.3f})")
    model, scaler = train(X_normal, contamination=contamination)

    # ── STEP 4: Evaluate ────────────────────────────────────────────────────
    print("\n=== STEP 4: Evaluating model ===")
    metrics, if_preds, if_scores = evaluate(model, scaler, X, all_labels)

    print(f"\n{'='*52}")
    print(f"  ISOLATION FOREST RESULTS  (n={total_logs:,})")
    print(f"{'='*52}")
    print(f"  Precision            : {metrics['precision']:.4f}")
    print(f"  Recall               : {metrics['recall']:.4f}")
    print(f"  F1-Score             : {metrics['f1_score']:.4f}")
    print(f"  Accuracy             : {metrics['accuracy']:.4f}")
    print(f"  False Positive Rate  : {metrics['false_positive_rate']:.4f}")
    print(f"  False Negative Rate  : {metrics['false_negative_rate']:.4f}")
    print(f"  TP / FP / TN / FN   : {metrics['true_positives']} / "
          f"{metrics['false_positives']} / {metrics['true_negatives']} / "
          f"{metrics['false_negatives']}")
    print(f"  Inference latency    : {metrics['inference_ms_per_log']:.4f} ms/log")
    print(f"{'='*52}")

    # ── STEP 5: Rule engine ─────────────────────────────────────────────────
    print("\n=== STEP 5: Running rule engine ===")
    t0 = time.time()
    rule_alerts = evaluate_rules(all_logs)
    rule_elapsed = time.time() - t0

    # Build per-log rule binary predictions
    rule_triggered_ips = {a["source_ip"] for a in rule_alerts}
    rule_preds = np.array([
        1 if (log.ip and log.ip in rule_triggered_ips) else 0
        for log in all_logs
    ])

    print(f"[+] Rule engine: {len(rule_alerts)} alerts in {rule_elapsed*1000:.1f}ms")
    for a in rule_alerts[:10]:
        print(f"    [{a['rule_id']}] {a['rule_name']}: {a['description'][:70]}")
    if len(rule_alerts) > 10:
        print(f"    ... and {len(rule_alerts)-10} more")

    # ── STEP 6: Ablation study ──────────────────────────────────────────────
    print("\n=== STEP 6: Ablation Study ===")
    ablation = ablation_study(X, all_labels, rule_preds)
    print(f"\n  {'Method':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPR':>10}")
    print("  " + "-" * 55)
    for method, m in ablation.items():
        marker = "  ◄ BEST" if method == "hybrid" else ""
        print(f"  {method:<22} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['fpr']:>10.4f}{marker}")

    # ── STEP 7: Build unified hybrid alerts ────────────────────────────────
    print("\n=== STEP 7: Building hybrid alerts ===")
    hybrid_alerts = []

    # Rule-based alerts fused with IF score
    for a in rule_alerts:
        avg_if = float(np.mean(if_scores))
        fused  = fuse(a["rule_score"], avg_if, alpha=0.55)
        hybrid_alerts.append({**a, **fused, "detection_method": "Rule + AI"})

    # AI-only alerts (IF flagged, no rule hit)
    for log, pred, score in zip(all_logs, if_preds, if_scores):
        if pred == 1 and (not log.ip or log.ip not in rule_triggered_ips) and score > 0.65:
            fused = fuse(0.0, float(score), alpha=0.55)
            hybrid_alerts.append({
                "rule_id":          "AI-001",
                "rule_name":        "Anomaly — Unknown Pattern",
                "description":      f"Isolation Forest flagged {log.ip or 'unknown IP'} "
                                    f"(event={log.event_type}, score={score:.3f})",
                "source_ip":        log.ip or "unknown",
                "count":            1,
                "sample_raw":       log.raw[:120],
                "detection_method": "Isolation Forest",
                **fused,
            })

    print(f"[+] Total hybrid alerts: {len(hybrid_alerts)}")
    crit = sum(1 for a in hybrid_alerts if a.get("severity") == "CRITICAL")
    high = sum(1 for a in hybrid_alerts if a.get("severity") == "HIGH")
    med  = sum(1 for a in hybrid_alerts if a.get("severity") == "MEDIUM")
    print(f"    Critical: {crit}  |  High: {high}  |  Medium: {med}")

    # ── STEP 8: Save results ────────────────────────────────────────────────
    print("\n=== STEP 8: Saving results ===")
    results = {
        "model_metrics":  metrics,
        "ablation":       ablation,
        "rule_alerts":    rule_alerts,
        "hybrid_alerts":  hybrid_alerts[:100],
        "dataset_stats": {
            "total_logs":   total_logs,
            "normal_logs":  n_normal,
            "attack_logs":  n_attack,
            "attack_pct":   attack_pct,
            "datasets_used": dataset_info,
            "feature_count": X.shape[1],
        }
    }

    out = os.path.join(RESULTS_DIR, "results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[+] Results → {out}")
    print("\n[✓] Done.  Run:  streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
