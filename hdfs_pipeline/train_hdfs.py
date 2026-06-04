"""
train_hdfs.py  —  HDFS Phase 1 Pipeline (Fixed Rules v3)
=========================================================
Usage:
    cd hdfs_pipeline && python train_hdfs.py

Expects:
    HDFS_LOG_PATH   = path to HDFS.log
    LABEL_PATH      = path to anomaly_label.csv
    RESULTS_PATH    = where to save results JSON
"""

import json
import time
import os
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
HDFS_LOG_PATH = r"C:\Users\ssrsh\Documents\projects\minor project\HDFS.log"
LABEL_PATH    = r"C:\Users\ssrsh\Documents\projects\minor project\data\anomaly_label.csv"
RESULTS_PATH  = r"C:\Users\ssrsh\Documents\projects\minor project\minor2\results\hdfs_results.json"
# ─────────────────────────────────────────────────────────────────────────────

from hdfs_parser   import parse_hdfs_log, load_labels, group_by_block
from hdfs_features import build_feature_matrix, get_feature_columns
from hdfs_rules    import evaluate_all_blocks
from hdfs_model    import (train_isolation_forest, compute_anomaly_scores, compute_threat_scores, evaluate, print_results)


def main():
    t_start = time.time()
    print("\n" + "="*65)
    print("  AI-ASSISTED HYBRID VULNERABILITY DETECTION — HDFS Phase")
    print("="*65 + "\n")

    # 1. Parse logs
    df = parse_hdfs_log(HDFS_LOG_PATH)

    # 2. Load ground truth labels
    label_map = load_labels(LABEL_PATH)

    # 3. Group by block ID
    block_groups = group_by_block(df)

    # 4. Extract features
    feature_df = build_feature_matrix(block_groups, label_map)

    # 5. Get feature columns
    feat_cols = get_feature_columns(feature_df)
    print(f"\n[Main] Using {len(feat_cols)} features: {feat_cols[:5]}...")

    # 6. Rule engine
    rule_results = evaluate_all_blocks(block_groups, feature_df)

    # 7. Train Isolation Forest
    model, scaler = train_isolation_forest(feature_df, feat_cols)

    # 8. Anomaly scores
    print("\n[Main] Computing anomaly scores...")
    t0 = time.time()
    anomaly_scores = compute_anomaly_scores(model, scaler, feature_df, feat_cols)
    latency_ms = (time.time() - t0) / len(feature_df) * 1000
    print(f"[Main] Inference latency: {latency_ms:.4f} ms/block")

    # 9. ThreatScore fusion
    threat_scores = compute_threat_scores(anomaly_scores, rule_results)

    # 10. Evaluate
    results = evaluate(feature_df, threat_scores, anomaly_scores, rule_results)
    results['latency_ms_per_block'] = round(latency_ms, 4)
    results['total_time_sec']       = round(time.time() - t_start, 1)

    # 11. Print and save
    print_results(results)

    os.makedirs(Path(RESULTS_PATH).parent, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Main] Results saved to {RESULTS_PATH}")
    print(f"[Main] Total time: {results['total_time_sec']}s")


if __name__ == '__main__':
    main()
