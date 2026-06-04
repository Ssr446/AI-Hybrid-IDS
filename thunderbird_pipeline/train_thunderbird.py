"""
train_thunderbird.py — Thunderbird Phase 2 Pipeline
===================================================
Usage:
    cd thunderbird_pipeline && python train_thunderbird.py

Dataset: Thunderbird.log from Loghub
"""

import json
import time
import os
import sys
from pathlib import Path

# ── PATHS ─────────────────────────────────────────────────────────────────────
# We process a sample subset of Thunderbird by default since it is ~35GB unzipped
TB_LOG_PATH  = r"C:\Users\ssrsh\Documents\projects\minor project\Thunderbird.log"
RESULTS_PATH = r"C:\Users\ssrsh\Documents\projects\minor project\minor2\results\thunderbird_results.json"
MAX_LINES    = 5000000 # Evaluate first 5 million lines by default if log is huge
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))
from thunderbird_parser   import parse_thunderbird_log, make_windows
from thunderbird_features import build_feature_matrix, get_feature_columns
from thunderbird_rules    import evaluate_all_windows
from thunderbird_model    import (train_isolation_forest, compute_anomaly_scores,
                                  compute_threat_scores, evaluate, print_results)


def main():
    t_start = time.time()
    print("\n" + "="*65)
    print("  AI-ASSISTED HYBRID VULNERABILITY DETECTION — Thunderbird Phase")
    print("="*65 + "\n")

    if not os.path.exists(TB_LOG_PATH):
        print(f"[Error] Thunderbird log not found at {TB_LOG_PATH}.")
        print("Please ensure Thunderbird.log is extracted before running.")
        sys.exit(1)

    # 1. Parse (with MAX_LINES to avoid memory overload for testing)
    df = parse_thunderbird_log(TB_LOG_PATH, max_lines=MAX_LINES)

    # 2. Sliding windows per node
    windows = make_windows(df)

    # 3. Features
    feature_df = build_feature_matrix(windows)
    feat_cols  = get_feature_columns(feature_df)
    print(f"\n[Main] Using {len(feat_cols)} features: {feat_cols[:5]}...")

    # 4. Rule engine
    rule_results = evaluate_all_windows(windows, feature_df)

    # 5. Isolation Forest
    model, scaler = train_isolation_forest(feature_df, feat_cols)

    # 6. Scores
    print("\n[Main] Computing anomaly scores...")
    t0 = time.time()
    anomaly_scores = compute_anomaly_scores(model, scaler, feature_df, feat_cols)
    latency_ms = (time.time() - t0) / max(len(feature_df), 1) * 1000
    print(f"[Main] Inference latency: {latency_ms:.4f} ms/window")

    # 7. Fusion
    threat_scores = compute_threat_scores(anomaly_scores, rule_results)

    # 8. Evaluate
    results = evaluate(feature_df, threat_scores, anomaly_scores, rule_results)
    results['latency_ms_per_window'] = round(latency_ms, 4)
    results['total_time_sec']        = round(time.time() - t_start, 1)

    # 9. Print & save
    print_results(results)
    os.makedirs(Path(RESULTS_PATH).parent, exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[Main] Results saved to {RESULTS_PATH}")
    print(f"[Main] Total time: {results['total_time_sec']}s")


if __name__ == '__main__':
    main()
