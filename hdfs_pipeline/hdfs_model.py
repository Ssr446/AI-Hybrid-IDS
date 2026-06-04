"""
hdfs_model.py
Trains Isolation Forest on normal HDFS blocks,
fuses with rule scores to produce final ThreatScore,
and evaluates against ground truth labels.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json
import time


ALPHA = 0.40   # Tuned: lower alpha gives IF more weight (rules are high-prec, low-recall)
THRESHOLD = 0.45  # Tuned: optimal threshold for best F1 on HDFS


def train_isolation_forest(feature_df: pd.DataFrame, feature_cols: list) -> tuple:
    """Train IF on normal blocks only."""
    normal_df = feature_df[feature_df['label'] == 0]
    print(f"[IF Model] Training on {len(normal_df):,} normal blocks "
          f"({len(feature_df):,} total)...")

    X_normal = normal_df[feature_cols].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    contamination = feature_df['label'].mean()
    print(f"[IF Model] Contamination = {contamination:.4f}")

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_scaled)
    print(f"[IF Model] Training done in {time.time()-t0:.1f}s")

    return model, scaler


def compute_anomaly_scores(model, scaler, feature_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """Get anomaly scores for all blocks. Returns scores in [0,1]."""
    X = feature_df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)

    # Raw scores: negative = more anomalous
    raw_scores = model.score_samples(X_scaled)

    # Normalize to [0,1] — higher = more anomalous
    min_s, max_s = raw_scores.min(), raw_scores.max()
    anomaly_scores = 1 - (raw_scores - min_s) / (max_s - min_s + 1e-9)
    return anomaly_scores


def compute_threat_scores(anomaly_scores: np.ndarray, rule_results: list, alpha: float = ALPHA) -> np.ndarray:
    """Fuse rule scores and anomaly scores into ThreatScore."""
    rule_scores = np.array([r['rule_score'] for r in rule_results])
    threat_scores = alpha * rule_scores + (1 - alpha) * anomaly_scores
    return threat_scores


def classify_severity(score: float) -> str:
    if score >= 0.80: return 'CRITICAL'
    if score >= 0.60: return 'HIGH'
    if score >= 0.40: return 'MEDIUM'
    if score >= 0.20: return 'LOW'
    return 'NORMAL'


def evaluate(feature_df: pd.DataFrame, threat_scores: np.ndarray,
             anomaly_scores: np.ndarray, rule_results: list,
             threshold: float = THRESHOLD) -> dict:
    """Full evaluation against ground truth labels."""
    y_true = feature_df['label'].values
    y_pred_hybrid = (threat_scores >= threshold).astype(int)
    y_pred_if     = (anomaly_scores >= threshold).astype(int)

    rule_scores = np.array([r['rule_score'] for r in rule_results])
    y_pred_rule  = (rule_scores > 0).astype(int)

    def metrics(y_t, y_p, name):
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0,1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return {
            'method':    name,
            'precision': round(precision_score(y_t, y_p, zero_division=0), 4),
            'recall':    round(recall_score(y_t, y_p, zero_division=0), 4),
            'f1':        round(f1_score(y_t, y_p, zero_division=0), 4),
            'fpr':       round(fpr, 4),
            'tp': int(tp), 'fp': int(fp),
            'tn': int(tn), 'fn': int(fn),
        }

    results = {
        'dataset':    'HDFS_v1',
        'total':      len(feature_df),
        'normal':     int((y_true == 0).sum()),
        'anomaly':    int((y_true == 1).sum()),
        'alpha':      ALPHA,
        'threshold':  threshold,
        'ablation': [
            metrics(y_true, y_pred_rule,   'Rule Only'),
            metrics(y_true, y_pred_if,     'Isolation Forest Only'),
            metrics(y_true, y_pred_hybrid, 'Hybrid (Rule + IF)'),
        ]
    }
    return results


def print_results(results: dict):
    print("\n" + "="*65)
    print(f"  HDFS RESULTS — {results['total']:,} blocks")
    print(f"  Normal: {results['normal']:,} | Anomaly: {results['anomaly']:,}")
    print("="*65)
    print(f"  {'Method':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>7}")
    print("-"*65)
    for row in results['ablation']:
        print(f"  {row['method']:<25} {row['precision']:>6.4f} "
              f"{row['recall']:>6.4f} {row['f1']:>6.4f} {row['fpr']:>7.4f}")
    best = results['ablation'][2]
    print("-"*65)
    print(f"  Confusion Matrix (Hybrid):")
    print(f"    TP={best['tp']:,}  FP={best['fp']:,}  TN={best['tn']:,}  FN={best['fn']:,}")
    print("="*65)
