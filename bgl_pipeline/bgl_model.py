"""
bgl_model.py
Isolation Forest + Rule Fusion for BGL windows.
"""

import numpy as np
import pandas as pd
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix
)

ALPHA     = 0.55   # Rule weight (tuned same as SSH; BGL rules are strong)
THRESHOLD = 0.40   # Decision threshold


def train_isolation_forest(feature_df: pd.DataFrame, feature_cols: list) -> tuple:
    normal_df = feature_df[feature_df['label'] == 0]
    print(f"[IF Model] Training on {len(normal_df):,} normal windows "
          f"({len(feature_df):,} total)...")
    X_normal = normal_df[feature_cols].fillna(0).values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    contamination = float(feature_df['label'].mean())
    contamination = max(0.001, min(contamination, 0.499))
    model = IsolationForest(n_estimators=100, contamination=contamination,
                            random_state=42, n_jobs=-1)
    t0 = time.time()
    model.fit(X_scaled)
    print(f"[IF Model] Done in {time.time()-t0:.1f}s | contamination={contamination:.4f}")
    return model, scaler


def compute_anomaly_scores(model, scaler, feature_df: pd.DataFrame,
                            feature_cols: list) -> np.ndarray:
    X        = feature_df[feature_cols].fillna(0).values
    X_scaled = scaler.transform(X)
    raw      = model.score_samples(X_scaled)
    mn, mx   = raw.min(), raw.max()
    return 1 - (raw - mn) / (mx - mn + 1e-9)


def compute_threat_scores(anomaly_scores: np.ndarray,
                           rule_results: list, alpha: float = ALPHA) -> np.ndarray:
    rule_scores = np.array([r['rule_score'] for r in rule_results])
    return alpha * rule_scores + (1 - alpha) * anomaly_scores


def evaluate(feature_df, threat_scores, anomaly_scores, rule_results,
             threshold=THRESHOLD) -> dict:
    y_true       = feature_df['label'].values
    y_pred_hybrid = (threat_scores >= threshold).astype(int)
    y_pred_if    = (anomaly_scores >= threshold).astype(int)
    rule_scores  = np.array([r['rule_score'] for r in rule_results])
    y_pred_rule  = (rule_scores > 0).astype(int)

    def metrics(y_t, y_p, name):
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
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

    return {
        'dataset':   'BGL',
        'total':     len(feature_df),
        'normal':    int((y_true == 0).sum()),
        'anomaly':   int((y_true == 1).sum()),
        'alpha':     ALPHA,
        'threshold': threshold,
        'ablation':  [
            metrics(y_true, y_pred_rule,   'Rule Only'),
            metrics(y_true, y_pred_if,     'Isolation Forest Only'),
            metrics(y_true, y_pred_hybrid, 'Hybrid (Rule + IF)'),
        ]
    }


def print_results(results: dict):
    print("\n" + "="*65)
    print(f"  BGL RESULTS — {results['total']:,} windows")
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
