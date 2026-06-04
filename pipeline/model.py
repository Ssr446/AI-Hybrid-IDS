"""
pipeline/model.py
Trains Isolation Forest on normal log features.
Evaluates with full metrics: Precision, Recall, F1, FPR.
Saves model to models/isolation_forest.pkl
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")


def train(X_normal: pd.DataFrame, contamination: float = 0.05):
    """Train Isolation Forest on normal-only feature vectors."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    print(f"[+] Model trained on {len(X_normal)} normal samples")
    print(f"[+] Saved to {MODEL_PATH}")
    return model, scaler


def load():
    """Load trained model and scaler."""
    with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
    return model, scaler


def predict(model, scaler, X: pd.DataFrame):
    """
    Returns anomaly scores and binary predictions.
    Isolation Forest: -1 = anomaly, 1 = normal
    We convert to: 1 = anomaly (attack), 0 = normal
    """
    X_scaled     = scaler.transform(X)
    raw_preds    = model.predict(X_scaled)           # -1 or 1
    scores       = model.decision_function(X_scaled) # lower = more anomalous
    norm_scores  = _normalize_scores(scores)         # 0–1, higher = more anomalous

    binary_preds = (raw_preds == -1).astype(int)     # 1 if anomaly
    return binary_preds, norm_scores


def evaluate(model, scaler, X: pd.DataFrame, y_true: np.ndarray):
    """Full evaluation with all research paper metrics."""
    start = time.perf_counter()
    y_pred, anomaly_scores = predict(model, scaler, X)
    elapsed_ms = (time.perf_counter() - start) * 1000

    # Core metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, cm[0][0])

    fpr = fp / max(fp + tn, 1)      # False Positive Rate
    fnr = fn / max(fn + tp, 1)      # False Negative Rate
    accuracy = (tp + tn) / max(len(y_true), 1)

    latency_per_log = elapsed_ms / max(len(X), 1)

    metrics = {
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1_score":          round(f1, 4),
        "accuracy":          round(accuracy, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "true_positives":    int(tp),
        "false_positives":   int(fp),
        "true_negatives":    int(tn),
        "false_negatives":   int(fn),
        "total_samples":     len(y_true),
        "attack_samples":    int(sum(y_true)),
        "normal_samples":    int(sum(y_true == 0)),
        "inference_ms_total":  round(elapsed_ms, 2),
        "inference_ms_per_log": round(latency_per_log, 4),
        "confusion_matrix":  cm.tolist(),
    }
    return metrics, y_pred, anomaly_scores


def ablation_study(X: pd.DataFrame, y_true: np.ndarray, rule_preds: np.ndarray):
    """
    Compare: rule-only vs IF-only vs hybrid
    Returns a summary dict.
    """
    model, scaler = load()
    if_preds, if_scores = predict(model, scaler, X)

    alpha = 0.5  # equal weight

    results = {}

    # Rule-only
    r_p = precision_score(y_true, rule_preds, zero_division=0)
    r_r = recall_score(y_true, rule_preds, zero_division=0)
    r_f = f1_score(y_true, rule_preds, zero_division=0)
    r_cm = confusion_matrix(y_true, rule_preds)
    r_tn, r_fp, r_fn, r_tp = r_cm.ravel() if r_cm.shape == (2,2) else (0,0,0,0)
    results["rule_only"] = {
        "precision": round(r_p, 4), "recall": round(r_r, 4),
        "f1": round(r_f, 4), "fpr": round(r_fp / max(r_fp+r_tn, 1), 4)
    }

    # IF-only
    i_p = precision_score(y_true, if_preds, zero_division=0)
    i_r = recall_score(y_true, if_preds, zero_division=0)
    i_f = f1_score(y_true, if_preds, zero_division=0)
    i_cm = confusion_matrix(y_true, if_preds)
    i_tn, i_fp, i_fn, i_tp = i_cm.ravel() if i_cm.shape == (2,2) else (0,0,0,0)
    results["isolation_forest"] = {
        "precision": round(i_p, 4), "recall": round(i_r, 4),
        "f1": round(i_f, 4), "fpr": round(i_fp / max(i_fp+i_tn, 1), 4)
    }

    # Hybrid (alpha weighted)
    hybrid_scores = alpha * rule_preds.astype(float) + (1 - alpha) * if_scores
    hybrid_preds  = (hybrid_scores >= 0.5).astype(int)
    h_p = precision_score(y_true, hybrid_preds, zero_division=0)
    h_r = recall_score(y_true, hybrid_preds, zero_division=0)
    h_f = f1_score(y_true, hybrid_preds, zero_division=0)
    h_cm = confusion_matrix(y_true, hybrid_preds)
    h_tn, h_fp, h_fn, h_tp = h_cm.ravel() if h_cm.shape == (2,2) else (0,0,0,0)
    results["hybrid"] = {
        "precision": round(h_p, 4), "recall": round(h_r, 4),
        "f1": round(h_f, 4), "fpr": round(h_fp / max(h_fp+h_tn, 1), 4)
    }

    return results


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize decision scores to 0–1 where 1 = most anomalous."""
    s_min, s_max = scores.min(), scores.max()
    if s_max == s_min:
        return np.zeros_like(scores)
    normalized = (scores - s_min) / (s_max - s_min)
    return 1.0 - normalized  # invert: lower decision score = higher anomaly
