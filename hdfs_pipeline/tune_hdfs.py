"""
tune_hdfs.py — Find optimal alpha and threshold for HDFS hybrid model.
Sweeps alpha [0.5, 0.6, 0.7, 0.8, 0.9] and threshold [0.1, 0.2, 0.3, 0.4]
and reports the best F1 combination.
"""
import sys, json, os, time
import numpy as np
import pandas as pd
sys.path.insert(0, '.')

from hdfs_parser   import parse_hdfs_log, load_labels, group_by_block
from hdfs_features import build_feature_matrix, get_feature_columns
from hdfs_rules    import evaluate_all_blocks
from hdfs_model    import train_isolation_forest, compute_anomaly_scores
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

HDFS_LOG_PATH = r"C:\Users\ssrsh\Documents\projects\minor project\HDFS.log"
LABEL_PATH    = r"C:\Users\ssrsh\Documents\projects\minor project\data\anomaly_label.csv"

print("Loading data...")
t0 = time.time()
df           = parse_hdfs_log(HDFS_LOG_PATH)
label_map    = load_labels(LABEL_PATH)
block_groups = group_by_block(df)
feature_df   = build_feature_matrix(block_groups, label_map)
feat_cols    = get_feature_columns(feature_df)

rule_results   = evaluate_all_blocks(block_groups, feature_df)
model, scaler  = train_isolation_forest(feature_df, feat_cols)

X = feature_df[feat_cols].fillna(0).values
X_scaled = scaler.transform(X)
raw_scores = model.score_samples(X_scaled)
min_s, max_s = raw_scores.min(), raw_scores.max()
anomaly_scores = 1 - (raw_scores - min_s) / (max_s - min_s + 1e-9)

rule_scores = np.array([r['rule_score'] for r in rule_results])
y_true = feature_df['label'].values

print(f"\nData loaded in {time.time()-t0:.0f}s. Sweeping alpha x threshold...\n")
print(f"{'Alpha':>6} {'Thresh':>6} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FPR':>7}")
print("-" * 50)

best = {'f1': 0}
alphas     = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

for alpha in alphas:
    for thresh in thresholds:
        threat = alpha * rule_scores + (1 - alpha) * anomaly_scores
        y_pred = (threat >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        fpr = fp / (fp + tn) if (fp+tn) > 0 else 0
        p   = precision_score(y_true, y_pred, zero_division=0)
        r   = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best['f1']:
            best = {'alpha': alpha, 'threshold': thresh, 'f1': f1,
                    'precision': p, 'recall': r, 'fpr': fpr,
                    'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}

print(f"\n{'='*50}")
print(f"BEST: alpha={best['alpha']}, threshold={best['threshold']}")
print(f"  Precision={best['precision']:.4f}  Recall={best['recall']:.4f}")
print(f"  F1={best['f1']:.4f}  FPR={best['fpr']:.4f}")
print(f"  TP={best['tp']}  FP={best['fp']}  TN={best['tn']}  FN={best['fn']}")

# Save best config
out = {
    'best_alpha': best['alpha'],
    'best_threshold': best['threshold'],
    'best_f1': round(best['f1'], 4),
    'best_precision': round(best['precision'], 4),
    'best_recall': round(best['recall'], 4),
    'best_fpr': round(best['fpr'], 4),
    'confusion': {'tp': best['tp'], 'fp': best['fp'],
                  'tn': best['tn'], 'fn': best['fn']},
}
os.makedirs(r"C:\Users\ssrsh\Documents\projects\minor project\minor2\results", exist_ok=True)
with open(r"C:\Users\ssrsh\Documents\projects\minor project\minor2\results\hdfs_tuning.json", 'w') as f:
    json.dump(out, f, indent=2)
print("\nSaved to hdfs_tuning.json")
