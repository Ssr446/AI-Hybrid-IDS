"""
phase6_shap.py
===============
SHAP feature importance analysis for BGL and HDFS.
Generates waterfall + summary bar charts for the journal.

Run: python phase6_shap.py
"""
import numpy as np
import pandas as pd
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT    = Path(r"C:\Users\ssrsh\Documents\projects\minor project\minor2")
RESULTS = ROOT / "results"
PLOTS   = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bgl_pipeline"))
sys.path.insert(0, str(ROOT / "hdfs_pipeline"))


def compute_shap_approximation(model, scaler, X, feat_cols, n_samples=500):
    """
    Approximate SHAP values using permutation importance
    (full SHAP requires the shap package; this is a dependency-free fallback).
    """
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    X_s = X[idx]
    X_sc = scaler.transform(X_s)
    baseline = model.score_samples(X_sc).mean()

    importances = []
    for j in range(X_s.shape[1]):
        X_perm     = X_sc.copy()
        X_perm[:, j] = rng.permutation(X_perm[:, j])
        perturbed  = model.score_samples(X_perm).mean()
        importances.append(abs(baseline - perturbed))

    importances = np.array(importances)
    importances = importances / (importances.sum() + 1e-9)
    return importances


def plot_shap_bar(importances, feat_cols, title, save_path, color='#4F86F7'):
    order = np.argsort(importances)[-15:]  # top-15
    imp_s = importances[order]
    labs  = [feat_cols[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#0D1117')
    ax.set_facecolor('#161B22')
    bars = ax.barh(labs, imp_s, color=color, height=0.6)
    for bar, val in zip(bars, imp_s):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', color='white', fontsize=8)
    ax.set_xlabel('Permutation Importance (normalized)', color='#8B949E', fontsize=10)
    ax.set_title(title, color='white', fontsize=12, pad=12)
    ax.tick_params(colors='#8B949E', labelsize=8)
    ax.spines[:].set_color('#30363D')
    ax.grid(axis='x', color='#21262D', linestyle='--', linewidth=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print(f"[SHAP] Saved: {save_path}")


def main():
    print("\n[Phase 6] SHAP Feature Importance Analysis\n")

    shap_results = {}

    # ── BGL ──────────────────────────────────────────────────────────────────
    try:
        from bgl_parser   import parse_bgl_log, make_windows
        from bgl_features import build_feature_matrix, get_feature_columns
        from bgl_rules    import evaluate_all_windows
        from bgl_model    import train_isolation_forest

        df      = parse_bgl_log(r"C:\Users\ssrsh\Documents\projects\minor project\BGL.log")
        windows = make_windows(df)
        feat_df = build_feature_matrix(windows)
        cols    = get_feature_columns(feat_df)
        model, scaler = train_isolation_forest(feat_df, cols)
        X = feat_df[cols].fillna(0).values

        imps = compute_shap_approximation(model, scaler, X, cols)
        shap_results['BGL'] = dict(zip(cols, imps.tolist()))
        plot_shap_bar(imps, cols, 'BGL — Feature Importance (SHAP Approximation)',
                      PLOTS / "shap_bgl.png", color='#6BCB77')
        print(f"  BGL top feature: {cols[np.argmax(imps)]} ({imps.max():.3f})")
    except Exception as e:
        print(f"[SHAP BGL] Error: {e}")

    # ── HDFS ─────────────────────────────────────────────────────────────────
    try:
        from hdfs_parser   import parse_hdfs_log, group_by_block, load_labels
        from hdfs_features import build_feature_matrix as hbfm, get_feature_columns as hgfc
        from hdfs_model    import train_isolation_forest as htif

        label_map = load_labels(r"C:\Users\ssrsh\Documents\projects\minor project\data\anomaly_label.csv")
        df2   = parse_hdfs_log(r"C:\Users\ssrsh\Documents\projects\minor project\HDFS.log")
        blocks= group_by_block(df2)
        feat2 = hbfm(blocks, label_map)
        cols2 = hgfc(feat2)
        model2, scaler2 = htif(feat2, cols2)
        X2 = feat2[cols2].fillna(0).values

        imps2 = compute_shap_approximation(model2, scaler2, X2, cols2)
        shap_results['HDFS'] = dict(zip(cols2, imps2.tolist()))
        plot_shap_bar(imps2, cols2, 'HDFS — Feature Importance (SHAP Approximation)',
                      PLOTS / "shap_hdfs.png", color='#FF6B6B')
        print(f"  HDFS top feature: {cols2[np.argmax(imps2)]} ({imps2.max():.3f})")
    except Exception as e:
        print(f"[SHAP HDFS] Error: {e}")

    out = RESULTS / "phase6_shap.json"
    with open(out, 'w') as f:
        json.dump(shap_results, f, indent=2)
    print(f"\n[Phase 6] Done. Saved: {out}")


if __name__ == '__main__':
    main()
