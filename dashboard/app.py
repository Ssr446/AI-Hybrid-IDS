"""
dashboard/app.py
Real-time Streamlit dashboard for the AI Hybrid Vulnerability Detection System.
Run: streamlit run dashboard/app.py
"""

import os, sys, json
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import tempfile

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.parser import parse_file
from pipeline.features import get_combined_features
from pipeline.rules import evaluate_rules
import pipeline.model as pipe_model
from pipeline.hybrid import fuse

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Vulnerability Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0A0F1E; color: #B0C4D8; }
    .metric-card {
        background: #111D35; border: 1px solid #1E3A5F;
        border-radius: 8px; padding: 16px; text-align: center;
    }
    .metric-value { font-size: 2.2em; font-weight: bold; }
    .metric-label { font-size: 0.85em; color: #8899AA; margin-top: 4px; }
    .alert-critical { border-left: 4px solid #FF3D57; background: #1A0A15; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .alert-high     { border-left: 4px solid #FF9100; background: #1A1200; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .alert-medium   { border-left: 4px solid #FFD600; background: #1A1800; padding: 8px 12px; border-radius: 4px; margin: 4px 0; }
    .sidebar-header { color: #00D4FF; font-weight: bold; font-size: 1.1em; margin-bottom: 8px; }
    div[data-testid="stMetricValue"] { color: #00D4FF; }
</style>
""", unsafe_allow_html=True)

# ── LOAD RESULTS ──────────────────────────────────────────────────────────────
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "results.json")
SHAP_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "phase6_shap.json")
SHAP_PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")

@st.cache_data
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)

@st.cache_data
def load_shap_data():
    if not os.path.exists(SHAP_JSON_PATH):
        return None
    with open(SHAP_JSON_PATH) as f:
        return json.load(f)

results = load_results()
shap_data = load_shap_data()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-header">🛡️ AI Vulnerability Monitor</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio("Navigation", [
        "📊 Dashboard",
        "🚨 Live Alerts",
        "📈 Model Metrics",
        "🔬 System Comparison",
        "📋 Dataset Info",
        "🔍 Live Log Tester",
        "🧠 SHAP Explainability"
    ])

    st.markdown("---")
    if results:
        ds = results["dataset_stats"]
        st.markdown("**Dataset**")
        st.caption(f"Total logs: {ds['total_logs']:,}")
        st.caption(f"Attacks: {ds['attack_logs']:,}")
        st.caption(f"Normal: {ds['normal_logs']:,}")
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()

# ── NO RESULTS YET ────────────────────────────────────────────────────────────
if not results:
    st.error("⚠️ No results found. Please run `python train.py` first.")
    st.code("cd vulnerability_detection\npython train.py", language="bash")
    st.stop()

metrics  = results["model_metrics"]
ablation = results["ablation"]
alerts   = results["hybrid_alerts"]
ds       = results["dataset_stats"]

# ── PAGE: DASHBOARD ───────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("🛡️ AI-Assisted Hybrid Vulnerability Detection System")
    st.caption("Real-time log analysis using Rule Engine + Isolation Forest")
    st.markdown("---")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Log Events", f"{ds['total_logs']:,}")
    with col2:
        crit = sum(1 for a in alerts if a.get("severity") == "CRITICAL")
        st.metric("Critical Alerts", crit, delta=None)
    with col3:
        high = sum(1 for a in alerts if a.get("severity") == "HIGH")
        st.metric("High Alerts", high)
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
    with col5:
        st.metric("Avg Inference", f"{metrics['inference_ms_per_log']:.4f} ms")

    st.markdown("---")

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        # Severity distribution pie
        sev_counts = {}
        for a in alerts:
            s = a.get("severity", "UNKNOWN")
            sev_counts[s] = sev_counts.get(s, 0) + 1

        fig_pie = go.Figure(go.Pie(
            labels=list(sev_counts.keys()),
            values=list(sev_counts.values()),
            hole=0.45,
            marker_colors=["#FF3D57", "#FF9100", "#FFD600", "#00E676"],
        ))
        fig_pie.update_layout(
            title="Alert Severity Distribution",
            paper_bgcolor="#111D35", plot_bgcolor="#111D35",
            font_color="#B0C4D8", height=300,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        # Threat score histogram
        scores = [a.get("threat_score", 0) for a in alerts]
        fig_hist = go.Figure(go.Histogram(
            x=scores, nbinsx=20,
            marker_color="#00D4FF", opacity=0.8
        ))
        fig_hist.update_layout(
            title="ThreatScore Distribution",
            paper_bgcolor="#111D35", plot_bgcolor="#0A0F1E",
            font_color="#B0C4D8", height=300,
            xaxis_title="ThreatScore", yaxis_title="Count",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Detection methods bar chart
    method_counts = {}
    for a in alerts:
        m = a.get("detection_method", "Unknown")
        method_counts[m] = method_counts.get(m, 0) + 1

    fig_bar = go.Figure(go.Bar(
        x=list(method_counts.keys()),
        y=list(method_counts.values()),
        marker_color=["#00D4FF", "#7B2FFF", "#00E676"][:len(method_counts)],
    ))
    fig_bar.update_layout(
        title="Alerts by Detection Method",
        paper_bgcolor="#111D35", plot_bgcolor="#0A0F1E",
        font_color="#B0C4D8", height=260,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ── PAGE: LIVE ALERTS ─────────────────────────────────────────────────────────
elif page == "🚨 Live Alerts":
    st.title("🚨 Live Alert Feed")
    st.markdown("---")

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        sev_filter = st.multiselect(
            "Filter by Severity",
            ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
            default=["CRITICAL", "HIGH", "MEDIUM"]
        )
    with col2:
        method_filter = st.multiselect(
            "Filter by Detection",
            ["Rule + AI", "Isolation Forest", "rule"],
            default=["Rule + AI", "Isolation Forest", "rule"]
        )

    filtered = [
        a for a in alerts
        if a.get("severity") in sev_filter
        and a.get("detection_method") in method_filter
    ]

    st.caption(f"Showing {len(filtered)} alerts")

    # Alert table
    if filtered:
        df = pd.DataFrame([{
            "Rule ID":    a.get("rule_id", "-"),
            "Attack Type": a.get("rule_name", "-"),
            "Source IP":  a.get("source_ip", "-"),
            "Detection":  a.get("detection_method", "-"),
            "Rule Score": f"{a.get('rule_score', 0):.3f}",
            "AI Score":   f"{a.get('anomaly_score', 0):.3f}",
            "ThreatScore":f"{a.get('threat_score', 0):.3f}",
            "Severity":   a.get("severity", "-"),
        } for a in filtered])

        def color_severity(val):
            colors = {
                "CRITICAL": "background-color: #3D0010; color: #FF3D57; font-weight: bold",
                "HIGH":     "background-color: #2D1800; color: #FF9100; font-weight: bold",
                "MEDIUM":   "background-color: #2D2600; color: #FFD600; font-weight: bold",
                "LOW":      "background-color: #002D10; color: #00E676; font-weight: bold",
            }
            return colors.get(val, "")

        styled = df.style.applymap(color_severity, subset=["Severity"])
        st.dataframe(styled, use_container_width=True, height=500)

        # Expandable raw log for first CRITICAL
        crits = [a for a in filtered if a.get("severity") == "CRITICAL"]
        if crits:
            with st.expander("📋 Sample raw log — first CRITICAL alert"):
                a = crits[0]
                st.markdown(f"**{a.get('rule_name')}** from `{a.get('source_ip')}`")
                st.code(a.get("sample_raw", "N/A"), language="text")
                st.markdown(f"**Description:** {a.get('description', '')}")
                st.markdown(f"**ThreatScore:** `{a.get('threat_score', 0):.4f}` (α=0.55)")


# ── PAGE: MODEL METRICS ───────────────────────────────────────────────────────
elif page == "📈 Model Metrics":
    st.title("📈 Isolation Forest — Model Metrics")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision",  f"{metrics['precision']:.4f}")
    col2.metric("Recall",     f"{metrics['recall']:.4f}")
    col3.metric("F1-Score",   f"{metrics['f1_score']:.4f}")
    col4.metric("Accuracy",   f"{metrics['accuracy']:.4f}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("False Positive Rate", f"{metrics['false_positive_rate']:.4f}")
    col6.metric("False Negative Rate", f"{metrics['false_negative_rate']:.4f}")
    col7.metric("True Positives",  metrics['true_positives'])
    col8.metric("True Negatives",  metrics['true_negatives'])

    st.markdown("---")

    col_cm, col_bar = st.columns(2)

    with col_cm:
        # Confusion matrix heatmap
        cm = metrics["confusion_matrix"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=["Predicted Normal", "Predicted Attack"],
            y=["Actual Normal", "Actual Attack"],
            colorscale=[[0, "#0A0F1E"], [1, "#00D4FF"]],
            text=cm, texttemplate="%{text}",
            showscale=True,
        ))
        fig_cm.update_layout(
            title="Confusion Matrix",
            paper_bgcolor="#111D35", font_color="#B0C4D8",
            height=320, margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_bar:
        # Metrics radar
        metric_names = ["Precision", "Recall", "F1-Score", "Accuracy", "1 - FPR"]
        metric_vals  = [
            metrics["precision"], metrics["recall"], metrics["f1_score"],
            metrics["accuracy"],  1 - metrics["false_positive_rate"]
        ]
        fig_radar = go.Figure(go.Scatterpolar(
            r=metric_vals + [metric_vals[0]],
            theta=metric_names + [metric_names[0]],
            fill="toself",
            line_color="#00D4FF",
            fillcolor="rgba(0,212,255,0.15)",
            name="Hybrid System"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], color="#8899AA"),
                bgcolor="#0A0F1E"
            ),
            paper_bgcolor="#111D35", font_color="#B0C4D8",
            height=320, title="Performance Radar",
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown(f"""
    **Inference Performance**
    - Total inference time: `{metrics['inference_ms_total']:.2f} ms`
    - Per-log latency: `{metrics['inference_ms_per_log']:.4f} ms`
    - Total samples evaluated: `{metrics['total_samples']:,}`
    """)


# ── PAGE: ABLATION STUDY ──────────────────────────────────────────────────────
elif page == "🔬 System Comparison":
    st.title("🔬 System Comparison")
    st.caption("Rule-Only vs Isolation Forest-Only vs Hybrid")
    st.markdown("---")

    ab_data = []
    labels_map = {
        "rule_only":        "Rule Engine Only",
        "isolation_forest": "Isolation Forest Only",
        "hybrid":           "Hybrid (Rule + AI)",
    }
    for key, vals in ablation.items():
        ab_data.append({
            "Method":    labels_map.get(key, key),
            "Precision": vals["precision"],
            "Recall":    vals["recall"],
            "F1-Score":  vals["f1"],
            "FPR":       vals["fpr"],
        })

    df_ab = pd.DataFrame(ab_data)
    st.dataframe(
        df_ab.style.highlight_max(subset=["Precision","Recall","F1-Score"], color="#003D20")
                   .highlight_min(subset=["FPR"], color="#003D20")
                   .format({"Precision":"{:.4f}","Recall":"{:.4f}","F1-Score":"{:.4f}","FPR":"{:.4f}"}),
        use_container_width=True
    )

    st.markdown("---")

    metrics_to_plot = ["Precision", "Recall", "F1-Score"]
    fig_ab = go.Figure()
    colors = ["#00D4FF", "#7B2FFF", "#00E676"]
    for i, col in enumerate(metrics_to_plot):
        fig_ab.add_trace(go.Bar(
            name=col, x=df_ab["Method"], y=df_ab[col],
            marker_color=colors[i]
        ))
    fig_ab.update_layout(
        barmode="group", title="Ablation: Precision / Recall / F1 Comparison",
        paper_bgcolor="#111D35", plot_bgcolor="#0A0F1E",
        font_color="#B0C4D8", height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(range=[0, 1])
    )
    st.plotly_chart(fig_ab, use_container_width=True)

    st.info("✅ The Hybrid system achieves the best F1-Score while keeping FPR low — validating the core research contribution.")


# ── PAGE: DATASET INFO ────────────────────────────────────────────────────────
elif page == "📋 Dataset Info":
    st.title("📋 Dataset & Attack Distribution")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Log Entries", f"{ds.get('total_logs', 0):,}")
    col2.metric("Normal Entries", f"{ds.get('normal_logs', 0):,}")
    col3.metric("Attack Entries", f"{ds.get('attack_logs', 0):,}")

    st.markdown("---")

    attack_types = {}
    for a in results["rule_alerts"]:
        name = a.get("rule_name", "Unknown")
        attack_types[name] = attack_types.get(name, 0) + a.get("count", 1)

    if attack_types:
        df_attacks = pd.DataFrame(
            sorted(attack_types.items(), key=lambda x: -x[1]),
            columns=["Attack Type", "Event Count"]
        )
        fig_attacks = px.bar(
            df_attacks, x="Event Count", y="Attack Type",
            orientation="h", color="Event Count",
            color_continuous_scale=["#1E3A5F", "#00D4FF"]
        )
        fig_attacks.update_layout(
            title="Attack Events by Type",
            paper_bgcolor="#111D35", plot_bgcolor="#0A0F1E",
            font_color="#B0C4D8", height=400,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_attacks, use_container_width=True)

    st.markdown("""
    **Log Sources**
    - `data/auth.log` — Linux SSH authentication events (failed logins, sudo, sessions)
    - `data/apache.log` — HTTP access log (GET/POST requests, status codes, user agents)

    **Injected Attack Types**
    SSH Brute Force, User Enumeration, Privilege Escalation, Suspicious Root Login,
    SQL Injection, XSS, Directory Traversal, HTTP Flood, Scanner (Nikto/sqlmap)
    """)

# ── PAGE: LIVE LOG TESTER ─────────────────────────────────────────────────────
elif page == "🔍 Live Log Tester":
    st.title("🔍 Live Log Tester")
    st.markdown("Paste your raw system logs here and our pipeline will instantly parse, evaluate, and score them using your Hybrid Model.")
    
    log_type = st.radio("Log Type", ["ssh", "hdfs", "apache", "linux"], horizontal=True)
    
    raw_text = st.text_area("Raw Log Data", height=200, placeholder="Paste log lines here...")
    
    if st.button("Run Hybrid Analysis"):
        if not raw_text.strip():
            st.warning("Please paste some log data first.")
        else:
            with st.spinner("Analyzing logs..."):
                try:
                    # 1. Parse text via temporary file because parser natively reads from filepath
                    tf = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
                    tf.write(raw_text)
                    tf.close()
                    
                    parsed_logs = parse_file(tf.name, log_type)
                    os.unlink(tf.name)
                    
                    if not parsed_logs:
                        st.error("Failed to parse logs or unrecognized format.")
                    else:
                        # 2. Extract features
                        X = get_combined_features(parsed_logs)
                        
                        # 3. Rule Engine Evaluation
                        rule_alerts = evaluate_rules(parsed_logs)
                        rule_score = 0.0
                        if rule_alerts:
                            rule_score = max([a.get("rule_score", 0.0) for a in rule_alerts])
                        
                        # 4. Isolation Forest Evaluation
                        model_path = os.path.join(os.path.dirname(__file__), "..", "models", "isolation_forest.pkl")
                        if not os.path.exists(model_path):
                            st.warning("No Isolation Forest model found! Please ensure you have run the training pipeline first.")
                        else:
                            model, scaler = pipe_model.load()
                            _, ai_scores = pipe_model.predict(model, scaler, X)
                            avg_ai_score = float(np.mean(ai_scores))
                            
                            # 5. Math Fusion
                            alpha_weight = 0.55 if log_type == "ssh" else 0.40
                            hybrid_res = fuse(rule_score, avg_ai_score, alpha=alpha_weight)
                            final_score = hybrid_res["threat_score"]
                            severity = hybrid_res["severity"]
                            
                            # Streamlit Display
                            st.markdown("### 🛡️ Analysis Complete")
                            col1, col2, col3 = st.columns(3)
                            
                            # Define UI colors for severity
                            sev_color = "🟢"
                            if severity == "CRITICAL": sev_color = "🔴"
                            elif severity == "HIGH": sev_color = "🟠"
                            elif severity == "MEDIUM": sev_color = "🟡"
                            
                            col1.metric("Risk Severity", f"{sev_color} {severity}")
                            col2.metric("ThreatScore", f"{final_score:.2f}")
                            col3.metric("Rule Hits", len(rule_alerts))
                            
                            # Clamp value to prevent StreamlitAPIException (value > 1.0)
                            safe_score = min(1.0, max(0.0, float(final_score)))
                            st.progress(safe_score, text=f"Threat Level: {severity}")
                            
                            st.markdown("#### Diagnostic Values:")
                            st.write(f"- **Max Rule Score:** `{rule_score:.2f}`")
                            st.write(f"- **AI Context Anomaly:** `{avg_ai_score:.2f}`")
                            st.write(f"- **Parsed Events Evaluated:** `{len(parsed_logs)}`")
                            
                            if rule_alerts:
                                st.error("🚨 Signatures Triggered:")
                                for a in rule_alerts:
                                    st.write(f"- **[{a['rule_name']}]**: {a['description']}")
                            elif final_score >= 0.4:
                                st.warning("⚠️ No direct signatures triggered, but AI marked the aggregate sequence as structurally anomalous.")
                            else:
                                st.success("✅ Sequence is structurally clean.")
                                
                            # ── REAL SHAP EXPLAINABILITY (phase6_shap.py method) ──
                            if final_score > 0.1:
                                st.markdown("### 🧠 SHAP Explainability — Permutation Importance")
                                st.caption(
                                    "Feature importances computed via **permutation-based SHAP approximation** "
                                    "(same method as `phase6_shap.py`): each feature column is shuffled and "
                                    "the drop in Isolation Forest score is measured."
                                )
                                try:
                                    import sys as _sys
                                    import os as _os
                                    _root = _os.path.join(_os.path.dirname(__file__), "..")
                                    if _root not in _sys.path:
                                        _sys.path.insert(0, _root)
                                    from phase6_shap import compute_shap_approximation

                                    feat_cols = list(X.columns)
                                    X_np = X.fillna(0).values

                                    with st.spinner("Computing SHAP permutation importances…"):
                                        importances = compute_shap_approximation(
                                            model, scaler, X_np, feat_cols, n_samples=min(300, len(X_np))
                                        )

                                    # Build sorted dataframe — top 12 features
                                    shap_df = pd.DataFrame({
                                        "Feature": feat_cols,
                                        "SHAP Importance": importances
                                    }).sort_values("SHAP Importance", ascending=True).tail(12)

                                    fig_shap = go.Figure(go.Bar(
                                        x=shap_df["SHAP Importance"],
                                        y=shap_df["Feature"],
                                        orientation="h",
                                        marker=dict(
                                            color=shap_df["SHAP Importance"],
                                            colorscale=[[0, "#1E3A5F"], [0.5, "#7B2FFF"], [1, "#FF3D57"]],
                                            showscale=True,
                                            colorbar=dict(title="Importance", tickfont=dict(color="#B0C4D8"))
                                        ),
                                        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
                                    ))
                                    fig_shap.update_layout(
                                        title="SHAP Feature Importance (Permutation Method)",
                                        xaxis_title="Permutation Importance (normalised)",
                                        paper_bgcolor="#111D35", plot_bgcolor="#0A0F1E",
                                        font_color="#B0C4D8", height=420,
                                        margin=dict(l=10, r=10, t=40, b=10),
                                        xaxis=dict(gridcolor="#1E3A5F"),
                                        yaxis=dict(gridcolor="#1E3A5F")
                                    )
                                    st.plotly_chart(fig_shap, use_container_width=True)

                                    # Show top-1 explanation text
                                    top_idx = int(np.argmax(importances))
                                    st.info(
                                        f"🔑 **Most influential feature:** `{feat_cols[top_idx]}` "
                                        f"with normalised importance **{importances[top_idx]:.4f}**"
                                    )
                                except Exception as shap_err:
                                    import traceback
                                    st.warning(f"SHAP computation failed: {shap_err}\n```python\n{traceback.format_exc()}\n```")
                                
                except Exception as e:
                    import traceback
                    st.error(f"Error during execution: {e}")
                    st.code(traceback.format_exc(), language="python")

