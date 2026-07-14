<div align="center">
  <h1>🛡️ AI-Assisted Hybrid Vulnerability Detection System</h1>
  <p><strong>A Real-Time, Explainable Threat Detection Engine for Security Operations Centers (SOCs)</strong></p>
  
  [![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://ai-hybrid-ids-mq6frlreyokd3wdtsnncy5.streamlit.app/)
  
  <br />
  <!-- Drop your dashboard screenshot inside the assets/ folder and name it dashboard_screenshot.png -->
  <img src="assets/dashboard_screenshot.png" alt="Dashboard Screenshot" width="800"/>
</div>

<br />

## 📖 Research Context & Overview

### The Base Paper & Problem
Recent state-of-the-art anomaly detection systems in cybersecurity heavily rely on Deep Learning models like **DeepLog** (LSTM) or **LogBERT**. While highly effective at identifying zero-day anomalies, these base architectures suffer from two critical flaws when deployed in real-world Security Operations Centers (SOCs):
1. **The "Black Box" Problem:** Analysts are given an anomaly score but no explanation as to *why* the sequence was flagged.
2. **Alert Fatigue:** Pure unsupervised learning over low-context logs results in prohibitively high False Positive Rates (FPR).

### Our Novel Addition
This project completely rearchitects the intrusion detection pipeline. Instead of relying purely on deep learning, we propose an **Optimized Hybrid Pipeline**. It runs in parallel:
- A deterministic **Cybersecurity Rule Engine** (12 signatures mimicking tools like Snort).
- An unsupervised **Isolation Forest (AI)** array.

Furthermore, we solve the Black Box problem by integrating **SHAP (Shapley Additive exPlanations)** directly into the real-time inference pipeline, allowing the system to mathematically reverse-engineer its own AI decisions and present them visually to security analysts.

### The Hybrid Fusion Formula
To bridge deterministic security rules with probabilistic AI, we developed a weighted scalar fusion layer. The outputs of both parallel engines are normalized and combined using the following formula:

> **$\text{ThreatScore} = \alpha \times \text{RuleScore} + (1 - \alpha) \times \text{AnomalyScore}$**

Where `α` (Alpha) is a hyperparameter determining the trust weight assigned to hard rules versus AI intuition (calibrated to `0.55` for OpenSSH).

---

## 🏗️ System Architecture

The pipeline processes raw system logs through a 5-layer architecture before rendering them in a real-time SOC dashboard.

```mermaid
graph TD
    A[Raw Log Input] --> B[Drain3 Parser]
    B --> C[Feature Extractor<br/>26-Dimensional Vector]
    
    C --> D[Parallel Engine Layer]
    
    subgraph D [Parallel Extractor]
        E[Rule Engine<br/>12 Cybersecurity Signatures]
        F[Isolation Forest<br/>Unsupervised ML Anomaly]
    end
    
    E --> G[Hybrid Fusion Layer<br/>ThreatScore = α * RuleScore + <br/>1-α * AnomalyScore]
    F --> G
    
    G --> H[SHAP Explainer<br/>Feature Attribution]
    H --> I[Streamlit SOC Dashboard<br/>Real-Time UI Monitoring]
    
    classDef default fill:#f9f9fa,stroke:#d0d7de,stroke-width:2px,color:#24292f,rx:6px,ry:6px;
    classDef main fill:#ebf5ff,stroke:#0969da,stroke-width:2px,color:#0969da,rx:6px,ry:6px;
    classDef highlight fill:#34d058,stroke:#28a745,color:#ffffff,rx:6px,ry:6px;
    classDef alert fill:#fa4549,stroke:#d73a49,color:#ffffff,rx:6px,ry:6px;
    classDef dark fill:#24292f,stroke:#1b1f23,color:#ffffff,rx:6px,ry:6px;
    
    class A,B,C default;
    class E,F main;
    class G alert;
    class H highlight;
    class I dark;
```

---

## 🚀 Features

1. **Live Log Tester UI:** Paste raw SSH, Apache, HDFS, or Linux logs into the dashboard to parse and score them in real-time.
2. **SHAP Waterfall Charts:** Instantly see which mathematical features (e.g., `ip_fail_count`, `is_invalid_user`) contributed most to the AI's anomaly score.
3. **Multi-Dataset Support:** Pre-configured extraction pipelines for OpenSSH, HDFS, BGL, and Thunderbird supercomputer logs.

<!-- Drop your SHAP chart inside the assets/ folder and name it shap_chart.png -->
<div align="center">
  <img src="assets/shap_chart.png" alt="SHAP Explainability Chart" width="600"/>
</div>

4. **Automated Research Pipelines:** Includes scripts for 5-Fold Cross Validation, Alpha Parameter Sensitivity sweeping, and comparative ablation against LSTM baselines.

---

## 💻 Local Installation & Setup

Ensure you have Python 3.9+ installed on your system. 

**1. Clone the repository**
```bash
git clone https://github.com/Ssr446/AI-Hybrid-IDS.git
cd AI-Hybrid-IDS
```

**2. Create a virtual environment**
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Mac/Linux:
source .venv/bin/activate
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Train the ML Model**
Before running the dashboard, you must train the Isolation Forest model on normal behavior.
```bash
python train.py
```

**5. Launch the SOC Dashboard**
```bash
streamlit run dashboard/app.py
```
*Navigate to `http://localhost:8501` to view the active defense dashboard!*

---

## 🧪 Running the Evaluation Pipelines

To reproduce the research metrics, cross-validations, and ablation studies, run the master orchestrator:

```bash
python run_all_phases.py
```
This will automatically generate high-resolution PNG charts in the `results/plots/` directory and compile a CSV table comparing this hybrid approach against DeepLog and LogBERT.

---

## 🛠️ Built With

*   **[Streamlit](https://streamlit.io/)** - For the interactive SOC Dashboard.
*   **[Scikit-Learn](https://scikit-learn.org/)** - For the Isolation Forest implementation.
*   **[Drain3](https://github.com/logpai/Drain3)** - For automated log templating and parsing.
*   **[Plotly](https://plotly.com/) & [Seaborn](https://seaborn.pydata.org/)** - For real-time explainability visualizations and metric plots.

---
*Developed as a B.Tech Minor Project.*
