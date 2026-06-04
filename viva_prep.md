# Viva Preparation & Intricate Project Details
**Project:** AI-Assisted Hybrid Vulnerability Detection System Using Log-Based Analysis

---

## 🧠 The Machine Learning Models Used
Examiners will specifically ask what ML models you implemented. Here is the exact breakdown:

### 1. Isolation Forest (The Core Anomaly Detector)
*   **What it is:** An Unsupervised Machine Learning algorithm based on decision trees. Unsupervised means it does *not* require labeled data (like "this is an attack, this is normal") to train.
*   **How it works:** 
    *   Unlike normal ML models that try to profile what "normal" data looks like, an Isolation Forest tries to isolate *abnormal* data.
    *   It creates a "forest" of random decision trees (our project uses 200 trees). 
    *   For each tree, it picks a random feature (e.g., `ip_fail_count`) and a random split value, over and over, until every point is isolated in its own leaf node.
    *   **The Math:** Anomalies are mathematically rare and different. Therefore, it takes *very few splits* to isolate them. Normal data points are dense and similar, so they require a massive number of splits to separate. 
    *   **The Output:** It calculates an Anomaly Score based on the **Average Path Length** across all 200 trees. Short paths = Anomaly. Deep paths = Normal.

### 2. SHAP (The Explainability Model)
*   **What it is:** SHAP (SHapley Additive exPlanations) is a game-theoretic model used to interpret and explain the outputs of "black-box" machine learning models (like our Isolation Forest).
*   **How it works:** 
    *   It treats the ML prediction as a "game" where the features (like `is_external_ip`, `request_burst_score`) are the "players". 
    *   It uses **Permutation Feature Importance**. It essentially asks the Isolation Forest: *"What would the anomaly score be if I completely randomized the `ip_fail_count` column, while keeping everything else the same?"*
    *   If randomizing a feature causes the anomaly score to drastically drop, SHAP knows that feature was highly important for that specific prediction. It then assigns a mathematical weight (importance score) to every feature, which is what we display on our Dashboard's waterfall charts.

---

## 🔍 Intricate Project Details (Other Pipeline Components)

### 1. Log Parsing (The Drain3 Algorithm)
*   **How it works:** Instead of hardcoding Regular Expressions (Regex) for every possible log format, **Drain3** uses a fixed-depth **Prefix Tree (Trie)**. It first groups logs by token length, then walks the tree to match starting words. If there isn’t a match, it dynamically creates a new template.
*   **Why it's smart:** It automatically recognizes and masks variables into tokens (e.g., `<IP>`, `<NUM>`, `<HEX>`). This converts messy raw strings into structured IDs, making ML processing possible.

### 2. Feature Engineering (The 26-Dimensional Vector)
*   Instead of just giving the raw log to the ML model, we use a **Sliding Window** to group logs. 
*   We extract mathematically complex features from SSH logs:
    *   `ip_fail_norm`: The normalized frequency of failed authentication attempts from a specific IP relative to the total block size.
    *   `ip_fail_count`: The total absolute counting of failed attempts in a burst window.
    *   `is_invalid_user`: Tracks specific attacks like User Enumeration where bots attempt fake usernames instead of valid system accounts.

### 3. The Fusion Equation (The Secret Sauce)
*   **Equation:** `ThreatScore = α × RuleScore + (1−α) × AnomalyScore`
*   Our extensive grid search proved that `α = 0.55` is the optimal mathematical sweet-spot. 
*   **Why?** If the Rule Engine spots an attack (RuleScore = 1), it mathematically overwrites a low AI score, catching known threats with perfect precision. If the Rule Engine misses it (RuleScore = 0), a massive AnomalyScore from a zero-day attack will still push the ThreatScore above the threshold.

---

## 👨‍🏫 Top 6 Expected Viva Questions & Answers

**Q1: What exact Machine Learning models are you using?**
**Answer:** The primary machine learning model is an **Isolation Forest**, which is an unsupervised anomaly detection algorithm. We also utilize **SHAP** (SHapley Additive exPlanations) as an auxiliary machine learning explainability model to interpret the Isolation Forest's outputs.

**Q2: Why didn't you just use Deep Learning architectures like LSTM, DeepLog, or LogBERT?**
**Answer:** Three reasons:
1.  **Latency & Compute:** Deep learning requires GPUs and averages 0.1ms to 1.0ms latency per log. Our Isolation Forest averages **0.0101ms** on a standard CPU—11 times faster.
2.  **Explainability:** Deep learning is a black box. A SOC analyst cannot trust an alert if they don't know *why* it was triggered. Our SHAP implementation explains every single alert.
3.  **Label Independence:** Deep learning requires massive amounts of labeled attack data to train. Isolation Forest is unsupervised—it learns what "normal" is and flags anything different. 

**Q3: How is your "Hybrid Model" different from an ensemble of models?**
**Answer:** An ensemble usually strings multiple Machine Learning models together and takes an average or majority vote. Our hybrid approach uniquely fuses a **deterministic rule-engine** (like Snort) with an **unsupervised AI model** via a mathematical weighting alpha parameter (`α=0.55`). This eliminates the strict rigidness of rule-bases while suppressing the high false-positive rate of pure ML models.

**Q4: How does Drain3 parsing work, and why is it better than Regex?**
**Answer:** Regex is rigid—if a log format changes by one space or comma, regex breaks. It demands permanent manual maintenance. Drain3 is an online, dynamic algorithm. It uses a fixed-depth prefix tree to cluster logically similar logs instantly, masking variables out without needing to strictly know the format ahead of time. It's built to generalize.

**Q5: What happens if an attacker targets your system with an attack it has never seen before (Zero-Day)?**
**Answer:** The Rule Engine will fail to see it and return a score of 0, because it has no signature. However, the attack behavior will mathematically deviate from the normal execution patterns we profiled. The Isolation Forest will flag the high deviation, outputting a high AnomalyScore. Because we use the `(1-α)` fusion weight, this will push the total ThreatScore past our alerting threshold. We catch the Zero-day without needing a signature.

**Q6: What is the significance of the 0.0003 False Positive Rate?**
**Answer:** Alert fatigue is the primary reason why SOCs fail. The baseline AI model (Isolation Forest operating alone) had a False Positive rate of **17.97%**, which would equate to roughly 90,000 irrelevant false alarms a day in an enterprise setting. By fusing the Rule Engine to strictly suppress uncertain AI assumptions, we dropped the FPR to **0.0003%**, practically eliminating alert fatigue completely while retaining a 0.9654 F1 Score.
