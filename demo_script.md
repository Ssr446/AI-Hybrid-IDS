# Live-Demo Focused Script: AI-Assisted Hybrid Vulnerability Detection System

**Team Members:** Sai Shivaram, Archana, Jeyanirudh
**Style:** Focused entirely on the live software demonstration.
**Setup:** Have the Streamlit Dashboard running (`streamlit run dashboard/app.py`). Keep `demo_sample_logs.txt` open on the side so you can easily copy/paste the 3 scenarios.

---

## 🎤 Part 1: The Setup & The Problem Definition
**Speaker:** Sai Shivaram 
**Action:** Have the dashboard open on the **"System Status"** or **"Project Overview"** tab.

**[Opening]**
"Good Morning/Afternoon. We are Sai, Archana, and Jeyanirudh. Instead of walking through traditional slides, we are going to dive straight into a live, interactive demonstration of our minor project: an **AI-Assisted Hybrid Vulnerability Detection System**."

**[The Problem being Solved]**
"Modern Security Operations Centers face a massive crisis: **Alert Fatigue**. Analysts are bombarded with false alarms from AI systems, or they use strictly rule-based systems that completely miss stealthy, zero-day attacks. Our system solves this by physically fusing a deterministic Rule Engine with an unsupervised AI model (an Isolation Forest) to get the best of both worlds—high precision and high recall. 

Let's open the Live Tester and see it handle different attack scenarios."

---

## 🎤 Part 2: Live Demo Phase 1 — Normal Traffic vs Brute Force
**Speaker:** Archana
**Action:** Click over to the **"Live Log Tester"** tab. 

**[Scenario 1: Normal Traffic]**
*(Action: Copy inside `demo_sample_logs.txt` -> Paste the "Normal Healthy Traffic" block -> Hit Analyze)*
"First, a vital requirement for an IDS is not to block legitimate users. I am pasting in a block of normal, successful admin logins. As it analyzes, you can see the Threat Score remains practically zero. The dashboard confirms this is healthy traffic."

**[Scenario 2: Brute Force Attack]**
*(Action: Clear the box -> Paste the "Brute Force Attack" block -> Hit Analyze)*
"Now, let's simulate a hostile attack. I am pasting a barrage of failed login attempts. Immediately, the dashboard flags this as a **CRITICAL** threat. 

While that ran instantly, under the hood three things just happened:
1. **Dynamic Parsing:** A Drain3 algorithm stripped away IPs and timestamps, logically grouping the text.
2. **Feature Extraction:** It calculated a 26-dimensional mathematical vector measuring burst velocity and failure counts.
3. **The Hybrid Fusion:** It ran these features simultaneously through our 12-signature Rule Engine AND our Isolation Forest, fusing them via our mathematical Threat Score formula."

---

## 🎤 Part 3: Live Demo Phase 2 — Advanced Attacks & Explainability
**Speaker:** Jeyanirudh

**[Scenario 3: User Enumeration Attack]**
*(Action: Clear the box -> Paste the "User Enumeration Attack" block -> Hit Analyze)*
"A simple brute force is easy to catch, but what about stealthier attacks? I am pasting an advanced User Enumeration attack, where a bot slowly tries to guess different existing usernames. 

As you can see, our system still catches it. But more importantly, we need to know *why*."

**[The Explainability Factor]**
*(Action: Scroll down to specifically show the **SHAP Waterfall/Bar Chart** on the screen)*
"Traditional Deep Learning models like DeepLog are black boxes; they just give a score. Our system explains exactly *why* the alert fired. Looking right here at our **SHAP Feature Importance Chart**, you can mathematically see that the `is_invalid_user` and `ip_fail_norm` features are exactly what pushed the threat score into the red zone. A SOC analyst instantly knows the exact nature of the attack."

**[The Final Numbers]**
"To conclude, we evaluated this hybrid pipeline on over 650,000 logs:
* By fusing ML with rules, we crushed the False Positive Rate down to **0.0003%**—virtually eliminating alert fatigue.
* And because we run on an Isolation Forest rather than demanding deep neural networks, our system processes logs at **0.0101 milliseconds per log on a standard CPU**. That is **11 times faster** than peer deep learning models, without using a single GPU.

Thank you, we are now ready to answer any questions."
