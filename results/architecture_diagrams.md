# System Architecture and Flow Diagrams

> **Note:** If you are viewing this in VS Code, right-click and "Open Preview" or use the Mermaid Markdown extension. You can also copy and paste these blocks into 👉 [Mermaid Live Editor](https://mermaid.live/) to instantly download high-quality PNGs or SVGs to paste into your Word documents!

***

### Fig 3.1: System Architecture Block Diagram (5-Layer Pipeline)

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

***

### Fig 4.1: Internal Execution Sequence Diagram

```mermaid
sequenceDiagram
    participant LS as Log Stream
    participant D3 as Drain3 Parser
    participant FE as Feature Eng.
    participant RE as Rule Engine
    participant IF as Isolation Forest
    participant FL as Fusion Layer
    participant XAI as SHAP
    participant DB as SOC Dashboard
    
    LS->>D3: Raw Log Event
    D3->>FE: Structured Template String + Meta
    FE->>RE: 26-Dim Feature Vector
    FE->>IF: 26-Dim Feature Vector
    
    par Parallel Execution
        RE-->>FL: Rule Score [0-1] & Triggered
        IF-->>FL: Anomaly Score [0-1] (Min-Max)
    end
    
    FL->>FL: Compute: ThreatScore = α(Rule) + (1-α)(IF)
    FL->>XAI: Condition: Priority Alert Triggered
    
    XAI-->>DB: Map Contribution (e.g. ip_fail_count)
    FL-->>DB: Deliver Final Alert JSON Payload
    
    Note over DB: Analyst reviews UI<br/>(Metrics & Causes)
    DB->>DB: Render Severity & SHAP Waterfall Bar
```
