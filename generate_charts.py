import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create plots dir if it doesn't exist
os.makedirs("results/plots", exist_ok=True)

# 1. Confusion Matrix (Fig 6.1)
def plot_confusion_matrix():
    # TP=288,893, FP=104, TN=345,561, FN=20,589
    cm = np.array([[345561, 104],
                   [20589, 288893]])
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', 
                     xticklabels=['Normal', 'Anomaly'],
                     yticklabels=['Normal', 'Anomaly'],
                     annot_kws={"size": 16})
    plt.title('Fig 6.1: Confusion Matrix - Hybrid SSH', fontsize=18)
    plt.ylabel('Actual Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig('results/plots/confusion_matrix_fig6_1.png', dpi=300)
    plt.close()

# 2. SHAP Feature Attribution (Fig 3.2)
def plot_shap_waterfall():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We will use a standard horizontal bar chart to represent the SHAP values 
    # as described in the prompt.
    labels = ['ip_fail_count', 'is_external_ip', 'request_burst_score', 'time_of_day_deviation', 'unique_users_targeted']
    shap_vals = [0.42, 0.31, 0.18, 0.09, 0.06]
    
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, shap_vals, align='center', color='#d62728')
    
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('+ Mean |SHAP Value| (Impact on ThreatScore)')
    ax.set_title('Fig 3.2: SHAP Feature Importance for HIGH Severity SSH Alert')
    
    for bar, val in zip(bars, shap_vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'+{val:.2f}', 
                va='center', ha='left', color='black', fontweight='bold')

    plt.tight_layout()
    plt.savefig('results/plots/shap_feature_importance_fig3_2.png', dpi=300)
    plt.close()

# 3. Model Metrics Chart
def plot_model_metrics():
    # Comparing against baseline
    models = ['DeepLog', 'LogBERT', 'Our Hybrid System']
    precision = [0.913, 0.955, 0.9997]
    f1_score = [0.913, 0.955, 0.9654]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, precision, width, label='Precision', color='#1f77b4')
    rects2 = ax.bar(x + width/2, f1_score, width, label='F1-Score', color='#ff7f0e')
    
    ax.set_ylabel('Scores')
    ax.set_title('System Performance Comparison vs Baselines')
    ax.set_xticks(x, models)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('results/plots/model_metrics_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_confusion_matrix()
    plot_shap_waterfall()
    plot_model_metrics()
    print("Successfully generated Data Plots in results/plots/")
