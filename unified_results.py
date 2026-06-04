"""
unified_results.py
Generates the final comparison table across all datasets for the journal publication.
"""

import json
import pandas as pd
from pathlib import Path

# Paths to the result JSON files
RESULTS_DIR = Path(r"C:\Users\ssrsh\Documents\projects\minor project\minor2\results")

DATASETS = [
    {'name': 'OpenSSH', 'file': 'ssh_results.json'},
    {'name': 'HDFS',    'file': 'hdfs_results.json'},
    {'name': 'BGL',     'file': 'bgl_results.json'},
    {'name': 'Thunderbird', 'file': 'thunderbird_results.json'}
]

# Hardcoded results for those evaluated before (if JSONs are missing locally in minor2/results)
FALLBACK = {
    'OpenSSH': {'f1': 0.9654, 'prec': 0.9997, 'rec': 0.9333, 'fpr': 0.0003, 'lat': 0.0101},
    'HDFS':    {'f1': 0.5281, 'prec': 0.8149, 'rec': 0.3900, 'fpr': 0.0001, 'lat': 0.0150},
}


def load_result(dataset):
    path = RESULTS_DIR / dataset['file']
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                best = data['ablation'][2]  # Hybrid is index 2
                return {
                    'Dataset': dataset['name'],
                    'Method': 'Hybrid (Rule + IF)',
                    'F1-Score': best['f1'],
                    'Precision': best['precision'],
                    'Recall': best['recall'],
                    'FPR': best['fpr'],
                    'Latency (ms)': data.get('latency_ms_per_window', 'N/A')
                }
        except Exception as e:
            print(f"Error reading {path}: {e}")
            
    if dataset['name'] in FALLBACK:
        best = FALLBACK[dataset['name']]
        return {
            'Dataset': dataset['name'],
            'Method': 'Hybrid (Rule + IF)',
            'F1-Score': best['f1'],
            'Precision': best['prec'],
            'Recall': best['rec'],
            'FPR': best['fpr'],
            'Latency (ms)': best['lat']
        }
    return None


def main():
    print("\n" + "="*70)
    print("  PHASE 2 UNIFIED COMPARISON TABLE (JOURNAL PUBLICATION)")
    print("="*70)
    
    rows = []
    for ds in DATASETS:
        res = load_result(ds)
        if res:
            rows.append(res)
        else:
            rows.append({
                'Dataset': ds['name'],
                'Method': 'Pending',
                'F1-Score': '---',
                'Precision': '---',
                'Recall': '---',
                'FPR': '---',
                'Latency (ms)': '---'
            })
            
    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False) + "\n")
    print("="*70)
    
    csv_path = RESULTS_DIR / "journal_comparison_table.csv"
    RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved tabular results to {csv_path}\n")


if __name__ == '__main__':
    main()
