"""
bgl_rules.py
Rule engine for BGL (BlueGene/L) log windows.
BGL anomalies: hardware faults, kernel FATAL events, APP failures, memory errors.
All rules use per-line matching to avoid cross-event false positives (lesson from HDFS).
"""

import re

RULES = [
    {
        'id':      'BGL-001',
        'name':    'Kernel FATAL Event',
        'pattern': re.compile(r'KERNEL\s+FATAL', re.IGNORECASE),
        'score':   1.0,
        'severity':'CRITICAL',
        'match':   'per_line',
    },
    {
        'id':      'BGL-002',
        'name':    'APP FATAL Failure',
        'pattern': re.compile(r'APP\s+FATAL', re.IGNORECASE),
        'score':   0.95,
        'severity':'CRITICAL',
        'match':   'per_line',
    },
    {
        'id':      'BGL-003',
        'name':    'ciod Process Failure',
        'pattern': re.compile(r'ciod:.*(?:fail|error|Error|cannot|No such)', re.IGNORECASE),
        'score':   0.9,
        'severity':'HIGH',
        'match':   'per_line',
    },
    {
        'id':      'BGL-004',
        'name':    'Data TLB / Storage Interrupt',
        'pattern': re.compile(r'(data\s+TLB\s+error|data\s+storage\s+interrupt)', re.IGNORECASE),
        'score':   0.95,
        'severity':'CRITICAL',
        'match':   'per_line',
    },
    {
        'id':        'BGL-005',
        'name':      'High Fatal Ratio in Window',
        'threshold': 0.3,    # >30% of lines are FATAL
        'score':     0.85,
        'severity':  'HIGH',
        'match':     'feature',
    },
    {
        'id':      'BGL-006',
        'name':    'Machine Check / Program Interrupt',
        'pattern': re.compile(r'(machine\s+check|program\s+interrupt|machine\s+state)', re.IGNORECASE),
        'score':   0.9,
        'severity':'CRITICAL',
        'match':   'per_line',
    },
    {
        'id':      'BGL-007',
        'name':    'Discovery SEVERE / WARNING',
        'pattern': re.compile(r'DISCOVERY\s+(SEVERE|WARNING)', re.IGNORECASE),
        'score':   0.7,
        'severity':'MEDIUM',
        'match':   'per_line',
    },
]


def evaluate_window(win_id: str, events: list, features: dict) -> dict:
    triggered = []
    n = len(events)

    for rule in RULES:
        rid   = rule['id']
        match = rule.get('match', 'per_line')

        if match == 'per_line':
            if any(rule['pattern'].search(e) for e in events):
                triggered.append(rule)

        elif match == 'feature':
            if rid == 'BGL-005':
                if features.get('fatal_ratio', 0) >= rule['threshold']:
                    triggered.append(rule)

    rule_score = max((r['score'] for r in triggered), default=0.0)
    return {
        'win_id':       win_id,
        'rule_score':   rule_score,
        'rules_hit':    [r['id'] for r in triggered],
        'rule_names':   [r['name'] for r in triggered],
        'top_severity': triggered[0]['severity'] if triggered else 'NORMAL',
    }


def evaluate_all_windows(windows: dict, feature_df) -> list:
    print(f"[BGL Rules] Evaluating {len(windows):,} windows...")
    results     = []
    feat_lookup = feature_df.set_index('win_id').to_dict('index') \
                  if 'win_id' in feature_df.columns else {}

    for i, (win_id, w) in enumerate(windows.items()):
        features = feat_lookup.get(win_id, {})
        result   = evaluate_window(win_id, w['events'], features)
        results.append(result)

        if (i + 1) % 50_000 == 0:
            print(f"  Evaluated {i+1:,} windows...")

    flagged = sum(1 for r in results if r['rule_score'] > 0)
    print(f"[BGL Rules] Done — {flagged:,} flagged ({flagged/max(len(results),1)*100:.1f}%)")
    return results
