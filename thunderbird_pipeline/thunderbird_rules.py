"""
thunderbird_rules.py
Rule engine for Thunderbird log windows.
All rules use per-line matching to avoid cross-event false positives.
"""

import re

RULES = [
    {
        'id':      'TB-001',
        'name':    'Hardware FATAL',
        'pattern': re.compile(r'FATAL', re.IGNORECASE),
        'score':   1.0,
        'severity':'CRITICAL',
        'match':   'per_line',
    },
    {
        'id':      'TB-002',
        'name':    'Memory Parity/ECC Error',
        'pattern': re.compile(r'(parity\s+error|ecc\s+error|uncorrectable)', re.IGNORECASE),
        'score':   0.95,
        'severity':'CRITICAL',
        'match':   'per_line',
    },
    {
        'id':      'TB-003',
        'name':    'Link Failure',
        'pattern': re.compile(r'link\s+(down|fail|error|lost)', re.IGNORECASE),
        'score':   0.9,
        'severity':'HIGH',
        'match':   'per_line',
    },
    {
        'id':      'TB-004',
        'name':    'Switch/Fabric Error',
        'pattern': re.compile(r'ib_sm.*(error|fail|cannot)', re.IGNORECASE),
        'score':   0.85,
        'severity':'HIGH',
        'match':   'per_line',
    },
    {
        'id':      'TB-005',
        'name':    'Node Timeout / Offline',
        'pattern': re.compile(r'(not\s+answer|connection\s+refused|timeout|unreachable)', re.IGNORECASE),
        'score':   0.7,
        'severity':'MEDIUM',
        'match':   'per_line',
    },
    {
        'id':        'TB-006',
        'name':      'High Error Ratio',
        'threshold': 0.3,    # >30% of lines are errors
        'score':     0.8,
        'severity':  'HIGH',
        'match':     'feature',
    },
]


def evaluate_window(win_id: str, events: list, features: dict) -> dict:
    triggered = []

    for rule in RULES:
        rid   = rule['id']
        match = rule.get('match', 'per_line')

        if match == 'per_line':
            if any(rule['pattern'].search(e) for e in events):
                triggered.append(rule)

        elif match == 'feature':
            if rid == 'TB-006':
                if features.get('error_ratio', 0) >= rule['threshold']:
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
    print(f"[Thunderbird Rules] Evaluating {len(windows):,} windows...")
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
    print(f"[Thunderbird Rules] Done — {flagged:,} flagged ({flagged/max(len(results),1)*100:.1f}%)")
    return results
