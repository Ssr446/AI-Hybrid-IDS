"""
hdfs_rules.py  (v4 — per-line matching to prevent cross-event false positives)

Key insight from diagnosis:
- HDFS-001 was matching 'PacketResponder' in one event + 'Exception' in ANOTHER event
  of the same block (joined string), causing 45% FPR on normals.

Fix: All regex rules now match against INDIVIDUAL event lines, not the joined string.
"""

import re

# Match per-line (not joined) patterns
RULES = [
    {
        'id':      'HDFS-001',
        'name':    'PacketResponder Exception',
        'pattern': re.compile(r'PacketResponder.*\bException\b', re.IGNORECASE),
        'score':   1.0,
        'severity':'CRITICAL',
        'match':   'per_line',   # match each event line independently
    },
    {
        'id':      'HDFS-002',
        'name':    'Block Transfer Failed',
        'pattern': re.compile(r'(writeBlock|receiveBlock).*[Ff]ailed|[Ff]ailed.*(writeBlock|receiveBlock)'),
        'score':   0.9,
        'severity':'HIGH',
        'match':   'per_line',
    },
    {
        'id':      'HDFS-003',
        'name':    'IOException During Block Op',
        'pattern': re.compile(r'\bIOException\b'),
        'score':   0.85,
        'severity':'HIGH',
        'match':   'per_line',
    },
    {
        'id':        'HDFS-004',
        'name':      'Excessive Replication',
        'threshold': 6,
        'score':     0.7,
        'severity':  'MEDIUM',
        'match':     'feature',
    },
    {
        'id':        'HDFS-006',
        'name':      'High Error Rate in Block',
        'threshold': 0.5,
        'score':     0.8,
        'severity':  'HIGH',
        'match':     'feature',
    },
    {
        'id':      'HDFS-007',
        'name':    'Excessive Warnings',
        'pattern': re.compile(r'\bWARN\b'),
        'threshold': 5,
        'score':   0.5,
        'severity':'MEDIUM',
        'match':   'count_lines',  # count matching lines, threshold on count
    },
    {
        'id':      'HDFS-008',
        'name':    'Short Failed Block',
        'pattern': re.compile(r'\b(Exception|IOException|Failed)\b'),
        'max_len': 5,
        'score':   0.8,
        'severity':'HIGH',
        'match':   'per_line',
    },
]


def evaluate_block(block_id: str, events: list, features: dict) -> dict:
    triggered = []
    n = len(events)

    for rule in RULES:
        rid   = rule['id']
        match = rule.get('match', 'joined')

        if match == 'per_line':
            # Rule fires only if pattern matches in at least one individual event line
            pat = rule['pattern']
            if rid == 'HDFS-008':
                # Additionally require short sequence
                if n <= rule['max_len'] and any(pat.search(e) for e in events):
                    triggered.append(rule)
            else:
                if any(pat.search(e) for e in events):
                    triggered.append(rule)

        elif match == 'feature':
            if rid == 'HDFS-004':
                if features.get('replicate_count', 0) >= rule['threshold']:
                    triggered.append(rule)
            elif rid == 'HDFS-006':
                if features.get('error_ratio', 0) >= rule['threshold']:
                    triggered.append(rule)

        elif match == 'count_lines':
            # Count lines matching pattern; fire if count >= threshold
            pat = rule['pattern']
            count = sum(1 for e in events if pat.search(e))
            if count >= rule['threshold']:
                triggered.append(rule)

    rule_score = max((r['score'] for r in triggered), default=0.0)

    return {
        'block_id':     block_id,
        'rule_score':   rule_score,
        'rules_hit':    [r['id'] for r in triggered],
        'rule_names':   [r['name'] for r in triggered],
        'top_severity': triggered[0]['severity'] if triggered else 'NORMAL',
    }


def evaluate_all_blocks(block_groups: dict, feature_df) -> list:
    print(f"[HDFS Rules] Evaluating {len(block_groups):,} blocks...")
    results     = []
    feat_lookup = feature_df.set_index('block_id').to_dict('index') \
                  if 'block_id' in feature_df.columns else {}

    for i, (block_id, events) in enumerate(block_groups.items()):
        features = feat_lookup.get(block_id, {})
        result   = evaluate_block(block_id, events, features)
        results.append(result)

        if (i + 1) % 50_000 == 0:
            print(f"  Evaluated {i+1:,} blocks...")

    flagged = sum(1 for r in results if r['rule_score'] > 0)
    print(f"[HDFS Rules] Done — {flagged:,} blocks flagged ({flagged/len(results)*100:.1f}%)")
    return results