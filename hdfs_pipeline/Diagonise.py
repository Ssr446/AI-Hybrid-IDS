"""
diagnose.py — Run this to see WHY normal blocks are being flagged.
Samples 1000 normal blocks and 1000 anomaly blocks, shows which rules fire.
"""
import sys
sys.path.insert(0, '.')

from hdfs_parser   import parse_hdfs_log, load_labels, group_by_block
from hdfs_features import build_feature_matrix, get_feature_columns
from hdfs_rules    import evaluate_block
from collections   import Counter

HDFS_LOG_PATH = r"C:\Users\91900\Documents\logbased\misc\HDFS.log"
LABEL_PATH    = r"C:\Users\91900\Documents\logbased\data\anomaly_label.csv"
SAMPLE_SIZE   = 2000  # only parse first 2M lines for speed

print("Parsing first 2M lines for diagnosis...")
df        = parse_hdfs_log(HDFS_LOG_PATH, max_lines=2_000_000)
label_map = load_labels(LABEL_PATH)
block_groups = group_by_block(df)
feature_df   = build_feature_matrix(block_groups, label_map)
feat_lookup  = feature_df.set_index('block_id').to_dict('index')

normal_blocks  = [b for b in block_groups if label_map.get(b) == 'Normal'][:500]
anomaly_blocks = [b for b in block_groups if label_map.get(b) == 'Anomaly'][:500]

print(f"\nSampling {len(normal_blocks)} normal + {len(anomaly_blocks)} anomaly blocks\n")

def analyze(blocks, tag):
    rule_counter  = Counter()
    flagged       = 0
    for bid in blocks:
        events   = block_groups[bid]
        features = feat_lookup.get(bid, {})
        result   = evaluate_block(bid, events, features)
        if result['rule_score'] > 0:
            flagged += 1
            for r in result['rules_hit']:
                rule_counter[r] += 1

    print(f"{'='*50}")
    print(f"{tag} — {flagged}/{len(blocks)} flagged ({flagged/max(len(blocks),1)*100:.1f}%)")
    print(f"Rules firing:")
    for rule, count in rule_counter.most_common():
        print(f"  {rule}: {count} blocks")

    # Show a sample flagged block's events
    for bid in blocks:
        events   = block_groups[bid]
        features = feat_lookup.get(bid, {})
        result   = evaluate_block(bid, events, features)
        if result['rule_score'] > 0:
            print(f"\nSample flagged {tag} block: {bid}")
            print(f"  Rules: {result['rules_hit']}")
            print(f"  Events ({len(events)} total):")
            for e in events[:5]:
                print(f"    {e[:120]}")
            break

analyze(normal_blocks,  "NORMAL blocks")
analyze(anomaly_blocks, "ANOMALY blocks")