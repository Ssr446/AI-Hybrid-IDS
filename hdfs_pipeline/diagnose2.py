"""
diagnose2.py — Identify which rules still fire on NORMAL blocks after v3 fix.
Samples first 500 normal and 500 anomaly blocks from first 3M lines.
"""
import sys, re
from collections import Counter

sys.path.insert(0, '.')
from hdfs_parser   import parse_hdfs_log, load_labels, group_by_block
from hdfs_features import build_feature_matrix, get_feature_columns
from hdfs_rules    import evaluate_block

HDFS_LOG_PATH = r"C:\Users\ssrsh\Documents\projects\minor project\HDFS.log"
LABEL_PATH    = r"C:\Users\ssrsh\Documents\projects\minor project\data\anomaly_label.csv"
SAMPLE_LINES  = 3_000_000

print("Parsing first 3M lines for diagnosis...")
df           = parse_hdfs_log(HDFS_LOG_PATH, max_lines=SAMPLE_LINES)
label_map    = load_labels(LABEL_PATH)
block_groups = group_by_block(df)
feature_df   = build_feature_matrix(block_groups, label_map)
feat_lookup  = feature_df.set_index('block_id').to_dict('index')

normal_blocks  = [b for b in block_groups if label_map.get(b) == 'Normal'][:500]
anomaly_blocks = [b for b in block_groups if label_map.get(b) == 'Anomaly'][:500]

print(f"\nSampling {len(normal_blocks)} normal + {len(anomaly_blocks)} anomaly blocks\n")

WARN_PAT    = re.compile(r'\bWARN\b')
IOEX_PAT    = re.compile(r'\bIOException\b')
ERRPAT      = re.compile(r'\b(Exception|IOException|Failed)\b')

def analyze(blocks, tag):
    rule_counter = Counter()
    flagged = 0
    warn_counts    = []
    ioex_counts    = []
    error_ratios   = []
    replicate_cnts = []

    for bid in blocks:
        events   = block_groups[bid]
        features = feat_lookup.get(bid, {})
        result   = evaluate_block(bid, events, features)

        joined = ' '.join(events)
        warnc  = sum(1 for e in events if WARN_PAT.search(e))
        ioexc  = sum(1 for e in events if IOEX_PAT.search(e))
        warn_counts.append(warnc)
        ioex_counts.append(ioexc)
        error_ratios.append(features.get('error_ratio', 0))
        replicate_cnts.append(features.get('replicate_count', 0))

        if result['rule_score'] > 0:
            flagged += 1
            for r in result['rules_hit']:
                rule_counter[r] += 1

    print(f"{'='*60}")
    print(f"{tag} — {flagged}/{len(blocks)} flagged ({flagged/max(len(blocks),1)*100:.1f}%)")
    print(f"Rules firing on {tag}:")
    for rule, count in rule_counter.most_common():
        print(f"  {rule}: {count}/{len(blocks)} ({count/len(blocks)*100:.1f}%)")
    if warn_counts:
        import statistics
        print(f"\nWARN count stats: mean={statistics.mean(warn_counts):.2f}, max={max(warn_counts)}, "
              f"blocks with >=5 WARN: {sum(1 for w in warn_counts if w>=5)}")
        print(f"IOEx count stats: mean={statistics.mean(ioex_counts):.2f}, max={max(ioex_counts)}, "
              f"blocks with >=1 IOEx: {sum(1 for w in ioex_counts if w>=1)}")
        print(f"error_ratio: mean={statistics.mean(error_ratios):.3f}, "
              f"blocks with >=0.5: {sum(1 for r in error_ratios if r>=0.5)}")
        print(f"replicate_count: mean={statistics.mean(replicate_cnts):.2f}, max={max(replicate_cnts)}, "
              f"blocks with >=6: {sum(1 for r in replicate_cnts if r>=6)}")

    # Show a sample flagged block
    for bid in blocks:
        events   = block_groups[bid]
        features = feat_lookup.get(bid, {})
        result   = evaluate_block(bid, events, features)
        if result['rule_score'] > 0:
            print(f"\nSample flagged {tag}: {bid}")
            print(f"  Rules: {result['rules_hit']}")
            print(f"  Events ({len(events)} total):")
            for e in events[:6]:
                print(f"    {e[:130]}")
            break

analyze(normal_blocks,  "NORMAL blocks")
analyze(anomaly_blocks, "ANOMALY blocks")
