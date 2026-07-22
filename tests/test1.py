# import json

# with open('data/processed/chunks_sentence.jsonl', 'r', encoding='utf-8') as f:
#     chunks = [json.loads(l) for l in f]

# gold = [c for c in chunks if c['chunk_id'].split('::')[0] == '0000287']

# for c in gold:
#     print(c['focus'], '|', c['qtype'])
#     print(c['chunk_text'][:300])
#     print()

# import json
# report = json.load(open('data/eval/eval_report.json'))
# misses = [i for i in report['retrieval']['per_item'] if i['type']=='real' and i.get('hybrid_hit') is False]
# for m in misses:
#     print(m['question'], '| dense_hit:', m['dense_hit'], '| hybrid_hit:', m['hybrid_hit'])


# import json
# items = json.load(open('data/eval/eval_set.json'))
# for i in items:
#     if 'Lung Cancer' in i['question'] and i['type']=='real':
#         print(i['gold_pair_uid'])

import json

with open("data/processed/chunks_sentence.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(l) for l in f]
gold = [c for c in chunks if c["chunk_id"].split("::")[0] == "0015286"]
print("=== GOLD chunk ===")
print("source:", gold[0]["source"], "| focus:", gold[0]["focus"], "| qtype:", gold[0]["qtype"])
print(gold[0]["chunk_text"][:200])
print()

# Now find which pair_uid the reranker actually picked instead, by matching
# the exact preview text from the diagnose_hybrid.py output
picks_preview = [
    "Doctors treat patients with non-small cell lung cancer",
    "Researchers continue to look at new ways to combine",
]
for p in picks_preview:
    matches = [c for c in chunks if c["chunk_text"].startswith(p)]
    for m in matches:
        print("=== Reranker pick ===")
        print("pair_uid:", m["chunk_id"].split("::")[0], "| source:", m["source"], "| focus:", m["focus"], "| qtype:", m["qtype"])
        print(m["chunk_text"][:200])
        print()