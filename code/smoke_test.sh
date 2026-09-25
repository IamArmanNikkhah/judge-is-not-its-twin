#!/usr/bin/env bash
# End-to-end plumbing test on tiny models (~minutes, laptop-scale).
# Verifies: generate -> merge -> typicality -> judge chain produces valid,
# well-formed JSONL with the provenance fields analysis needs.
# Stand-ins: sshleifer/tiny-gpt2 (2MB, random weights) + EleutherAI/pythia-70m.
# Text quality is irrelevant here; field flow and logit plumbing are the test.
set -euo pipefail
cd "$(dirname "$0")"
PY=../.venv/bin/python
RUNS=../data/runs/smoke
mkdir -p "$RUNS"

$PY generate.py --model sshleifer/tiny-gpt2 --lineage smokeA --stage smoke \
  --prompts ../data/items/prompts.jsonl --out "$RUNS/gen-A.jsonl" \
  --k 2 --prompt-limit 2 --max-new-tokens 30

$PY generate.py --model EleutherAI/pythia-70m --lineage smokeB --stage smoke \
  --prompts ../data/items/prompts.jsonl --out "$RUNS/gen-B.jsonl" \
  --k 2 --prompt-limit 2 --max-new-tokens 30

cat "$RUNS/gen-A.jsonl" "$RUNS/gen-B.jsonl" > "$RUNS/items.jsonl"

$PY typicality.py --model EleutherAI/pythia-70m --model-tag smoke-ref \
  --items "$RUNS/items.jsonl" --out "$RUNS/typ.jsonl"

# judge-lineage smokeC = neither source lineage -> exclusion logic active, all cross pairs eligible
$PY judge.py --model EleutherAI/pythia-70m --judge-lineage smokeC --judge-stage smoke \
  --items "$RUNS/items.jsonl" --prompts ../data/items/prompts.jsonl \
  --out "$RUNS/judge.jsonl" --pairs-per-prompt 4

# self-pair exclusion check: judging as smokeB must yield ZERO pairs
# (only cross-model pairs are eligible, and every cross pair contains a smokeB item)
if $PY judge.py --model EleutherAI/pythia-70m --judge-lineage smokeB --judge-stage smoke \
  --items "$RUNS/items.jsonl" --prompts ../data/items/prompts.jsonl \
  --out "$RUNS/judge-selfexcl.jsonl" --pairs-per-prompt 4 2>/dev/null; then
  echo "FAIL: smokeB judge should have found no eligible pairs"; exit 1
else
  echo "self-pair exclusion behaves (smokeB judge correctly found no pairs)"
fi

$PY - <<'EOF'
import json
gen = [json.loads(l) for l in open("../data/runs/smoke/items.jsonl")]
typ = [json.loads(l) for l in open("../data/runs/smoke/typ.jsonl")]
jdg = [json.loads(l) for l in open("../data/runs/smoke/judge.jsonl")]
assert len(gen) == 8, f"expected 8 generations, got {len(gen)}"
assert len(typ) == len(gen), f"typicality rows {len(typ)} != items {len(gen)}"
assert all(0.0 <= r["p_x"] <= 1.0 for r in jdg)
ids = {g["item_id"] for g in gen}
assert all(r["item_id"] in ids for r in typ)
assert all(r["item_x"] in ids and r["item_y"] in ids for r in jdg)
orders = [abs(r["p_x_first_order"] - (1 - r["p_y_first_order"])) for r in jdg]
print(f"SMOKE PASS: {len(gen)} items, {len(typ)} typicality rows, {len(jdg)} judged pairs")
print(f"  order-swap spread (position-bias signal, expected nonzero): "
      f"mean {sum(orders)/len(orders):.4f}, max {max(orders):.4f}")
EOF
echo "smoke test complete"
