#!/usr/bin/env bash
# Remote driver: B4b competence gate — every judge stage scores the same 180 known-answer
# pairs (data/validity/validity-pairs.jsonl), both orders. Sequential, one nohup launch.
# Writes /content/out/judge-<lineage>-<stage>.jsonl and /content/logs/<lineage>-<stage>.log;
# DONE-<tag> sentinel per judge, ALL-DONE at the end. HF cache purged between judges.
set -uo pipefail
cd /content
mkdir -p out logs
JUDGES=(
  "olmo2  base   allenai/OLMo-2-1124-7B"
  "olmo2  sft    allenai/OLMo-2-1124-7B-SFT"
  "olmo2  dpo    allenai/OLMo-2-1124-7B-DPO"
  "olmo2  final  allenai/OLMo-2-1124-7B-Instruct"
  "zephyr base   mistralai/Mistral-7B-v0.1"
  "zephyr sft    HuggingFaceH4/mistral-7b-sft-beta"
  "zephyr dpo    HuggingFaceH4/zephyr-7b-beta"
)
for spec in "${JUDGES[@]}"; do
  set -- $spec; L=$1; S=$2; MODEL=$3; TAG="$L-$S"
  if [ -f "out/DONE-$TAG" ]; then echo "skip $TAG (done)"; continue; fi
  echo "=== $(date -Is) start $TAG" >> "logs/$TAG.log"
  python judge.py --model "$MODEL" --judge-lineage "$L" --judge-stage "$S" \
    --items validity-items.jsonl --prompts prompts.jsonl --pairs-file validity-pairs.jsonl \
    --out "out/judge-$TAG.jsonl" >> "logs/$TAG.log" 2>&1
  RC=$?
  echo "=== $(date -Is) end $TAG rc=$RC" >> "logs/$TAG.log"
  if [ $RC -eq 0 ]; then touch "out/DONE-$TAG"; else echo "JUDGE FAILED $TAG rc=$RC" >> logs/driver.log; exit $RC; fi
  rm -rf /root/.cache/huggingface/hub/models--* 2>/dev/null
done
touch out/ALL-DONE
echo "ALL DONE $(date -Is)" >> logs/driver.log
