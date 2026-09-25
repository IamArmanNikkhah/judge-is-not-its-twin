#!/usr/bin/env bash
# Phase 3 remote driver (Amendment A3, ratified 2026-09-04): per judge stage, one download, three runs:
#   gate-better   the 180 known-answer pairs under "better"        -> out/gate-better-<tag>.jsonl
#   traj-creative real bank pairs (self-lineage excluded) "creative" -> out/traj-<tag>-creative.jsonl
#   traj-better   the SAME pairs (same seed) under "better"         -> out/traj-<tag>-better.jsonl
# Sequential, one nohup launch, DONE sentinels per run, HF cache purged per judge.
set -uo pipefail
cd /content
mkdir -p out logs
PAIRS_PER_PROMPT=20
JUDGES=(
  "olmo2  base   allenai/OLMo-2-1124-7B"
  "olmo2  sft    allenai/OLMo-2-1124-7B-SFT"
  "olmo2  dpo    allenai/OLMo-2-1124-7B-DPO"
  "olmo2  final  allenai/OLMo-2-1124-7B-Instruct"
  "zephyr base   mistralai/Mistral-7B-v0.1"
  "zephyr sft    HuggingFaceH4/mistral-7b-sft-beta"
  "zephyr dpo    HuggingFaceH4/zephyr-7b-beta"
)
run() {  # run <tag> <outfile> <args...>
  local TAG=$1 OUTF=$2; shift 2
  if [ -f "out/DONE-$TAG" ]; then echo "skip $TAG"; return 0; fi
  echo "=== $(date -Is) start $TAG" >> "logs/$TAG.log"
  python judge.py "$@" --out "$OUTF" >> "logs/$TAG.log" 2>&1
  local RC=$?
  echo "=== $(date -Is) end $TAG rc=$RC" >> "logs/$TAG.log"
  if [ $RC -eq 0 ]; then touch "out/DONE-$TAG"; else echo "FAILED $TAG rc=$RC" >> logs/driver.log; exit $RC; fi
}
for spec in "${JUDGES[@]}"; do
  set -- $spec; L=$1; S=$2; MODEL=$3; T="$L-$S"
  run "gate-better-$T" "out/gate-better-$T.jsonl" --model "$MODEL" --judge-lineage "$L" --judge-stage "$S" \
      --items validity-items.jsonl --prompts prompts.jsonl --pairs-file validity-pairs.jsonl --criterion better
  run "traj-$T-creative" "out/traj-$T-creative.jsonl" --model "$MODEL" --judge-lineage "$L" --judge-stage "$S" \
      --items bank-pilot.jsonl --prompts prompts.jsonl --pairs-per-prompt $PAIRS_PER_PROMPT --criterion creative
  run "traj-$T-better" "out/traj-$T-better.jsonl" --model "$MODEL" --judge-lineage "$L" --judge-stage "$S" \
      --items bank-pilot.jsonl --prompts prompts.jsonl --pairs-per-prompt $PAIRS_PER_PROMPT --criterion better
  rm -rf /root/.cache/huggingface/hub/models--* 2>/dev/null
done
touch out/ALL-DONE-phase3
echo "PHASE3 ALL DONE $(date -Is)" >> logs/driver.log
