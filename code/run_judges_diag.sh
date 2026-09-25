#!/usr/bin/env bash
# Diagnostic for the B4b inversion (2026-09-03): trained OLMo stages default to slot A on
# shuffled pairs in the bare completion frame. Two candidate causes, crossed on the same
# 180 known-answer pairs, one representative trained stage per lineage:
#   frame     : completion (protocol default) vs chat (model's own template, same prefill)
#   criterion : creative (protocol default) vs better (the secondary probe)
# The (completion, creative) cell is already on disk from run_judges_validity.sh.
# Uses judge_v2.py (chat mode + readout-mass fields) so the still-running v1 driver is untouched.
set -uo pipefail
cd /content
mkdir -p out logs
JUDGES=(
  "olmo2  sft  allenai/OLMo-2-1124-7B-SFT"
  "zephyr sft  HuggingFaceH4/mistral-7b-sft-beta"
)
CELLS=(
  "completion better"
  "chat       creative"
  "chat       better"
)
for spec in "${JUDGES[@]}"; do
  set -- $spec; L=$1; S=$2; MODEL=$3
  for cell in "${CELLS[@]}"; do
    set -- $cell; FR=$1; CR=$2; TAG="$L-$S-$FR-$CR"
    if [ -f "out/DONE-diag-$TAG" ]; then echo "skip $TAG"; continue; fi
    CHAT=""; [ "$FR" = "chat" ] && CHAT="--use-chat-template"
    echo "=== $(date -Is) start $TAG" >> "logs/diag-$TAG.log"
    python judge_v2.py --model "$MODEL" --judge-lineage "$L" --judge-stage "$S" \
      --items validity-items.jsonl --prompts prompts.jsonl --pairs-file validity-pairs.jsonl \
      --criterion "$CR" $CHAT --out "out/diag-$TAG.jsonl" >> "logs/diag-$TAG.log" 2>&1
    RC=$?
    echo "=== $(date -Is) end $TAG rc=$RC" >> "logs/diag-$TAG.log"
    if [ $RC -eq 0 ]; then touch "out/DONE-diag-$TAG"; else echo "DIAG FAILED $TAG rc=$RC" >> logs/driver.log; exit $RC; fi
  done
  rm -rf /root/.cache/huggingface/hub/models--* 2>/dev/null
done
touch out/ALL-DONE-diag
echo "DIAG ALL DONE $(date -Is)" >> logs/driver.log
