#!/usr/bin/env bash
# Remote driver: the four reference passes under the completion-prefix frame, sequentially,
# each resumable from a banked partial. Launched ONCE via nohup from a single `colab exec`
# (launch-and-arm is one action; RUNLOG 09-01: a pass launched in a call that did something
# else was lost entirely). Writes /content/out/typ-<tag>-prefix.jsonl and /content/logs/<tag>.log.
# A sentinel file /content/out/DONE-<tag> marks each finished pass; /content/out/ALL-DONE the set.
set -uo pipefail
cd /content
mkdir -p out logs
PASSES=(
  "mistral-base    mistralai/Mistral-7B-v0.1"
  "olmo2-base      allenai/OLMo-2-1124-7B"
  "olmo2-instruct  allenai/OLMo-2-1124-7B-Instruct"
  "zephyr-final    HuggingFaceH4/zephyr-7b-beta"
)
for spec in "${PASSES[@]}"; do
  set -- $spec; TAG=$1; MODEL=$2
  if [ -f "out/DONE-$TAG" ]; then echo "skip $TAG (done)"; continue; fi
  RESUME=""
  [ -f "out/typ-$TAG-prefix.partial.jsonl" ] && RESUME="--resume-from out/typ-$TAG-prefix.partial.jsonl"
  echo "=== $(date -Is) start $TAG" >> "logs/$TAG.log"
  python typicality.py --model "$MODEL" --model-tag "$TAG" --frame completion-prefix \
    --prompts prompts.jsonl --items bank-pilot.jsonl \
    --out "out/typ-$TAG-prefix.jsonl" $RESUME >> "logs/$TAG.log" 2>&1
  RC=$?
  echo "=== $(date -Is) end $TAG rc=$RC" >> "logs/$TAG.log"
  if [ $RC -eq 0 ]; then touch "out/DONE-$TAG"; else echo "PASS FAILED $TAG rc=$RC" >> logs/driver.log; exit $RC; fi
  # free the HF cache between passes: T4 disk is finite and each checkpoint is ~14GB
  rm -rf /root/.cache/huggingface/hub/models--* 2>/dev/null
done
touch out/ALL-DONE
echo "ALL DONE $(date -Is)" >> logs/driver.log
