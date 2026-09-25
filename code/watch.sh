#!/usr/bin/env bash
# Local watcher for a remote sequential driver (Monitor v3 shape, promoted from the
# 09-01 scratchpad scripts that a session boundary wiped — RUNLOG "standing fix 2").
#   watch.sh <session> <poll_seconds> [job]      job = typicality (default) | validity
# Every poll: probe each pass (REMOTE_OK sentinel), bank the partial output locally so a
# dying VM can take at most one poll of work, and print ONLY milestone crossings,
# terminal states, and death. Every emitted line interpolates ${SESSION} (09-01: a
# sed-derived watcher reported the wrong machine dead).
# Portability: macOS ships bash 3.2 — no associative arrays (bit 2026-09-03), so per-tag
# state lives in a temp dir. Tag/paths are sed-baked into a per-tag probe copy because
# neither argv nor env crosses `colab exec`.
set -uo pipefail
SESSION=${1:?session}; POLL=${2:-90}; JOB=${3:-typicality}; STALL_POLLS=${STALL_POLLS:-4}
cd "$(dirname "$0")/.."
COLAB=".venv/bin/python code/colab_call.py 120"   # hard client-side timeout on every call (2026-09-04: a hung exec froze the watcher 2.5 h)
case "$JOB" in
  typicality)
    TAGS="mistral-base olmo2-base olmo2-instruct zephyr-final"; TOTAL=1389; PROC="typicality.py"
    remote() { echo "/content/out/typ-$1-prefix.jsonl"; }
    partial() { echo "data/runs/typ-$1-prefix.partial.jsonl"; }
    final() { echo "data/typicality/typ-$1-prefix.jsonl"; }
    done_mark() { echo "data/runs/DONE-$1-prefix"; }
    STEP=250 ;;
  validity)
    TAGS="olmo2-base olmo2-sft olmo2-dpo olmo2-final zephyr-base zephyr-sft zephyr-dpo"; TOTAL=180; PROC="judge.py"
    remote() { echo "/content/out/judge-$1.jsonl"; }
    partial() { echo "data/runs/validity-$1.partial.jsonl"; }
    final() { echo "data/validity/judge-$1.jsonl"; }
    done_mark() { echo "data/runs/DONE-validity-$1"; }
    STEP=60 ;;
  diag)
    TAGS="olmo2-sft-completion-better olmo2-sft-chat-creative olmo2-sft-chat-better zephyr-sft-completion-better zephyr-sft-chat-creative zephyr-sft-chat-better"; TOTAL=180; PROC="judge_v2.py"
    remote() { echo "/content/out/diag-$1.jsonl"; }
    partial() { echo "data/runs/diag-$1.partial.jsonl"; }
    final() { echo "data/validity/diag-$1.jsonl"; }
    done_mark() { echo "data/runs/DONE-diag-$1"; }
    STEP=60; LOGPFX="diag-" ;;
  phase3)
    TAGS=""; for j in olmo2-base olmo2-sft olmo2-dpo olmo2-final zephyr-base zephyr-sft zephyr-dpo; do TAGS="$TAGS gate-better-$j traj-$j-creative traj-$j-better"; done
    TOTAL=200; PROC="judge.py"
    remote() { echo "/content/out/$1.jsonl"; }
    partial() { echo "data/runs/$1.partial.jsonl"; }
    final() { case "$1" in gate-*) echo "data/validity/$1.jsonl";; *) echo "data/judge/$1.jsonl";; esac; }
    done_mark() { echo "data/runs/DONE-$1"; }
    total() { case "$1" in gate-*) echo 180;; *) echo 200;; esac; }
    STEP=60 ;;
  *) echo "unknown job $JOB"; exit 1 ;;
esac
type total >/dev/null 2>&1 || total() { echo "$TOTAL"; }
STATE=$(mktemp -d /tmp/watch-${SESSION}.XXXX); MISSES=0
mkdir -p data/runs data/typicality data/validity data/judge
say() { echo "[$(date +%H:%M:%S) ${SESSION}] $*"; }
for TAG in $TAGS; do
  sed -e "s/__TAG__/$TAG/" -e "s#__OUT__#$(remote $TAG)#" -e "s#__LOG__#/content/logs/${LOGPFX:-}$TAG.log#" -e "s#__PROC__#$PROC#" \
      code/probe_remote.py > "$STATE/probe_$TAG.py"
done
say "watcher armed [$JOB]: $TAGS poll=${POLL}s state=$STATE"
while true; do
  ok=0
  for TAG in $TAGS; do
    [ -f "$(done_mark $TAG)" ] && continue
    OUT=$($COLAB exec -s "$SESSION" -f "$STATE/probe_$TAG.py" --timeout 60 2>/dev/null | grep "^REMOTE_OK $TAG " | tail -1)
    [ -z "$OUT" ] && continue
    ok=1
    ROWS=$(echo "$OUT" | awk '{print $3}'); ALIVE=$(echo "$OUT" | awk '{print $4}')
    if [ "${ROWS:-0}" -gt 0 ]; then
      $COLAB download -s "$SESSION" "$(remote $TAG)" "$(partial $TAG)" >/dev/null 2>&1
    fi
    M=$(( ROWS / STEP )); PREV=$(cat "$STATE/ms_$TAG" 2>/dev/null || echo x)
    if [ "$M" != "$PREV" ]; then echo "$M" > "$STATE/ms_$TAG"; say "$TAG ${ROWS}/$(total $TAG) alive=${ALIVE}"; fi
    if [ "$ROWS" -ge "$(total $TAG)" ]; then
      cp "$(partial $TAG)" "$(final $TAG)"
      touch "$(done_mark $TAG)"; say "$TAG COMPLETE ${ROWS}/$(total $TAG) -> $(final $TAG)"
    fi
    LOGTAIL=$(echo "$OUT" | cut -d' ' -f5-)
    # Coverage (2026-09-04, judgeQ: two silent hours, zero rows, then death): a dead process
    # is reported ONCE regardless of what the last log line looks like, and a live process
    # that makes no progress for STALL_POLLS polls is reported with its log tail.
    if [ "$ALIVE" = "0" ] && [ "$ROWS" -lt "$(total $TAG)" ] && [ -n "$LOGTAIL" ] && [ ! -f "$STATE/gone_$TAG" ]; then
      touch "$STATE/gone_$TAG"; say "$TAG PROCESS GONE at ${ROWS}: ${LOGTAIL}"
    fi
    # stall logic only for a job that has STARTED (has a log line); `alive` is machine-wide
    if [ "$ALIVE" = "1" ] && [ -n "$LOGTAIL" ]; then
      LASTROWS=$(cat "$STATE/rows_$TAG" 2>/dev/null || echo -1)
      if [ "$ROWS" = "$LASTROWS" ]; then N=$(( $(cat "$STATE/stall_$TAG" 2>/dev/null || echo 0) + 1 )); else N=0; fi
      echo "$ROWS" > "$STATE/rows_$TAG"; echo "$N" > "$STATE/stall_$TAG"
      if [ "$N" -ge "$STALL_POLLS" ] && [ $(( N % STALL_POLLS )) -eq 0 ]; then say "$TAG STALL ${N} polls at ${ROWS} rows: ${LOGTAIL}"; fi
    fi
  done
  if [ $ok -eq 1 ]; then MISSES=0; else MISSES=$((MISSES+1)); say "probe miss ${MISSES}/3"; fi
  if [ $MISSES -ge 3 ]; then say "DEAD: three consecutive probe misses — ${SESSION} presumed lost; partials banked in data/runs/"; exit 2; fi
  ALLDONE=1; for TAG in $TAGS; do [ -f "$(done_mark $TAG)" ] || ALLDONE=0; done
  if [ $ALLDONE -eq 1 ]; then say "ALL PASSES BANKED [$JOB]"; exit 0; fi
  sleep "$POLL"
done
