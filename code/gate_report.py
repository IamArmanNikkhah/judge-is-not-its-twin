#!/usr/bin/env python
"""B4b gate report — per judge stage, can it tell an intact story from a ruined one?

Reads data/validity/judge-<lineage>-<stage>.jsonl (judge.py --pairs-file output, one per
judge). For every judge x ruin: n pairs, accuracy = fraction with p_x > 0.5 (x = intact),
Wilson 95% CI, mean p_x, and the position-bias read.

PASS RULE (Design Amendment A2, author-ratified 2026-09-03 "80 percent and the three ruins"):
  a judge stage passes the gate iff on SHUFFLED pairs accuracy >= 0.80 AND the Wilson 95%
  lower bound > 0.50. Stub and offprompt accuracies are REPORTED, never gate: a base model
  that ignores the prompt is a finding about base models, not a defect in the instrument.
Failing stages are reported but excluded from gradient claims (judging-protocol.md).

Usage: python gate_report.py [--root ..] > ../data/analysis/gate-<date>.txt
"""
import argparse, glob, json, math
from collections import defaultdict
from pathlib import Path

GATE_RUIN = "shuffled"
GATE_ACC = 0.80
GATE_CI_LOW = 0.50
RUINS = ("shuffled", "stub", "offprompt")
STAGE_ORDER = {"base": 0, "sft": 1, "dpo": 2, "final": 3}


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--glob", default="data/validity/judge-*.jsonl",
                    help='"data/validity/gate-better-*.jsonl" for the A3 gate under "better"')
    args = ap.parse_args()
    R = Path(args.root)
    files = sorted(glob.glob(str(R / args.glob)))
    if not files:
        raise SystemExit(f"no files for {args.glob}")
    print(f"B4b COMPETENCE GATE — rule: {GATE_RUIN} acc >= {GATE_ACC:.2f} and Wilson95 low > {GATE_CI_LOW:.2f}")
    print(f"files: {len(files)}  ({args.glob})  criterion: {json.loads(open(files[0]).readline()).get('criterion')}")
    rows = []
    for f in files:
        recs = [json.loads(l) for l in open(f)]
        if not recs:
            continue
        L, S = recs[0]["judge_lineage"], recs[0]["judge_stage"]
        by = defaultdict(list)
        for r in recs:
            by[r.get("ruin", "?")].append(r)
        res = {}
        for ruin in RUINS:
            rs = by.get(ruin, [])
            n = len(rs); k = sum(1 for r in rs if r["p_x"] > 0.5)
            lo, hi = wilson(k, n)
            mp = sum(r["p_x"] for r in rs) / n if n else float("nan")
            pos = sum(abs(r["p_x_first_order"] - (1 - r["p_y_first_order"])) for r in rs) / n if n else float("nan")
            res[ruin] = (n, k / n if n else float("nan"), lo, hi, mp, pos)
        g = res[GATE_RUIN]
        passed = (g[0] > 0) and g[1] >= GATE_ACC and g[2] > GATE_CI_LOW
        rows.append((L, STAGE_ORDER.get(S, 9), S, res, passed, len(recs)))
    rows.sort(key=lambda r: (r[0], r[1]))
    print()
    print("%-7s %-6s %-5s | %-32s | %-32s | %-32s | pairs" % ("lineage", "stage", "GATE", "shuffled acc [CI] mean_px", "stub acc [CI] mean_px", "offprompt acc [CI] mean_px"))
    for L, _, S, res, passed, n in rows:
        cells = []
        for ruin in RUINS:
            nn, acc, lo, hi, mp, pos = res[ruin]
            cells.append("%.2f [%.2f,%.2f] %.2f (n=%d)" % (acc, lo, hi, mp, nn))
        print("%-7s %-6s %-5s | %-32s | %-32s | %-32s | %d" % (L, S, "PASS" if passed else "FAIL", *cells, n))
    print()
    print("position bias (mean |order disagreement| on shuffled pairs):")
    for L, _, S, res, passed, n in rows:
        print("   %-7s %-6s %.3f" % (L, S, res[GATE_RUIN][5]))
    print()
    print("Reading: shuffled is the harsh test and the only one that gates. A judge under ~0.6 on")
    print("stub prefers short to long or cannot see length; a judge near 0.5 on offprompt is not")
    print("reading the prompt. Both are findings about that stage, reported alongside its slope.")


if __name__ == "__main__":
    main()
