#!/usr/bin/env python
"""Frame x criterion diagnostic for the B4b inversion. Reads data/validity/diag-*.jsonl and the
(completion, creative) baseline from data/validity/judge-<L>-<S>.jsonl. Per cell x ruin:
accuracy (p_x > .5), slot behaviour (frac picking A in each order), readout mass on A/B, top-1."""
import glob, json, sys
from collections import Counter
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent.parent
rows = []
for f in sorted(glob.glob(str(R / "data/validity/judge-*-sft.jsonl"))) + sorted(glob.glob(str(R / "data/validity/diag-*.jsonl"))):
    rs = [json.loads(l) for l in open(f)]
    if not rs: continue
    L, S = rs[0]["judge_lineage"], rs[0]["judge_stage"]
    fr, cr = rs[0].get("frame", "completion"), rs[0]["criterion"]
    for ruin in ("shuffled", "stub", "offprompt"):
        r = [x for x in rs if x.get("ruin") == ruin]
        if not r: continue
        px = np.array([x["p_x"] for x in r]); pxf = np.array([x["p_x_first_order"] for x in r]); pyf = np.array([x["p_y_first_order"] for x in r])
        mass = np.median([x.get("mass_ab_first_order", float("nan")) for x in r])
        top = Counter(x.get("top1_first_order", "?") for x in r).most_common(2)
        rows.append((L, S, fr, cr, ruin, np.mean(px > .5), np.mean(pxf > .5), np.mean(pyf > .5), mass, top))
print("%-6s %-4s %-10s %-8s %-9s %5s %8s %8s %6s  top1" % ("judge", "stg", "frame", "crit", "ruin", "acc", "A|int1st", "A|ruin1st", "mass"))
for L, S, fr, cr, ruin, acc, a1, a2, mass, top in rows:
    print("%-6s %-4s %-10s %-8s %-9s %5.2f %8.2f %8.2f %6s  %s" % (L, S, fr, cr, ruin, acc, a1, a2, ("%.3f" % mass) if mass == mass else "  -  ", top))
