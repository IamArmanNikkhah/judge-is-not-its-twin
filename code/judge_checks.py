#!/usr/bin/env python
"""Amendment A3 tripwires, run on trajectory judge files as they land (before spending the
rest of the hour):

  NON-DEGENERACY  per judge x criterion on the real intact-vs-intact pairs:
                  sd(p_x) >= SD_MIN and mean |order disagreement| <= DIS_MAX.
                  A frozen readout fails; that criterion arm is non-executable for that stage.
  ENTANGLEMENT    per judge, on the "better" file: |r(p_better, delta-typicality)| <= R_MAX.
                  Above it, "better" is typicality in a costume for that stage: conditioning is
                  dropped and the raw "creative" slope ships with the non-degeneracy check only.
Delta-typicality = mean_nll(item_x) - mean_nll(item_y) under the HELD-OUT reader for the
judge's lineage (leave-own-lineage-out), completion-prefix frame.

Thresholds (A3, ratified 2026-09-04): SD_MIN .07, DIS_MAX .45 (halfway between the pilot base
run, sd .147 / dis .143, and the observed freezes, dis .46-.68); R_MAX .5 (the author's number).

Usage: python judge_checks.py [--root ..] [--glob "data/judge/traj-*.jsonl"]
"""
import argparse, glob, json
from pathlib import Path
import numpy as np

SD_MIN, DIS_MAX, R_MAX = 0.07, 0.45, 0.5
HELDOUT = {"olmo2": "mistral-base", "zephyr": "olmo2-base"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--glob", default="data/judge/traj-*.jsonl")
    args = ap.parse_args()
    R = Path(args.root)
    typ = {}
    for tag in set(HELDOUT.values()):
        typ[tag] = {r["item_id"]: r["mean_nll"] for r in map(json.loads, open(R / f"data/typicality/typ-{tag}-prefix.jsonl"))}
    files = sorted(glob.glob(str(R / args.glob)))
    if not files:
        raise SystemExit(f"no files for {args.glob}")
    print(f"A3 TRIPWIRES  sd(p_x) >= {SD_MIN}, order-disagreement <= {DIS_MAX}, |r(p_better, dtyp)| <= {R_MAX}")
    print("%-7s %-6s %-9s %5s %7s %7s %-6s | %8s %-6s" % ("judge", "stage", "crit", "n", "sd_px", "disagr", "FROZEN", "r_dtyp", "ENTGL"))
    for f in files:
        rs = [json.loads(l) for l in open(f)]
        if not rs:
            continue
        L, S, C = rs[0]["judge_lineage"], rs[0]["judge_stage"], rs[0]["criterion"]
        px = np.array([r["p_x"] for r in rs])
        dis = np.mean([abs(r["p_x_first_order"] - (1 - r["p_y_first_order"])) for r in rs])
        frozen = (px.std() < SD_MIN) or (dis > DIS_MAX)
        ref = typ[HELDOUT[L]]
        d = np.array([ref[r["item_x"]] - ref[r["item_y"]] for r in rs])
        rr = np.corrcoef(px, d)[0, 1] if px.std() > 0 else float("nan")
        ent = (C == "better") and abs(rr) > R_MAX
        print("%-7s %-6s %-9s %5d %7.3f %7.3f %-6s | %+8.3f %-6s" % (L, S, C, len(rs), px.std(), dis, "FAIL" if frozen else "ok", rr, ("FAIL" if ent else "ok") if C == "better" else "-"))


if __name__ == "__main__":
    main()
