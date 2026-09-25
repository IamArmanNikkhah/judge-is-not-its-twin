#!/usr/bin/env python
"""P2 judge-side trajectory (Amendment A3): does a judge's preference for TYPICAL text grow
across its training stages?

Per judge stage x criterion, on the real intact-vs-intact pairs (self-lineage excluded):
    p_x ~ dtyp + dlen                      (raw slope; dlen = log len_x - log len_y)
    p_x ~ dtyp + dlen + p_better           (creative, conditioned on the value axis; only
                                            when the A3 entanglement check passed)
    (p_creative - p_better) ~ dtyp + dlen  (the creativity-specific estimand)
dtyp = mean_nll(x) - mean_nll(y) under a READER, completion-prefix frame. NEGATIVE beta =
the judge prefers the more typical (lower-NLL) story. Two readers, because with two
lineages no reader is clean and the design's third lineage is absent:
    item-base   the ITEM's own base model (zephyr items -> mistral-base; olmo2 items ->
                olmo2-base). Judge-independent: the shared-diet regressor.
    judge-base  the JUDGE's own base model. The own-perplexity (P4) regressor.
CIs: cluster bootstrap over prompts (n=10, the only honest unit).

Usage: python judge_trajectory.py [--root ..] [--glob "data/judge/traj-*.jsonl"] [--reader item-base|judge-base]
"""
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

BASE_OF = {"olmo2": "olmo2-base", "zephyr": "mistral-base"}
STAGE_ORDER = {"base": 0, "sft": 1, "dpo": 2, "final": 3}
RNG = np.random.default_rng(411); NBOOT = 5000
R_MAX = 0.5


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def boot_beta(X, y, pids, col):
    """Cluster bootstrap (by prompt) of coefficient `col`."""
    P = sorted(set(pids)); idx = {p: np.where(pids == p)[0] for p in P}
    out = []
    for _ in range(NBOOT):
        take = np.concatenate([idx[p] for p in RNG.choice(P, size=len(P), replace=True)])
        out.append(ols(X[take], y[take])[col])
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--glob", default="data/judge/traj-*.jsonl")
    ap.add_argument("--reader", default="item-base", choices=["item-base", "judge-base"])
    ap.add_argument("--stage-control", action="store_true",
                    help="add dstage = ord(stage_x) - ord(stage_y) of the ITEMS (base=0..final=3): separates typicality from which training stage wrote the story")
    ap.add_argument("--min-chars", type=int, default=0,
                    help="keep only pairs where BOTH items have >= this many chars (stub sensitivity; OLMo items include 7-90 char stubs)")
    args = ap.parse_args()
    R = Path(args.root)
    bank = {r["item_id"]: r for r in map(json.loads, open(R / "data/items/bank-pilot.jsonl"))}
    typ = {t: {r["item_id"]: r["mean_nll"] for r in map(json.loads, open(R / f"data/typicality/typ-{t}-prefix.jsonl"))}
           for t in ("olmo2-base", "mistral-base")}

    runs = {}
    for f in sorted(glob.glob(str(R / args.glob))):
        rs = [json.loads(l) for l in open(f)]
        rs = [r for r in rs if len(bank[r["item_x"]]["text"]) >= args.min_chars and len(bank[r["item_y"]]["text"]) >= args.min_chars]
        if rs:
            runs[(rs[0]["judge_lineage"], rs[0]["judge_stage"], rs[0]["criterion"])] = rs

    def design(rs, L):
        px = np.array([r["p_x"] for r in rs])
        if args.reader == "item-base":
            d = np.array([typ[BASE_OF[bank[r["item_x"]]["lineage"]]][r["item_x"]] - typ[BASE_OF[bank[r["item_y"]]["lineage"]]][r["item_y"]] for r in rs])
        else:
            d = np.array([typ[BASE_OF[L]][r["item_x"]] - typ[BASE_OF[L]][r["item_y"]] for r in rs])
        dl = np.array([np.log(len(bank[r["item_x"]]["text"])) - np.log(len(bank[r["item_y"]]["text"])) for r in rs])
        pids = np.array([r["prompt_id"] for r in rs])
        if args.stage_control:
            ds = np.array([STAGE_ORDER[bank[r["item_x"]]["stage"]] - STAGE_ORDER[bank[r["item_y"]]["stage"]] for r in rs], float)
            dl = np.column_stack([dl, ds])
        return px, d, dl, pids

    print(f"P2 JUDGE-SIDE TRAJECTORY   reader = {args.reader}   min-chars = {args.min_chars}   stage-control = {args.stage_control}   negative beta = prefers the more typical story")
    print(f"cluster-bootstrap 95% CIs over prompts; n pairs per cell; r = simple correlation(p_x, dtyp)")
    print()
    print("%-7s %-6s %-9s %4s | %8s %-18s %7s | %8s %-18s | %8s" % ("judge", "stage", "crit", "n", "b_typ", "95% CI", "r", "b_len", "95% CI", "b_typ|better"))
    keys = sorted(runs, key=lambda k: (k[0], STAGE_ORDER.get(k[1], 9), k[2]))
    for (L, S, C) in keys:
        rs = runs[(L, S, C)]
        px, d, dl, pids = design(rs, L)
        X = np.column_stack([np.ones(len(px)), d, dl])
        b = ols(X, px); lo, hi = boot_beta(X, px, pids, 1); llo, lhi = boot_beta(X, px, pids, 2)
        r = np.corrcoef(px, d)[0, 1]
        cond = ""
        if C == "creative" and (L, S, "better") in runs:
            rb = runs[(L, S, "better")]
            pb = {x["item_x"] + "|" + x["item_y"]: x["p_x"] for x in rb}
            keyed = [x["item_x"] + "|" + x["item_y"] for x in rs]
            if all(k in pb for k in keyed):
                pbv = np.array([pb[k] for k in keyed])
                _, db, _, _ = design(rb, L)
                ent = abs(np.corrcoef(pbv, db)[0, 1])
                if ent <= R_MAX:
                    bc = ols(np.column_stack([X, pbv]), px)[1]
                    cond = "%+.4f" % bc
                else:
                    cond = "ENTANGLED(r=%.2f)" % ent
        print("%-7s %-6s %-9s %4d | %+8.4f [%+.4f,%+.4f] %+7.3f | %+8.4f [%+.4f,%+.4f] | %s" % (L, S, C, len(px), b[1], lo, hi, r, b[2], llo, lhi, cond))
    # contrast
    print()
    print("CREATIVITY-SPECIFIC ESTIMAND  (p_creative - p_better) ~ dtyp + dlen, same pairs")
    print("%-7s %-6s %4s | %8s %-18s | %8s" % ("judge", "stage", "n", "b_typ", "95% CI", "b_len"))
    for (L, S, C) in keys:
        if C != "creative" or (L, S, "better") not in runs:
            continue
        rc, rb = runs[(L, S, "creative")], runs[(L, S, "better")]
        pb = {x["item_x"] + "|" + x["item_y"]: x["p_x"] for x in rb}
        rs = [x for x in rc if x["item_x"] + "|" + x["item_y"] in pb]
        px, d, dl, pids = design(rs, L)
        y = px - np.array([pb[x["item_x"] + "|" + x["item_y"]] for x in rs])
        X = np.column_stack([np.ones(len(y)), d, dl])
        b = ols(X, y); lo, hi = boot_beta(X, y, pids, 1)
        print("%-7s %-6s %4d | %+8.4f [%+.4f,%+.4f] | %+8.4f" % (L, S, len(y), b[1], lo, hi, b[2]))


if __name__ == "__main__":
    main()
