#!/usr/bin/env python
"""Drift bound (post hoc, rule in data/analysis/drift-bound-RULE-2026-09-25.md).

How far could each trained judge's slope on predictability have moved TOWARD the
predictable story, beyond its own base judge's slope? Same model and reader as the
preregistered form of judge_trajectory.py (p_x ~ dtyp + dlen, item-base reader, all pairs).
Every stage in a family judged the same 200 pairs, so the cluster bootstrap is paired:
one draw of prompts refits base and stage together. drift = -(b_stage - b_base); positive
drift = toward the predictable story (the twin direction).
"""
import json, sys
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parent.parent
BASE_OF = {"olmo2": "olmo2-base", "zephyr": "mistral-base"}
STAGES = {"olmo2": ["sft", "dpo", "final"], "zephyr": ["sft", "dpo"]}
READER = sys.argv[1] if len(sys.argv) > 1 else "item-base"  # item-base (story's own family base) | judge-base (judge's family base; Amendment 2)
assert READER in ("item-base", "judge-base")
NBOOT = 5000
rng = np.random.default_rng(411)

bank = {r["item_id"]: r for r in map(json.loads, open(R / "data/items/bank-pilot.jsonl"))}
typ = {t: {r["item_id"]: r["mean_nll"] for r in map(json.loads, open(R / f"data/typicality/typ-{t}-prefix.jsonl"))}
       for t in BASE_OF.values()}


def load(L, S, C):
    rs = [json.loads(l) for l in open(R / f"data/judge/traj-{L}-{S}-{C}.jsonl")]
    rs.sort(key=lambda r: (r["item_x"], r["item_y"]))
    px = np.array([r["p_x"] for r in rs])
    if READER == "item-base":
        d = np.array([typ[BASE_OF[bank[r["item_x"]]["lineage"]]][r["item_x"]]
                      - typ[BASE_OF[bank[r["item_y"]]["lineage"]]][r["item_y"]] for r in rs])
    else:
        d = np.array([typ[BASE_OF[L]][r["item_x"]] - typ[BASE_OF[L]][r["item_y"]] for r in rs])
    dl = np.array([np.log(len(bank[r["item_x"]]["text"])) - np.log(len(bank[r["item_y"]]["text"])) for r in rs])
    X = np.column_stack([np.ones(len(px)), d, dl])
    keys = [(r["item_x"], r["item_y"]) for r in rs]
    return X, px, np.array([r["prompt_id"] for r in rs]), keys


def slope(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0][1]


print("DRIFT BOUND (post hoc). drift = -(b_stage - b_base); positive = toward the predictable story.")
print("Form: p_x ~ dtyp + dlen, reader = %s, all pairs." % READER + " Paired cluster bootstrap over prompts, %d draws, seed 411." % NBOOT)
print()
print("%-7s %-9s %-6s | %8s %8s | %8s %-20s | %9s | %s" % ("judge", "crit", "stage", "b_base", "b_stage", "drift", "95% CI (two-sided)", "95% upper", "max pick shift at a typical judged pair"))
for C in ("better", "creative"):
    for L, stages in STAGES.items():
        Xb, yb, pb, kb = load(L, "base", C)
        P = sorted(set(pb)); idx = {p: np.where(pb == p)[0] for p in P}
        for S in stages:
            Xs, ys, ps, ks = load(L, S, C)
            assert ks == kb and (ps == pb).all(), "pairs differ across stages"
            ruler = np.abs(Xb[:, 1]).mean()  # mean |dtyp| of the judged pairs under READER (Amendment 1/2)
            bb, bs = slope(Xb, yb), slope(Xs, ys)
            drift = -(bs - bb)
            boots = np.empty(NBOOT)
            for i in range(NBOOT):
                take = np.concatenate([idx[p] for p in rng.choice(P, size=len(P), replace=True)])
                boots[i] = -(slope(Xs[take], ys[take]) - slope(Xb[take], yb[take]))
            lo, hi = np.percentile(boots, [2.5, 97.5]); up = np.percentile(boots, 95)
            shift = up * ruler
            print("%-7s %-9s %-6s | %+8.4f %+8.4f | %+8.4f [%+.4f, %+.4f] | %+9.5f | %+.4f (x %.3f nats)"
                  % (L, C, S, bb, bs, drift, lo, hi, up, shift, ruler))
    print()
