#!/usr/bin/env python
"""Merge + first-pass analysis (v0 — will be FROZEN after pilot review, before full runs).

Inputs: generation JSONLs (items), typicality JSONLs (one per reference model),
judge JSONLs (one per judge stage). Outputs, per judge:
  n_pairs, typicality-preference beta (does p_x rise when x is more typical
  than y?), mean |order-swap spread| (position-bias magnitude).
Per item: shared_nll averaged over references (leave-own-lineage-out handled
upstream by which refs were run on which items).

beta here is the RAW typicality preference (logistic slope of choice on
Δtypicality). The paper's J subtracts the anchor panel's slope on identical
pairs — that term enters once anchor data exists (crowd pilot / experts).

Usage:
  python analyze.py --items <gen1.jsonl> [<gen2.jsonl> ...] \
      --typ <typ1.jsonl> [<typ2.jsonl> ...] --judges <judge1.jsonl> [...] \
      [--out summary.json]
"""
import argparse, json, math, sys
from collections import defaultdict


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def logistic_slope(xs, ps, iters=500, lr=0.5):
    """1-D logistic regression of choice prob on standardized delta; returns beta.
    Uses p_x as a soft target (each pair contributes fractionally); no intercept
    beyond the position-neutral construction (p already order-averaged)."""
    n = len(xs)
    if n < 3:
        return None
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / max(n - 1, 1)) or 1.0
    zs = [(x - mu) / sd for x in xs]
    b0, b1 = 0.0, 0.0
    for _ in range(iters):
        g0 = g1 = 0.0
        for z, p in zip(zs, ps):
            pred = 1.0 / (1.0 + math.exp(-(b0 + b1 * z)))
            g0 += pred - p
            g1 += (pred - p) * z
        b0 -= lr * g0 / n
        b1 -= lr * g1 / n
    return b1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--typ", nargs="+", required=True)
    ap.add_argument("--judges", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = {}
    for p in args.items:
        for it in load_jsonl(p):
            items[it["item_id"]] = it

    # shared_nll = mean over reference models that scored the item
    nll_acc = defaultdict(list)
    for p in args.typ:
        for r in load_jsonl(p):
            nll_acc[r["item_id"]].append(r["mean_nll"])
    shared_nll = {k: sum(v) / len(v) for k, v in nll_acc.items()}

    summary = {"n_items": len(items), "n_items_with_typicality": len(shared_nll), "judges": []}
    for p in args.judges:
        rows = load_jsonl(p)
        xs, ps, spreads = [], [], []
        for r in rows:
            nx, ny = shared_nll.get(r["item_x"]), shared_nll.get(r["item_y"])
            if nx is None or ny is None:
                continue
            # typicality = -nll; delta>0 means x is MORE typical than y
            xs.append(ny - nx)
            ps.append(r["p_x"])
            spreads.append(abs(r["p_x_first_order"] - (1 - r["p_y_first_order"])))
        beta = logistic_slope(xs, ps)
        summary["judges"].append({
            "judge_model": rows[0]["judge_model"] if rows else None,
            "judge_lineage": rows[0]["judge_lineage"] if rows else None,
            "judge_stage": rows[0]["judge_stage"] if rows else None,
            "criterion": rows[0]["criterion"] if rows else None,
            "n_pairs_scored": len(xs),
            "typicality_pref_beta": None if beta is None else round(beta, 4),
            "mean_order_swap_spread": None if not spreads else round(sum(spreads) / len(spreads), 4),
        })

    out = json.dumps(summary, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
