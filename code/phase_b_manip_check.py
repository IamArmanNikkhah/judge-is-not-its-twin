#!/usr/bin/env python3
"""Phase B manipulation check: is the unlikely arm actually less typical? (protocol §Manipulation check)

Positive control BEFORE any judging. On the screened pairs, compare each pair's unlikely
story to its ordinary story under one cross-family base reader (completion-prefix frame).
Reports the mean paired difference in mean per-token NLL (unlikely minus ordinary; positive
means the manipulation worked), the fraction of pairs where unlikely is less typical, and
a paired bootstrap interval. If the interval covers zero, Option B's wording is revised,
not the judges. No judge output is read here.
"""
import argparse
import json
from pathlib import Path
import random
import statistics


def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def paired_deltas(pairs, scores):
    """scores: item_id -> typicality row. Returns list of (treatment_nll - ordinary_nll)."""
    deltas = []
    for p in pairs:
        arms = {p["pb_arm_x"]: p["item_x"], p["pb_arm_y"]: p["item_y"]}
        if "ordinary" not in arms or len(arms) != 2:
            raise ValueError(f"pair {p.get('pb_pair_id')} does not span ordinary + one treatment arm")
        treatment = next(a for a in arms if a != "ordinary")
        so, su = scores[arms["ordinary"]], scores[arms[treatment]]
        for s in (so, su):
            if s.get("frame") != "completion-prefix":
                raise ValueError(f"{s['item_id']}: scored under {s.get('frame')!r}, need completion-prefix")
            if s.get("truncated"):
                raise ValueError(f"{s['item_id']}: truncated score; Rule 2 forbids using it")
        deltas.append(su["mean_nll"] - so["mean_nll"])
    return deltas


def summarize(deltas, n_boot=2000, seed=411):
    if not deltas:
        raise ValueError("no pairs")
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(deltas, k=len(deltas))) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    return {"n_pairs": len(deltas), "mean_delta_nll": statistics.fmean(deltas),
            "frac_treatment_less_typical": sum(d > 0 for d in deltas) / len(deltas),
            "bootstrap_95": [lo, hi],
            "manipulation_detected": lo > 0,
            "reading": "positive delta = treatment arm is LESS typical to the reader (the intended direction)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True, help="phase_b_screen.py pairs.jsonl")
    ap.add_argument("--scores", required=True, help="typicality.py output, one cross-family base reader")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    scores = {r["item_id"]: r for r in load_rows(args.scores)}
    tags = {r["ref_tag"] for r in scores.values()}
    if len(tags) != 1:
        raise ValueError(f"scores mix readers: {sorted(tags)}")
    result = summarize(paired_deltas(load_rows(args.pairs), scores))
    result["ref_tag"] = tags.pop()
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
