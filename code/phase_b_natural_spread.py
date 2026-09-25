#!/usr/bin/env python3
"""Phase B path (a) feasibility: is there REAL typicality spread among ordinary stories?

After the T=1.3 arm failed the author's blind read (2026-09-24), the candidate replacement is a
within-arm contrast: pair the more and less predictable ordinary stories of one prompt. The
naive check (most minus least predictable) is positive by construction, so here selection and
measurement use DIFFERENT tokens. Rule written in the task manifest before any number existed:

- per candidate pair, m = min(token counts), so both members cover the same token positions
  (later tokens are more predictable, which would hand the gap to the longer story);
  half 1 = tokens [0, m//2), half 2 = [m//2, m);
- disjoint pairs within prompt, word ratio <= MAX_RATIO (Rule 3), maximizing summed |gap| on
  the selection half; typical member = lower NLL there; held-out gap = NLL(atypical) minus
  NLL(typical) on the other half; both directions, averaged per prompt;
- bootstrap over prompts; the spread is REAL iff the 95% lower bound is above zero.

No judge output and no quality label is read. Whether the less predictable story is also the
worse one is NOT answerable here.
"""
import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import statistics

from phase_b_screen import MAX_RATIO, word_count


def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def matchings(ids, max_ratio, words):
    """Largest sets of disjoint Rule-3 pairs among ids (all of them, for the caller to rank)."""
    ok = [p for p in combinations(sorted(ids), 2)
          if max(words[p[0]], words[p[1]]) / min(words[p[0]], words[p[1]]) <= max_ratio]
    for k in range(len(ids) // 2, 0, -1):
        found = [c for c in combinations(ok, k) if len({i for p in c for i in p}) == 2 * k]
        if found:
            return found
    return []


def halves(a, b):
    """Positions-matched half means for two per-token NLL lists: ((a1, b1), (a2, b2))."""
    m = min(len(a), len(b))
    h = m // 2
    if h < 1:
        raise ValueError("story too short to split")
    return ((statistics.fmean(a[:h]), statistics.fmean(b[:h])),
            (statistics.fmean(a[h:m]), statistics.fmean(b[h:m])))


def prompt_contrast(ids, tok, words, max_ratio=MAX_RATIO):
    """Cross-fitted held-out gaps for one prompt. Returns (per-prompt stat, chosen pair rows)."""
    cands = matchings(ids, max_ratio, words)
    if not cands:
        return None, []
    rows = []
    for sel, held in ((0, 1), (1, 0)):
        def sel_gap(p):
            (a1, b1), (a2, b2) = halves(tok[p[0]], tok[p[1]])
            return abs(a1 - b1) if sel == 0 else abs(a2 - b2)
        # cands come out of combinations() over sorted ids, and max() keeps the first of a tie
        best = max(cands, key=lambda c: sum(sel_gap(p) for p in c))
        for a, b in best:
            hv = halves(tok[a], tok[b])
            typ, atyp = (a, b) if hv[sel][0] <= hv[sel][1] else (b, a)
            s = {a: hv[sel][0], b: hv[sel][1]}
            h = {a: hv[held][0], b: hv[held][1]}
            rows.append({"selected_on": f"half{sel + 1}", "typical": typ, "atypical": atyp,
                         "in_sample_gap": s[atyp] - s[typ], "held_out_gap": h[atyp] - h[typ],
                         "typical_is_longer": words[typ] > words[atyp],
                         "word_ratio": max(words[a], words[b]) / min(words[a], words[b]),
                         "positions_used": min(len(tok[a]), len(tok[b]))})
    return statistics.fmean(r["held_out_gap"] for r in rows), rows


def bootstrap(values, n_boot=2000, seed=411):
    rng = random.Random(seed)
    boots = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(n_boot))
    return [boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", nargs="+", required=True, help="generate.py outputs.jsonl, ordinary arm")
    ap.add_argument("--scores", required=True, help="typicality.py output with --save-token-nll")
    ap.add_argument("--excluded", required=True, help="phase_b_screen.py excluded.jsonl")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    items = {r["item_id"]: r for p in args.items for r in load_rows(p)}
    excluded = {r["item_id"] for r in load_rows(args.excluded)}
    if excluded & set(items):
        raise ValueError(f"screen-excluded items among inputs: {sorted(excluded & set(items))}")
    scores = {r["item_id"]: r for r in load_rows(args.scores) if r["item_id"] in items}
    if set(scores) != set(items):
        raise ValueError(f"unscored items: {sorted(set(items) - set(scores))}")
    for s in scores.values():
        if s.get("frame") != "completion-prefix" or s.get("truncated") or len(s["token_nll"]) != s["n_tokens"]:
            raise ValueError(f"{s['item_id']}: needs untruncated completion-prefix token_nll")
    tags = {s["ref_tag"] for s in scores.values()}
    if len(tags) != 1:
        raise ValueError(f"scores mix readers: {sorted(tags)}")
    tok = {i: scores[i]["token_nll"] for i in items}
    words = {i: word_count(items[i]["text"]) for i in items}
    by_prompt = defaultdict(list)
    for i, r in items.items():
        by_prompt[r["prompt_id"]].append(i)
    per_prompt, pairs = {}, []
    for pid in sorted(by_prompt):
        stat, rows = prompt_contrast(by_prompt[pid], tok, words)
        pairs += [dict(r, prompt_id=pid) for r in rows]
        if stat is not None:
            per_prompt[pid] = stat
    vals = list(per_prompt.values())
    lo, hi = bootstrap(vals)
    report = {"ref_tag": tags.pop(), "items": len(items), "prompts": len(by_prompt),
              "prompts_with_pairs": len(vals), "pairs_per_direction": len(pairs) // 2,
              "mean_held_out_gap": statistics.fmean(vals), "bootstrap_95": [lo, hi],
              "spread_real": lo > 0,
              "prompts_positive": sum(v > 0 for v in vals),
              "mean_in_sample_gap": statistics.fmean(r["in_sample_gap"] for r in pairs),
              "frac_typical_is_longer": sum(r["typical_is_longer"] for r in pairs) / len(pairs),
              "per_prompt_held_out_gap": per_prompt,
              "reference_t13_arm_delta": 0.7675955,
              "reading": "held-out gap in nats/token on tokens NOT used to pick the pair; the T=1.3 "
                         "delta is full-story and cross-arm, so the comparison is rough",
              "status": "FEASIBILITY_ONLY_NO_QUALITY_LABELS",
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "input_sha256": {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                               for p in [*args.items, args.scores, args.excluded]}}
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in pairs))
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("input_sha256", "per_prompt_held_out_gap")},
                     indent=2, sort_keys=True))
    print("per prompt:", {k: round(v, 3) for k, v in per_prompt.items()})


if __name__ == "__main__":
    main()
