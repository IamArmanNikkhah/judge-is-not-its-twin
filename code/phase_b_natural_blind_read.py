#!/usr/bin/env python3
"""Blind read for path (a): is the LESS predictable of two ordinary stories still a good story?

A quality stress test before any judge sees natural pairs. Rule written in the task manifest
before any story was picked:

- per prompt, among Rule-3 pairs that contain no story the author already graded, the pair with the
  widest full-story gap, positions-matched (mean NLL over the first m = min tokens, since later
  tokens are more predictable); then the N prompts with the widest such gaps. Widest on purpose:
  if quality rides with predictability it shows most there. These are NOT the judge pairs.
- orientation balanced (half the pairs put the predictable story in A), presentation order on a
  separate hash, and the review document carries no hint of the contrast. Everything else lives
  in the key.
"""
import argparse
from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import statistics

from phase_b_natural_spread import load_rows
from phase_b_screen import MAX_RATIO, word_count


def full_gap(a, b):
    """Positions-matched mean NLL for two per-token lists: (mean_a, mean_b, m)."""
    m = min(len(a), len(b))
    return statistics.fmean(a[:m]), statistics.fmean(b[:m]), m


def widest_pairs(by_prompt, tok, words, graded, n, max_ratio=MAX_RATIO):
    best = []
    for pid in sorted(by_prompt):
        cands = []
        for a, b in combinations(sorted(by_prompt[pid]), 2):
            if a in graded or b in graded:
                continue
            if max(words[a], words[b]) / min(words[a], words[b]) > max_ratio:
                continue
            na, nb, m = full_gap(tok[a], tok[b])
            typ, atyp = (a, b) if na <= nb else (b, a)
            cands.append({"prompt_id": pid, "more_predictable": typ, "less_predictable": atyp,
                          "gap_nll": abs(na - nb), "positions_used": m,
                          "words": {typ: words[typ], atyp: words[atyp]}})
        if cands:
            best.append(max(cands, key=lambda c: c["gap_nll"]))   # first of a tie, sorted ids
    best.sort(key=lambda c: (-c["gap_nll"], c["prompt_id"]))
    if len(best) < n:
        raise ValueError(f"only {len(best)} prompts have an eligible pair, need {n}")
    return best[:n]


def digest(salt, c):
    return hashlib.sha256((salt + "\0" + c["more_predictable"] + "\0" + c["less_predictable"]).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", nargs="+", required=True, help="generate.py outputs.jsonl, ordinary arm")
    ap.add_argument("--scores", required=True, help="typicality.py output with --save-token-nll")
    ap.add_argument("--excluded", required=True, help="phase_b_screen.py excluded.jsonl")
    ap.add_argument("--prompts", required=True, help="data/items/prompts.jsonl")
    ap.add_argument("--graded-key", required=True, help="an earlier blind-read KEY; its stories are skipped")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if args.n % 2:
        raise ValueError("--n must be even so orientation can balance")
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
    if len({s["ref_tag"] for s in scores.values()}) != 1:
        raise ValueError("scores mix readers")
    graded = {i for k, pair in json.loads(Path(args.graded_key).read_text()).items()
              if k.startswith("pair") for i in pair["ids"]}
    prompts = {p["id"]: p["text"] for p in load_rows(args.prompts)}
    tok = {i: scores[i]["token_nll"] for i in items}
    words = {i: word_count(items[i]["text"]) for i in items}
    by_prompt = defaultdict(list)
    for i, r in items.items():
        by_prompt[r["prompt_id"]].append(i)

    chosen = widest_pairs(by_prompt, tok, words, graded, args.n)
    for k, c in enumerate(sorted(chosen, key=lambda c: digest("orient-v1", c))):
        c["A"], c["B"] = ((c["more_predictable"], c["less_predictable"]) if k < args.n // 2
                          else (c["less_predictable"], c["more_predictable"]))
    chosen.sort(key=lambda c: digest("present-v1", c))

    doc = [f"# Blind read: {args.n} pairs (labels in blind-read-KEY.json, do not open first)\n",
           "For each pair: is each text a coherent, complete story that fits the prompt? "
           "Score each 1-5. Then which is better, which is more creative.\n"]
    key = {}
    for k, c in enumerate(chosen, 1):
        doc.append(f"## Pair {k} — prompt: {prompts[c['prompt_id']]}\n")
        for side in ("A", "B"):
            doc.append(f"### {side}\n\n{items[c[side]]['text'].strip()}\n")
        doc.append("---\n")
        key[f"pair {k}"] = {"A": "more_predictable" if c["A"] == c["more_predictable"] else "less_predictable",
                            "B": "more_predictable" if c["B"] == c["more_predictable"] else "less_predictable",
                            "ids": [c["A"], c["B"]], "prompt_id": c["prompt_id"],
                            "gap_nll": c["gap_nll"], "positions_used": c["positions_used"],
                            "words": [words[c["A"]], words[c["B"]]]}
    meta = {"rule": "widest positions-matched Rule-3 pair per prompt, skipping graded stories; top N prompts; "
                    "gate: every less-predictable story >= 3 coherence AND median >= 4",
            "ref_tag": next(iter(scores.values()))["ref_tag"], "skipped_graded": sorted(graded),
            "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                             for p in [*args.items, args.scores, args.excluded, args.prompts, args.graded_key]}}
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    (out / "blind-read.md").write_text("\n".join(doc))
    (out / "blind-read-KEY.json").write_text(json.dumps({**key, "_meta": meta}, indent=1) + "\n")
    print(f"wrote {out}/blind-read.md ({args.n} pairs) and the key; prompts:",
          [c["prompt_id"] for c in chosen])


if __name__ == "__main__":
    main()
