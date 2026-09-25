#!/usr/bin/env python3
"""Make an informal blinded craft screen, one hash-selected pair per prompt.

This does not recruit raters, collect research data or establish expert validity.
Source IDs and scores stay in the separate key, never in the review document.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from phase_b_pairs import index_unique, load_rows


def build_review(items, pairs, prompts, lineage, stage):
    bank = index_unique(items, "bank")
    prompt_map = {p["id"]: p["text"] for p in prompts}
    if len(prompt_map) != len(prompts):
        raise ValueError("duplicate prompt IDs")
    groups = defaultdict(list)
    for pair in pairs:
        if (pair["b0_lineage"], pair["b0_stage"]) != (lineage, stage):
            continue
        x, y = bank[pair["item_x"]], bank[pair["item_y"]]
        if x["item_id"] == y["item_id"]:
            raise ValueError("self pair")
        for k in ("prompt_id", "lineage", "stage", "source_model"):
            if x[k] != y[k]:
                raise ValueError(f"mismatched {k}")
        if x["prompt_id"] != pair["prompt_id"] or (x["lineage"], x["stage"]) != (lineage, stage):
            raise ValueError("pair metadata does not match bank")
        groups[pair["prompt_id"]].append(pair)
    if not groups:
        raise ValueError("no review pairs for selected source")
    selected = [min(g, key=lambda p: p["b0_pair_id"]) for _, g in sorted(groups.items())]
    # Shuffle presentation deterministically without using familiarity values.
    selected.sort(key=lambda p: hashlib.sha256(("review-v1:" + p["b0_pair_id"]).encode()).hexdigest())
    text = ["# Story-pair craft screen\n",
            "Informal feasibility screen only. Read before opening the separate key. "
            "The order carries no ranking. Assess coherence, craft and whether each story "
            "answers the prompt; do not decide which is more creative in this screen.\n",
            "For each pair, note whether both are usable stories and whether one has an "
            "obvious quality advantage. Unclear is a valid answer. These are observations, "
            "not a validated rating scale or an automatic inclusion rule.\n"]
    key = []
    for n, pair in enumerate(selected, 1):
        text += [f"## Pair {n}\n", "Prompt: " + prompt_map[pair["prompt_id"]] + "\n"]
        for label, field in (("A", "item_x"), ("B", "item_y")):
            text += [f"### Story {label}\n", *["> " + line for line in bank[pair[field]]["text"].splitlines()], ""]
        text += ["Notes: __\n", "Both usable / only A / only B / neither / unclear: __\n",
                 "Craft advantage: A / B / neither obvious / unclear: __\n"]
        key.append({"review_pair": n, **pair})
    return "\n".join(text), key


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("items", "pairs", "prompts", "lineage", "stage", "out-dir"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    review, key = build_review(load_rows(args.items), load_rows(args.pairs), load_rows(args.prompts),
                               args.lineage, args.stage)
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    (out / "review.md").write_text(review)
    (out / "key.json").write_text(json.dumps(key, indent=2) + "\n")
    provenance = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                  for p in (args.items, args.pairs, args.prompts, __file__)}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"{len(key)} pairs / {2*len(key)} stories -> {out / 'review.md'}; key separate")


if __name__ == "__main__":
    main()
