#!/usr/bin/env python3
"""Retokenized-length distributions; NOT recovered generation stop reasons."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from phase_b_pairs import index_unique, load_rows, quantiles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    items, rows = load_rows(args.items), load_rows(args.scores)
    bank, scores = index_unique(items, "bank"), index_unique(rows, "scores")
    if set(bank) != set(scores):
        raise ValueError("bank and score IDs differ")
    groups = defaultdict(list)
    for item in items:
        score = scores[item["item_id"]]
        if score["n_chars"] != len(item["text"]) or score["frame"] != "completion-prefix":
            raise ValueError("score frame/length mismatch")
        groups[(item["lineage"], item["stage"], item["source_model"])].append(item)
    cells = []
    for (lineage, stage, model), group in sorted(groups.items()):
        refs = {scores[x["item_id"]]["ref_model"] for x in group}
        if len(refs) != 1:
            raise ValueError("mixed reference models")
        ref = next(iter(refs))
        cells.append({"lineage": lineage, "stage": stage, "n": len(group), "source_model": model,
                      "ref_model": ref, "same_checkpoint_reader": model == ref,
                      "characters": quantiles([len(x["text"]) for x in group]),
                      "scored_token_histogram": dict(sorted(Counter(
                          scores[x["item_id"]]["n_tokens"] for x in group).items()))})
    report = {"interpretation": "Tokenized decoded/stripped text, not original generated IDs; no cap-hit prevalence claim.",
              "cells": cells, "input_sha256": {
                  str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                  for p in (args.items, args.scores, __file__)}}
    path = Path(args.out)
    with path.open("x") as out:
        json.dump(report, out, indent=2)
        out.write("\n")
    print(f"{len(cells)} distributions -> {path}")


if __name__ == "__main__":
    main()
