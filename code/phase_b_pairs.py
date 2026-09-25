#!/usr/bin/env python3
"""B0 retrospective feasibility: disjoint within-writer pairs, no judge inputs.

Pairs are adjacent in character length within prompt/checkpoint. This is a
deterministic descriptive baseline, NOT a quality match or causal manipulation.
No familiarity-gap cut or judge-output selection is permitted here. New runs
need their own prospective protocol; the Phase A bank remains untouched.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def index_unique(rows, label):
    out = {}
    for row in rows:
        ident = row.get("item_id")
        if not isinstance(ident, str) or not ident or ident in out:
            raise ValueError(f"{label}: missing/duplicate item_id {ident!r}")
        out[ident] = row
    return out


def quantiles(values):
    values = sorted(values)
    if not values:
        return {"n": 0}
    def q(p):
        pos = (len(values) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"n": len(values), "min": values[0], "q25": q(.25),
            "median": statistics.median(values), "q75": q(.75), "max": values[-1]}


def prepare(items, score_sets, references, max_length_ratio=None):
    if max_length_ratio is not None and (not math.isfinite(max_length_ratio) or max_length_ratio < 1):
        raise ValueError("maximum length ratio must be finite and >= 1")
    bank = index_unique(items, "bank")
    if not bank:
        raise ValueError("empty bank")
    refs = {}
    for tag, rows in score_sets.items():
        refs[tag] = index_unique(rows, tag)
        if set(refs[tag]) != set(bank):
            raise ValueError(f"{tag}: score IDs must exactly cover the bank")
        models = {r.get("ref_model") for r in rows}
        if len(models) != 1 or not next(iter(models)):
            raise ValueError(f"{tag}: mixed/missing reference model")
        for ident, row in refs[tag].items():
            if row.get("frame") != "completion-prefix" or row.get("ref_tag") != tag:
                raise ValueError(f"{tag}: wrong frame/tag for {ident}")
            value = row.get("mean_nll")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{tag}: nonfinite/missing score for {ident}")
            if row.get("n_tokens", 0) <= 0 or row.get("n_chars") != len(bank[ident]["text"]):
                raise ValueError(f"{tag}: token/character provenance mismatch for {ident}")
    groups = defaultdict(list)
    for item in bank.values():
        for key in ("prompt_id", "source_model", "lineage", "stage", "text"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"invalid bank {key}: {item['item_id']}")
        tag = references.get(item["lineage"])
        if tag not in refs:
            raise ValueError(f"no reference for {item['lineage']}")
        if refs[tag][item["item_id"]]["ref_model"] == item["source_model"]:
            raise ValueError("source model cannot be its own held-out reference")
        groups[(item["lineage"], item["stage"], item["source_model"], item["prompt_id"])].append(item)
    pairs, unmatched, identical, length_rejected = [], [], [], []
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: (len(r["text"]), r["item_id"]))
        if len(group) % 2:
            unmatched.append(group[-1]["item_id"])
        for a, b in zip(group[::2], group[1::2]):
            if a["text"].strip() == b["text"].strip():
                identical.append([a["item_id"], b["item_id"]])
                continue
            if max_length_ratio is not None and len(b["text"]) / len(a["text"]) > max_length_ratio:
                length_rejected.append([a["item_id"], b["item_id"]])
                continue
            # Orientation is independent of length, familiarity and judge results.
            digest = hashlib.sha256((a["item_id"] + "\0" + b["item_id"]).encode()).hexdigest()
            x, y = (a, b) if int(digest, 16) % 2 == 0 else (b, a)
            tag = references[x["lineage"]]
            sx, sy = refs[tag][x["item_id"]], refs[tag][y["item_id"]]
            pairs.append({"prompt_id": x["prompt_id"], "item_x": x["item_id"],
                          "item_y": y["item_id"], "b0_pair_id": digest,
                          "b0_lineage": x["lineage"], "b0_stage": x["stage"],
                          "b0_source_model": x["source_model"], "b0_ref_tag": tag,
                          "b0_nll_x": sx["mean_nll"], "b0_nll_y": sy["mean_nll"],
                          "b0_delta_nll": sx["mean_nll"] - sy["mean_nll"],
                          "b0_abs_delta_log_chars": abs(math.log(len(x["text"]) / len(y["text"]))),
                          "b0_chars_x": len(x["text"]), "b0_chars_y": len(y["text"])})
    cells = []
    for lin, stage in sorted({(i["lineage"], i["stage"]) for i in items}):
        cell = [p for p in pairs if (p["b0_lineage"], p["b0_stage"]) == (lin, stage)]
        cells.append({"lineage": lin, "stage": stage, "pairs": len(cell),
                      "prompts": len({p["prompt_id"] for p in cell}),
                      "abs_delta_nll": quantiles([abs(p["b0_delta_nll"]) for p in cell]),
                      "abs_delta_log_chars": quantiles([p["b0_abs_delta_log_chars"] for p in cell]),
                      "min_pair_chars": quantiles([min(p["b0_chars_x"], p["b0_chars_y"]) for p in cell])})
    report = {"status": "RETROSPECTIVE_FEASIBILITY_ONLY_NOT_QUALITY_VALIDATED",
              "selection": "adjacent character lengths; disjoint within prompt/model/lineage/stage; no gap cut",
              "items": len(bank), "groups": len(groups), "pairs": len(pairs),
              "all_within_group_candidates": sum(len(g)*(len(g)-1)//2 for g in groups.values()),
              "unmatched_ids": unmatched, "identical_text_pairs_skipped": identical,
              "max_length_ratio": max_length_ratio, "length_rejected_pairs": length_rejected,
              "items_per_group": dict(sorted(Counter(map(len, groups.values())).items())),
              "references": references, "cells": cells,
              "limitations": ["No independent quality ratings; nonempty is not good.",
                              "Known Phase A items and prompts, not a fresh confirmatory sample.",
                              "No new generations or judges run; no causal or equivalence claim.",
                              "Reference log probability is a familiarity proxy, not semantic novelty.",
                              "Judge --pairs-file bypasses lineage exclusion; split by opposite lineage before a run.",
                              "No numeric quality, gap, power or competence gate ratified for B0."]}
    return pairs, report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--items", required=True)
    ap.add_argument("--scores", nargs="+", required=True, help="TAG=/absolute/path.jsonl")
    ap.add_argument("--reference", nargs="+", required=True, help="LINEAGE=TAG")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-length-ratio", type=float, help="explicit exploratory bound; no ratified default")
    args = ap.parse_args()
    paths, refs = {}, {}
    for value, target in [(x, paths) for x in args.scores] + [(x, refs) for x in args.reference]:
        key, val = value.split("=", 1)
        if key in target:
            raise ValueError(f"duplicate mapping {key}")
        target[key] = val
    pairs, report = prepare(load_rows(args.items), {k: load_rows(v) for k, v in paths.items()}, refs,
                            args.max_length_ratio)
    report["input_sha256"] = {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                              for p in [args.items, *paths.values()]}
    report["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in pairs))
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
