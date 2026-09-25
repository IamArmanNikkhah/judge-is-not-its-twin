#!/usr/bin/env python
"""Item-bank construction — merge generation runs into one judgeable bank.

judge.py consumes a single items JSONL and builds within-prompt pairs from it.
This is the step that produces that file.

ON THE LENGTH FILTER — the reason the default is as weak as it is:
the measured length distribution over the OLMo-2 pilot (800 records) has NO
natural break. It is smooth from 0 chars up through the tail (11 empty, 14 <20,
20 <40, 42 <100, 65 <200, median ~960), so any positive cut is a researcher
degree of freedom, not something the data picked. Worse, it is not a NEUTRAL
one for this study: short completions plausibly sit at unusual perplexity, so
filtering on length conditions the sample on a quantity correlated with the
typicality measure that is the dependent variable. Default therefore drops only
what is undefined rather than merely short — an empty string has no artifact to
judge — and any positive --min-chars is a preregistered sensitivity pass that
must be reported, never a silent default. The drop table prints at every run so
the choice stays auditable either way.

Usage:
  python build_bank.py --runs ../data/runs/gen-olmo2-*.jsonl ../data/runs/gen-zephyr-*.jsonl \
      --out ../data/items/bank-pilot.jsonl [--min-chars 0]
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

REQUIRED = ("item_id", "prompt_id", "text", "source_model", "lineage", "stage")

# Reported at every run so a chosen cut can be read against the ones not chosen.
DROP_TABLE_THRESHOLDS = (1, 20, 40, 60, 100, 150, 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="generation JSONL files to merge")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=0,
                    help="drop completions shorter than this AFTER stripping; 0 = empty-only "
                         "(the default; any positive value is a sensitivity pass, report it)")
    args = ap.parse_args()

    records, seen_ids, malformed = [], set(), 0
    for path in args.runs:
        p = Path(path)
        if not p.exists():
            sys.exit(f"missing run file: {p}")
        for line in open(p):
            r = json.loads(line)
            if any(k not in r for k in REQUIRED):
                malformed += 1
                continue
            if r["item_id"] in seen_ids:      # same run banked twice — keep one
                continue
            seen_ids.add(r["item_id"])
            records.append(r)

    lengths = sorted(len(r["text"].strip()) for r in records)
    print(f"read {len(records)} records from {len(args.runs)} file(s)"
          + (f" ({malformed} malformed, skipped)" if malformed else ""), file=sys.stderr)
    print("drop table (what each candidate cut would remove):", file=sys.stderr)
    for t in DROP_TABLE_THRESHOLDS:
        n = sum(1 for x in lengths if x < t)
        mark = "  <-- applied" if t == args.min_chars else ""
        print(f"  chars < {t:4d}: {n:4d}/{len(lengths)}{mark}", file=sys.stderr)

    cut = max(args.min_chars, 1)   # an empty string is never a judgeable artifact
    kept = [r for r in records if len(r["text"].strip()) >= cut]
    dropped = Counter((r["lineage"], r["stage"]) for r in records
                      if len(r["text"].strip()) < cut)

    print(f"applied cut: >= {cut} chars ({'empty-only default' if args.min_chars == 0 else 'sensitivity pass'})",
          file=sys.stderr)
    if dropped:
        print("dropped by source:", file=sys.stderr)
        for (lin, stg), n in sorted(dropped.items()):
            print(f"  {lin}-{stg}: {n}", file=sys.stderr)

    # Self-pair exclusion is whole-LINEAGE, so a single-lineage bank yields zero
    # pairs for any judge drawn from it — fail loudly here, not inside judge.py.
    by_lineage = Counter(r["lineage"] for r in kept)
    if len(by_lineage) < 2:
        print(f"WARNING: bank has only one lineage ({list(by_lineage)}) — every judge from "
              "that lineage will produce ZERO pairs. Add a second family before judging.",
              file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kept.sort(key=lambda r: (r["prompt_id"], r["lineage"], r["stage"], r["item_id"]))
    with open(out_path, "w") as out:
        for r in kept:
            out.write(json.dumps({k: r[k] for k in REQUIRED}) + "\n")

    print(f"\nbank: {len(kept)} items -> {out_path}", file=sys.stderr)
    print("  by lineage: " + ", ".join(f"{k}={v}" for k, v in sorted(by_lineage.items())), file=sys.stderr)
    per_prompt = Counter(r["prompt_id"] for r in kept)
    print(f"  prompts: {len(per_prompt)}, items/prompt min={min(per_prompt.values())} "
          f"max={max(per_prompt.values())}", file=sys.stderr)


if __name__ == "__main__":
    main()
