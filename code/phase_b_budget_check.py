#!/usr/bin/env python3
"""Rule 2 pre-flight: tokenize the screened pairs under every intended reader/judge tokenizer
and confirm nothing will truncate. Runs on the Mac (tokenizers only, no weights, no GPU).

  python phase_b_budget_check.py --pairs <screen>/pairs.jsonl --records <arm files...> \
      --tokenizers HuggingFaceH4/zephyr-7b-beta allenai/OLMo-2-1124-7B mistralai/Mistral-7B-v0.1 \
      --judge-max-len 2048 --scorer-max-len 1024 --out <screen>/budget.json
Exit 1 if any tokenizer does not fit; the pilot does not proceed to scoring on a failed check.
"""
import argparse
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

from judge import FRAME as JUDGE_FRAME
from typicality import COMPLETION_FRAME
from phase_b_screen import token_budget_check

ROOT = Path(__file__).resolve().parent.parent


def load_rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--tokenizers", nargs="+", required=True)
    ap.add_argument("--judge-max-len", type=int, default=2048)
    ap.add_argument("--scorer-max-len", type=int, default=1024)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs = load_rows(args.pairs)
    items = {r["item_id"]: r for p in args.records for r in load_rows(p)}
    prompts = {p["id"]: p["text"] for p in load_rows(ROOT / "data/items/prompts.jsonl")}
    longest_prompt = max(prompts[p["prompt_id"]] for p in pairs) if pairs else ""
    report, ok = {}, True
    for name in args.tokenizers:
        tok = AutoTokenizer.from_pretrained(name)
        rep = token_budget_check(pairs, items, prompts, lambda s: tok(s)["input_ids"],
                                 args.judge_max_len, JUDGE_FRAME, args.scorer_max_len,
                                 COMPLETION_FRAME.format(prompt=longest_prompt))
        report[name] = rep
        ok = ok and rep["judge_fits"] and rep["scorer_fits"]
    report["_verdict"] = "fits" if ok else "TRUNCATION_WOULD_OCCUR"
    report["_pairs"] = len(pairs)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
