#!/usr/bin/env python3
"""Phase B completion screen + cross-arm pairing (PHASE-B-protocol.md Rules 1 and 3).

Input: generate.py records from BOTH arms of one run (ordinary + unlikely), any number
of files. Output: kept/excluded ledgers with per-arm, per-reason counts; the tripwire
verdict; pairs.jsonl in judge.py --pairs-file shape (prompt_id, item_x, item_y + meta).

Everything here is mechanical. "Complete" means the stopping mechanics and the text
surface pass; it is not narrative-completion ground truth, and nothing here reads a
familiarity score or a judge output. Orientation is a stable hash of the two ids.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

MIN_WORDS = 100          # Rule 3 floor
MAX_RATIO = 1.25         # Rule 3 pairing bound (longer/shorter, in words)
TRIPWIRE_ARM = 0.10      # Rule 1: >10% excluded in either arm stops the pilot
TRIPWIRE_GAP = 0.05      # Rule 1: arms differing by >5 points stops the pilot
ARMS = ("ordinary", "unlikely")   # default; --treatment-arm swaps the second (e.g. "decoding")

TERMINAL_RE = re.compile(r'[.!?]["\'”’)\]]*\s*$')
LEAK_MARKERS = ("Writing prompt:", "Response:", "<|", "[INST", "###")
# Unfilled template slots only: "[insert ...]", a bare "[name]"/"[title]"/"{name}". A bracket that
# is part of the story ("Sincerely, [Name withheld]" in a letter, pilot ff-03) is not a slot.
PLACEHOLDER_RE = re.compile(r"\[insert\b[^\]]*\]|\[(name|title|character|your name)\]|\{[a-z_]{2,30}\}", re.I)


def word_count(text):
    return len(text.split())


def screen_record(rec, min_words=MIN_WORDS):
    """Return the list of exclusion reasons; empty means the record is complete."""
    reasons = []
    term = rec.get("termination") or {}
    params = rec.get("params") or {}
    cap = params.get("max_new_tokens")
    if term.get("stop_reason") != "eos":
        reasons.append("no_eos")
    elif cap is not None and term.get("generated_token_count", 0) >= cap:
        reasons.append("eos_at_cap")
    text = (rec.get("text") or "").strip()
    if not TERMINAL_RE.search(text):
        reasons.append("no_terminal_punctuation")
    if any(m in text for m in LEAK_MARKERS):
        reasons.append("leaked_frame")
    if PLACEHOLDER_RE.search(text):
        reasons.append("placeholder")
    if word_count(text) < min_words:
        reasons.append("below_floor")
    return reasons


def screen_bank(records, min_words=MIN_WORDS):
    """Split records into kept / excluded. Identical text within one arm+prompt is
    excluded (both copies flagged, first kept) after the per-record checks."""
    kept, excluded = [], []
    seen = {}
    for rec in records:
        arm = rec.get("elicitation", "ordinary")
        reasons = screen_record(rec, min_words)
        key = (arm, rec["prompt_id"], rec["text"].strip())
        if not reasons:
            if key in seen:
                reasons.append("identical_text")
            else:
                seen[key] = rec["item_id"]
        if reasons:
            excluded.append({"item_id": rec["item_id"], "arm": arm, "prompt_id": rec["prompt_id"],
                             "reasons": reasons, "words": word_count(rec.get("text", ""))})
        else:
            kept.append(rec)
    return kept, excluded


def exclusion_report(records, excluded, arms=ARMS):
    per_arm = {}
    for arm in arms:
        total = sum(1 for r in records if r.get("elicitation", "ordinary") == arm)
        exc = [e for e in excluded if e["arm"] == arm]
        reasons = Counter(reason for e in exc for reason in e["reasons"])
        per_arm[arm] = {"total": total, "excluded": len(exc),
                        "rate": (len(exc) / total) if total else None,
                        "reasons": dict(sorted(reasons.items()))}
    rates = [v["rate"] for v in per_arm.values() if v["rate"] is not None]
    over = any(r > TRIPWIRE_ARM for r in rates)
    gap = (max(rates) - min(rates)) if len(rates) == 2 else None
    tripped = over or (gap is not None and gap > TRIPWIRE_GAP)
    return {"per_arm": per_arm, "arm_gap": gap,
            "tripwire": {"threshold_arm": TRIPWIRE_ARM, "threshold_gap": TRIPWIRE_GAP,
                         "tripped": tripped,
                         "verdict": "TRIPWIRE_STOP_BEFORE_JUDGING" if tripped else "clear"}}


def pair_arms(kept, max_ratio=MAX_RATIO, arms=ARMS):
    """Within each prompt, pair ordinary with unlikely by closest word count, greedily,
    without reuse; reject pairs over max_ratio. Deterministic: candidates sorted by
    |log ratio| then ids."""
    if max_ratio < 1 or not math.isfinite(max_ratio):
        raise ValueError("max_ratio must be finite and >= 1")
    control, treatment = arms
    by_prompt = defaultdict(lambda: {a: [] for a in arms})
    for rec in kept:
        arm = rec.get("elicitation", "ordinary")
        if arm not in arms:
            raise ValueError(f"unknown arm {arm!r} on {rec['item_id']} (expected one of {arms})")
        by_prompt[rec["prompt_id"]][arm].append(rec)
    pairs, rejected, unmatched = [], [], []
    for pid in sorted(by_prompt):
        ords, unls = by_prompt[pid][control], by_prompt[pid][treatment]
        cands = []
        for o in ords:
            for u in unls:
                wo, wu = word_count(o["text"]), word_count(u["text"])
                cands.append((abs(math.log(wo / wu)), o["item_id"], u["item_id"], o, u, wo, wu))
        cands.sort(key=lambda c: (c[0], c[1], c[2]))
        used = set()
        for gap, oid, uid, o, u, wo, wu in cands:
            if oid in used or uid in used:
                continue
            used.update((oid, uid))
            if max(wo, wu) / min(wo, wu) > max_ratio:
                rejected.append({"prompt_id": pid, "control": oid, "treatment": uid,
                                 "words_control": wo, "words_treatment": wu})
                continue
            digest = hashlib.sha256((oid + "\0" + uid).encode()).hexdigest()
            x, y = (o, u) if int(digest, 16) % 2 == 0 else (u, o)
            pairs.append({"prompt_id": pid, "item_x": x["item_id"], "item_y": y["item_id"],
                          "pb_pair_id": digest,
                          "pb_arm_x": x.get("elicitation", "ordinary"),
                          "pb_arm_y": y.get("elicitation", "ordinary"),
                          "pb_words_x": word_count(x["text"]), "pb_words_y": word_count(y["text"]),
                          "pb_log_word_ratio": math.log(word_count(x["text"]) / word_count(y["text"]))})
        for rec in ords + unls:
            if rec["item_id"] not in used:
                unmatched.append(rec["item_id"])
    return pairs, rejected, unmatched


def token_budget_check(pairs, items_by_id, prompts, tokenizer, judge_max_len, judge_frame,
                       scorer_max_len, scorer_prefix):
    """Longest tokenized judge input (both orders, both labels) and longest scored item, so
    nothing silently truncates downstream. `tokenizer(text) -> list[int]`."""
    worst_judge, worst_item = 0, 0
    for p in pairs:
        x, y = items_by_id[p["item_x"]], items_by_id[p["item_y"]]
        ptext = prompts[p["prompt_id"]]
        for a, b in ((x, y), (y, x)):
            for crit in ("creative", "better"):
                text = judge_frame.format(prompt=ptext, a=a["text"], b=b["text"], criterion=crit)
                worst_judge = max(worst_judge, len(tokenizer(text + " A")), len(tokenizer(text + " B")))
        for it in (x, y):
            worst_item = max(worst_item, len(tokenizer(it["text"])))
    return {"judge_max_len": judge_max_len, "longest_judge_input": worst_judge,
            "judge_fits": worst_judge <= judge_max_len,
            "scorer_max_len": scorer_max_len, "longest_item": worst_item,
            "scorer_fits": worst_item + len(tokenizer(scorer_prefix)) <= scorer_max_len}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records", nargs="+", required=True, help="generate.py outputs, both arms")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-words", type=int, default=MIN_WORDS)
    ap.add_argument("--max-ratio", type=float, default=MAX_RATIO)
    ap.add_argument("--treatment-arm", default="unlikely", help="elicitation tag of the treatment arm; control is always ordinary")
    args = ap.parse_args()
    arms = ("ordinary", args.treatment_arm)
    records = []
    for path in args.records:
        records.extend(json.loads(l) for l in Path(path).read_text().splitlines() if l.strip())
    ids = [r["item_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate item_id across record files")
    kept, excluded = screen_bank(records, args.min_words)
    report = exclusion_report(records, excluded, arms)
    pairs, rejected, unmatched = pair_arms(kept, args.max_ratio, arms)
    report.update({"arms": list(arms), "min_words": args.min_words, "max_ratio": args.max_ratio,
                   "records": len(records), "kept": len(kept), "pairs": len(pairs),
                   "pairs_rejected_by_ratio": len(rejected), "unmatched_ids": unmatched,
                   "input_sha256": {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                    for p in args.records},
                   "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "status": "MECHANICAL_SCREEN_ONLY_NOT_QUALITY_VALIDATED"})
    out = Path(args.out_dir)
    if out.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {out}")
    out.mkdir(parents=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in pairs))
    (out / "excluded.jsonl").write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in excluded))
    (out / "rejected_pairs.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rejected))
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
