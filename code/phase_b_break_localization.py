#!/usr/bin/env python3
"""Does the reader's surprise sit where a story's logic breaks? (Phase B write-up illustration)

Break sentences are marked by reading, committed BEFORE any per-sentence score (marks.json).
Per story: map each scored token back to its sentence (same tokenizer, add_special_tokens=False,
exactly as typicality.py scored the completion-prefix frame), take mean per-token NLL per
sentence, and report where each marked sentence ranks within its own story. Three stories and
seven marked sentences: an illustration, not a test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

from tokenizers import Tokenizer

from phase_b_natural_spread import load_rows

SENT_RE = re.compile(r'[^.!?]+(?:[.!?]+["\'”’)\]]*|$)')


def sentences(text):
    """(start, end, sentence) spans; a sentence ends at terminal punctuation plus closing quotes."""
    out = []
    for m in SENT_RE.finditer(text):
        s, e = m.start(), m.end()
        while s < e and text[s].isspace():
            s += 1
        if s < e:
            out.append((s, e, text[s:e].strip()))
    return out


def per_sentence(text, token_nll, tok):
    enc = tok.encode(text, add_special_tokens=False)
    if len(enc.ids) != len(token_nll):
        raise ValueError(f"token count {len(enc.ids)} != scored {len(token_nll)}")
    spans = sentences(text)
    buckets = [[] for _ in spans]
    for (a, b), v in zip(enc.offsets, token_nll):
        mid = a if not text[a:b].strip() else a + len(text[a:b]) - len(text[a:b].lstrip())
        for i, (s, e, _) in enumerate(spans):
            if s <= mid < e:
                buckets[i].append(v)
                break
    return [{"sentence": sent, "n_tokens": len(b), "mean_nll": statistics.fmean(b) if b else None}
            for (_, _, sent), b in zip(spans, buckets)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--marks", required=True)
    ap.add_argument("--analysis-dir", required=True, help="data/analysis/phase-b-pilot-2026-09-05")
    ap.add_argument("--items", nargs="+", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json of the OLMo-2-1124 family")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    items = {r["item_id"]: r for p in args.items for r in load_rows(p)}
    scores = {r["item_id"]: r for r in load_rows(args.scores)}
    marks = json.loads(Path(args.marks).read_text())
    report = {"stories": [], "marks_sha256": hashlib.sha256(Path(args.marks).read_bytes()).hexdigest(),
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for st in marks["stories"]:
        key = json.loads((Path(args.analysis_dir) / st["source"]).read_text())
        pair = key[st["pair"]]
        iid = pair["ids"][0 if st["side"] == "A" else 1]
        text, sc = items[iid]["text"], scores[iid]
        if sc["frame"] != "completion-prefix" or sc["truncated"]:
            raise ValueError(f"{iid}: need untruncated completion-prefix scores")
        rows = [r for r in per_sentence(text, sc["token_nll"], tok) if r["mean_nll"] is not None]
        ranked = sorted(rows, key=lambda r: r["mean_nll"], reverse=True)
        marked = []
        for b in st["breaks"]:
            hits = [r for r in rows if b["sentence_contains"] in r["sentence"]]
            if len(hits) != 1:
                raise ValueError(f"{st['label']}: mark {b['sentence_contains']!r} matched {len(hits)} sentences")
            r = hits[0]
            rank = ranked.index(r) + 1
            marked.append({"mark": b["sentence_contains"], "rank": rank, "of": len(rows),
                           "top_quarter": rank <= max(1, round(len(rows) / 4)),
                           "mean_nll": r["mean_nll"], "n_tokens": r["n_tokens"]})
        report["stories"].append({"label": st["label"], "item_id": iid, "story_mean_nll": sc["mean_nll"],
                                  "sentence_median_nll": statistics.median(r["mean_nll"] for r in rows),
                                  "marked": marked,
                                  "top3": [{"sentence": r["sentence"][:120], "mean_nll": r["mean_nll"],
                                            "n_tokens": r["n_tokens"]} for r in ranked[:3]]})
    all_marks = [m for s in report["stories"] for m in s["marked"]]
    report["marks_in_top_quarter"] = f"{sum(m['top_quarter'] for m in all_marks)} of {len(all_marks)}"
    report["expected_by_chance"] = "about 1 in 4"
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    for s in report["stories"]:
        print(f"\n{s['label']}  (story mean {s['story_mean_nll']:.2f}, sentence median {s['sentence_median_nll']:.2f})")
        for m in s["marked"]:
            print(f"  MARK rank {m['rank']:>2}/{m['of']}  nll {m['mean_nll']:.2f}  ({m['n_tokens']} tok)  {m['mark'][:60]}")
        for t in s["top3"]:
            print(f"  top  nll {t['mean_nll']:.2f}  ({t['n_tokens']} tok)  {t['sentence'][:90]}")
    print("\nmarked sentences in their story's top quarter:", report["marks_in_top_quarter"], "(chance ~1 in 4)")


if __name__ == "__main__":
    main()
