#!/usr/bin/env python
"""Pilot robustness pass — stresses the four claims in RESULTS-pilot-2026-09-02.md
with the data already on disk. No GPU. numpy only (the repo venv has no scipy).

Why this exists (2026-09-03): pilot_analysis.py reproduces the RESULTS tables and
is frozen as their provenance. The threats below are not in it and each one can
flip a claim, so they get their own artifact rather than a heredoc:

  R1 LENGTH   items are scored standalone (no prompt prefix, typicality.py), so
              the first tokens are contextless and a longer item gets a lower
              mean NLL for free. Aligned Zephyr is longer than its base. Is the
              -22% collapse a length artifact?
  R2 LOOPS    temperature-1.0 aligned models can fall into repetition; a loop is
              trivially predictable. Is the low NLL blandness or degeneracy?
  R3 PROMPTS  the design is 10 prompts x k=20. Items within a prompt are not
              independent. The only honest unit is the prompt: exact sign test
              + cluster bootstrap over prompts.
  R4 DECOMP   RESULTS §4 found a shared aligned-style component in the own-diet
              Delta-ppl. Item-level difference-in-differences: own pair minus
              the STRANGER pair on the SAME items. Stage differences cancel any
              constant offset between the pairs, so the gradient that survives
              is lineage-specific by construction.
  R5 SHAPE    is the Zephyr collapse a whole-distribution shift or a subset of
              items collapsing? Quantiles per stage.

Usage:
  python pilot_robustness.py [--root ..] > ../data/analysis/robustness-2026-09-03.txt
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path

import numpy as np

STAGES = ("base", "sft", "dpo", "final")
LINEAGE_STAGES = {"olmo2": ("base", "sft", "dpo", "final"), "zephyr": ("base", "sft", "dpo")}
# held-out reader for the shared-diet measure (leave-own-lineage-out)
HELDOUT = {"olmo2": "mistral-base", "zephyr": "olmo2-base"}
# own-diet pair (aligned, base) per lineage
OWNPAIR = {"olmo2": ("olmo2-instruct", "olmo2-base"), "zephyr": ("zephyr-final", "mistral-base")}
OTHER = {"olmo2": "zephyr", "zephyr": "olmo2"}
RNG = np.random.default_rng(411)
NBOOT = 10000


def jl(path):
    return [json.loads(l) for l in open(path)]


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def sign_test_p(n_pos, n):
    """Exact two-sided binomial sign test, p=0.5."""
    k = min(n_pos, n - n_pos)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def cluster_boot(per_prompt_values, nboot=NBOOT):
    """Bootstrap the mean over prompts (the cluster). Returns (mean, lo95, hi95)."""
    v = np.asarray(per_prompt_values, float)
    n = len(v)
    idx = RNG.integers(0, n, size=(nboot, n))
    means = v[idx].mean(axis=1)
    return v.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def rep4(text):
    """Fraction of repeated word 4-grams: 1 - distinct/total. 0 = no repeats."""
    w = text.split()
    if len(w) < 8:
        return 0.0
    grams = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1.0 - len(set(grams)) / len(grams)


def ols(X, y):
    """Plain least squares; returns coefficients."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--typ-suffix", default="",
                    help='"" = pilot standalone files; "-prefix" = the 2026-09-03 completion-prefix re-score')
    args = ap.parse_args()
    SUF = args.typ_suffix
    R = Path(args.root)

    bank = {r["item_id"]: r for r in jl(R / "data/items/bank-pilot.jsonl")}
    typ, nchars = {}, {}
    for tag in ("mistral-base", "olmo2-base", "olmo2-instruct", "zephyr-final"):
        rows = jl(R / f"data/typicality/typ-{tag}{SUF}.jsonl")
        typ[tag] = {r["item_id"]: r["mean_nll"] for r in rows}
        for r in rows:
            nchars[r["item_id"]] = r["n_chars"]
    items = sorted(bank)
    assert all(i in typ[t] for t in typ for i in items), "typicality coverage gap"

    def lin(i): return bank[i]["lineage"]
    def stg(i): return bank[i]["stage"]
    def pid(i): return bank[i]["prompt_id"]
    prompts = sorted({pid(i) for i in items})
    by_stage = defaultdict(list)
    for i in items:
        by_stage[(lin(i), stg(i))].append(i)

    # ------------------------------------------------------------------ R1
    hr("R1. LENGTH CONFOUND — standalone scoring makes longer items cheaper per token")
    print("   n_chars per stage (median) and within-stage Pearson r(mean_nll, log n_chars)")
    print("   under the held-out reader. A strong negative r inside every stage means")
    print("   length buys NLL; then the question is whether stage survives length control.")
    for L, stages in LINEAGE_STAGES.items():
        ref = HELDOUT[L]
        print(f"   {L}  read by {ref}")
        for s in stages:
            ids = by_stage[(L, s)]
            x = np.log([nchars[i] for i in ids]); y = np.array([typ[ref][i] for i in ids])
            r = np.corrcoef(x, y)[0, 1]
            print("     %-6s n=%3d  med chars %4d   r(nll, log len) = %+.3f" %
                  (s, len(ids), int(np.median([nchars[i] for i in ids])), r))
    print()
    print("   OLS within lineage: nll ~ stage dummies + log(n_chars). Stage effects are")
    print("   vs base. 'raw' = without the length term (matches RESULTS §2 deltas).")
    for L, stages in LINEAGE_STAGES.items():
        ref = HELDOUT[L]
        ids = [i for s in stages for i in by_stage[(L, s)]]
        y = np.array([typ[ref][i] for i in ids])
        logl = np.log([nchars[i] for i in ids])
        D = np.array([[1.0] + [1.0 if stg(i) == s else 0.0 for s in stages[1:]] for i in ids])
        b_raw = ols(D, y)
        b_len = ols(np.column_stack([D, logl - logl.mean()]), y)
        print(f"   {L} (read by {ref})")
        for k, s in enumerate(stages[1:], start=1):
            print("     %-6s raw %+.3f   length-adjusted %+.3f" % (s, b_raw[k], b_len[k]))
        print("     log-length coefficient %+.3f per e-fold of length" % b_len[-1])
    print()
    print("   Length-matched cut: items with 700 <= n_chars <= 1000 only.")
    for L, stages in LINEAGE_STAGES.items():
        ref = HELDOUT[L]
        cells = []
        for s in stages:
            v = [typ[ref][i] for i in by_stage[(L, s)] if 700 <= nchars[i] <= 1000]
            cells.append((s, len(v), np.mean(v)))
        base = cells[0][2]
        print(f"   {L}: " + "  ".join("%s %.3f (n=%d, %+.1f%%)" % (s, m, n, 100 * (m - base) / base)
                                       for s, n, m in cells))

    print()
    print("   R1b. MIN-LENGTH SWEEP — is the within-stage length correlation the short tail?")
    print("   (standalone scoring: a 6-token item is all contextless tokens; the 1/n cost of the")
    print("   cold start dominates only when items are short. Zephyr has no short items.)")
    for L, stages in LINEAGE_STAGES.items():
        ref = HELDOUT[L]
        print(f"   {L}  read by {ref}")
        for mn in (0, 200, 400, 600):
            cells, rs = [], []
            for s in stages:
                ids = [i for i in by_stage[(L, s)] if nchars[i] >= mn]
                v = np.array([typ[ref][i] for i in ids])
                x = np.log([nchars[i] for i in ids])
                cells.append((s, len(ids), v.mean()))
                rs.append(np.corrcoef(x, v)[0, 1])
            base = cells[0][2]
            print("     chars>=%4d  " % mn + "  ".join("%s %.3f/n%d(%+.1f%%)" % (s, m, n, 100 * (m - base) / base)
                                                  for s, n, m in cells))
            print("                 within-stage r(nll, log len): " + "  ".join("%+.2f" % r for r in rs))
        # prompt-level inference on the >=400 cut, base -> each aligned stage
        keep = {i for i in items if lin(i) == L and nchars[i] >= 400}
        def pp(s):
            return np.array([np.mean([typ[ref][i] for i in by_stage[(L, s)] if i in keep and pid(i) == p])
                             for p in prompts])
        b = pp("base")
        for s in stages[1:]:
            d = pp(s) - b
            neg = int((d < 0).sum())
            m, lo, hi = cluster_boot(d)
            print("     chars>=400 prompt-level base->%-5s drops %2d/10 sign p=%.4f mean %+.3f CI [%+.3f, %+.3f]" %
                  (s, neg, sign_test_p(neg, 10), m, lo, hi))
    print()
    print("   Own-diet Delta-ppl vs length (does the own-diet gradient ride on length too?)")
    for L, stages in LINEAGE_STAGES.items():
        a, bref = OWNPAIR[L]
        dppl = {i: typ[a][i] - typ[bref][i] for i in items if lin(i) == L}
        rs = []
        for s in stages:
            ids = by_stage[(L, s)]
            rs.append(np.corrcoef(np.log([nchars[i] for i in ids]), [dppl[i] for i in ids])[0, 1])
        long = {s: np.mean([dppl[i] for i in by_stage[(L, s)] if nchars[i] >= 400]) for s in stages}
        print("   %-6s within-stage r(dppl, log len): %s   | dppl on chars>=400: %s" %
              (L, "  ".join("%+.2f" % r for r in rs), "  ".join("%s %+.3f" % (s, long[s]) for s in stages)))
    print()
    print("   What the short OLMo items ARE (shortest 5 per aligned stage, first 90 chars):")
    for s in ("sft", "dpo", "final"):
        ids = sorted(by_stage[("olmo2", s)], key=lambda i: nchars[i])[:5]
        for i in ids:
            t = bank[i]["text"].strip().replace("\n", "\\n")
            print("     %-5s %4dch nll %.2f  %r" % (s, nchars[i], typ["mistral-base"][i], t[:90]))

    # ------------------------------------------------------------------ R2
    hr("R2. DEGENERACY — repeated word-4gram rate per stage (loops read as low NLL)")
    rep = {i: rep4(bank[i]["text"]) for i in items}
    for L, stages in LINEAGE_STAGES.items():
        ref = HELDOUT[L]
        print(f"   {L}  read by {ref}")
        for s in stages:
            ids = by_stage[(L, s)]
            rr = np.array([rep[i] for i in ids])
            loops = [i for i in ids if rep[i] > 0.20]
            clean = [i for i in ids if rep[i] <= 0.20]
            print("     %-6s mean rep4 %.3f  items rep4>0.2: %3d/%3d   nll all %.3f | loops %s | clean %.3f" %
                  (s, rr.mean(), len(loops), len(ids), np.mean([typ[ref][i] for i in ids]),
                   ("%.3f" % np.mean([typ[ref][i] for i in loops])) if loops else "  -  ",
                   np.mean([typ[ref][i] for i in clean])))
    print()
    print("   Collapse recomputed on loop-free items only (rep4 <= 0.2), held-out reader:")
    for L, stages in LINEAGE_STAGES.items():
        ref = HELDOUT[L]
        m = {s: np.mean([typ[ref][i] for i in by_stage[(L, s)] if rep[i] <= 0.20]) for s in stages}
        print(f"   {L}: " + "  ".join("%s %.3f (%+.1f%%)" % (s, m[s], 100 * (m[s] - m["base"]) / m["base"])
                                       for s in stages))

    # ------------------------------------------------------------------ R3
    hr("R3. PROMPT-LEVEL INFERENCE — the cluster is the prompt (n=10); items are not independent")
    print("   Per-prompt mean NLL by stage; delta = aligned - base per prompt. Exact sign test")
    print("   (two-sided) on the direction, cluster bootstrap (%d reps) for the mean delta." % NBOOT)

    def per_prompt(L, s, score):
        return np.array([np.mean([score[i] for i in by_stage[(L, s)] if pid(i) == p]) for p in prompts])

    print("   -- shared-diet (held-out reader) --")
    for L, stages in LINEAGE_STAGES.items():
        ref = HELDOUT[L]
        b = per_prompt(L, "base", typ[ref])
        for s in stages[1:]:
            d = per_prompt(L, s, typ[ref]) - b
            neg = int((d < 0).sum())
            m, lo, hi = cluster_boot(d)
            print("   %-6s base->%-5s  drops %2d/10  sign p=%.4f   mean delta %+.3f  95%% CI [%+.3f, %+.3f]" %
                  (L, s, neg, sign_test_p(neg, 10), m, lo, hi))
    print("   -- own-diet Delta-ppl: aligned reader minus base reader, grouped by producing stage --")
    for L, stages in LINEAGE_STAGES.items():
        a, bref = OWNPAIR[L]
        dppl = {i: typ[a][i] - typ[bref][i] for i in items if lin(i) == L}
        b = per_prompt(L, "base", dppl)
        for s in stages[1:]:
            d = per_prompt(L, s, dppl) - b
            neg = int((d < 0).sum())
            m, lo, hi = cluster_boot(d)
            print("   %-6s base->%-5s  drops %2d/10  sign p=%.4f   mean delta %+.3f  95%% CI [%+.3f, %+.3f]" %
                  (L, s, neg, sign_test_p(neg, 10), m, lo, hi))
    print("   -- monotonicity: is the stage order base > sft > dpu strictly decreasing inside each prompt? --")
    for L, stages in LINEAGE_STAGES.items():
        a, bref = OWNPAIR[L]
        dppl = {i: typ[a][i] - typ[bref][i] for i in items if lin(i) == L}
        seq = np.column_stack([per_prompt(L, s, dppl) for s in stages[:3]])  # base, sft, dpo
        mono = int(np.all(np.diff(seq, axis=1) < 0, axis=1).sum())
        print("   %-6s own-diet base>sft>dpo in %d/10 prompts" % (L, mono))

    # ------------------------------------------------------------------ R4
    hr("R4. DELTA-PPL DECOMPOSITION — own pair minus STRANGER pair on the same items")
    print("   own(i)      = NLL(own aligned) - NLL(own base)")
    print("   stranger(i) = NLL(other lineage aligned) - NLL(other lineage base)   [same item]")
    print("   specific(i) = own(i) - stranger(i). A constant calibration offset between the")
    print("   pairs moves every stage equally, so STAGE DIFFERENCES in 'specific' are what")
    print("   the shared aligned-style component cannot explain.")
    for L, stages in LINEAGE_STAGES.items():
        a, bref = OWNPAIR[L]
        a2, b2 = OWNPAIR[OTHER[L]]
        own = {i: typ[a][i] - typ[bref][i] for i in items if lin(i) == L}
        strg = {i: typ[a2][i] - typ[b2][i] for i in items if lin(i) == L}
        spec = {i: own[i] - strg[i] for i in own}
        print(f"   {L} items")
        print("     %-6s %9s %9s %9s" % ("stage", "own", "stranger", "specific"))
        for s in stages:
            ids = by_stage[(L, s)]
            print("     %-6s %+9.3f %+9.3f %+9.3f" % (s, np.mean([own[i] for i in ids]),
                                                    np.mean([strg[i] for i in ids]),
                                                    np.mean([spec[i] for i in ids])))
        so = [np.mean([own[i] for i in by_stage[(L, s)]]) for s in stages]
        ss = [np.mean([strg[i] for i in by_stage[(L, s)]]) for s in stages]
        sp = [np.mean([spec[i] for i in by_stage[(L, s)]]) for s in stages]
        print("     spread across stages: own %.3f  stranger %.3f  specific %.3f" %
              (max(so) - min(so), max(ss) - min(ss), max(sp) - min(sp)))
        # per-prompt: base is the most-foreign stage on the SPECIFIC measure?
        pp = np.column_stack([per_prompt(L, s, spec) for s in stages])
        most_foreign_base = int((pp.argmax(axis=1) == 0).sum())
        last = stages[-1] if L == "zephyr" else "dpo"
        d = per_prompt(L, last, spec) - per_prompt(L, "base", spec)
        neg = int((d < 0).sum())
        m, lo, hi = cluster_boot(d)
        print("     per-prompt: base most-foreign on 'specific' in %d/10; base->%s drops %d/10, sign p=%.4f, "
              "mean %+.3f CI [%+.3f, %+.3f]" % (most_foreign_base, last, neg, sign_test_p(neg, 10), m, lo, hi))
        # item-level: how much of own is predicted by stranger?
        x = np.array([strg[i] for i in own]); y = np.array([own[i] for i in own])
        r = np.corrcoef(x, y)[0, 1]
        print("     item-level r(own, stranger) = %+.3f  (shared component strength across items)" % r)

    # ------------------------------------------------------------------ R5
    hr("R5. DISTRIBUTION SHAPE — quantiles of held-out NLL per stage")
    for L, stages in LINEAGE_STAGES.items():
        ref = HELDOUT[L]
        print(f"   {L}  read by {ref}          p10    p25    p50    p75    p90   frac<base p50")
        bmed = np.median([typ[ref][i] for i in by_stage[(L, "base")]])
        for s in stages:
            v = np.array([typ[ref][i] for i in by_stage[(L, s)]])
            q = np.percentile(v, [10, 25, 50, 75, 90])
            print("     %-6s                   " % s + "  ".join("%.3f" % x for x in q) +
                  "     %.2f" % (v < bmed).mean())
    print()
    print("   Reading: a whole-distribution shift moves every quantile by about the same")
    print("   amount; a subset collapsing drags p10/p25 far more than p75/p90.")


if __name__ == "__main__":
    main()
