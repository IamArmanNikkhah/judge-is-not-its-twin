#!/usr/bin/env python
"""Pilot analysis — reproduces every table in RESULTS-pilot-2026-09-02.md.

Stdlib only, no GPU, runs in seconds. Every number in the results document must
come from THIS script's output (data/analysis/pilot-<date>.txt), never from a
chat transcript or a one-off snippet. The pilot's first numbers were computed in
throwaway heredocs on 2026-09-01/02; this file is their durable home.

Usage:
  python pilot_analysis.py [--root ..] > ../data/analysis/pilot-2026-09-02.txt

Sections (in order, matching the results doc):
  1 generation summary            — what was written, per stage
  2 bank composition + drop table — what was kept and why
  3 shared-diet typicality        — rarity under HELD-OUT base references
  4 positive control              — base-stage-only kinship check (the whole-family
                                    version is confounded; see RESULTS §3)
  5 shared-diet collapse          — base -> aligned, per lineage, per reader,
                                    per-prompt robustness
  6 own-diet Delta-ppl            — NLL(own aligned) - NLL(own base), by producing
                                    stage; per-prompt; calibration-offset check;
                                    cross-lineage specificity
  7 judge pipeline validation     — mechanical checks, degeneracy, position bias,
                                    and the slope that is computed but NOT read
"""
import argparse, json, statistics as st
from collections import defaultdict
from pathlib import Path

STAGES = ("base", "sft", "dpo", "final")
ALIGNED = ("sft", "dpo", "final")


def jl(path):
    return [json.loads(l) for l in open(path)]


def nll_map(path):
    return {r["item_id"]: r["mean_nll"] for r in jl(path)}


def mean(v):
    return st.mean(v) if v else float("nan")


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--typ-suffix", default="",
                    help='"" = pilot standalone files; "-prefix" = the 2026-09-03 completion-prefix re-score')
    args = ap.parse_args()
    SUF = args.typ_suffix
    R = Path(args.root)

    # ---------- inputs ----------
    gen_files = {
        ("olmo2", s): R / f"data/runs/gen-olmo2-{s}-pilot.jsonl" for s in STAGES
    }
    gen_files.update({("zephyr", s): R / f"data/runs/gen-zephyr-{s}-pilot.jsonl"
                      for s in ("base", "sft", "dpo")})
    bank = {r["item_id"]: r for r in jl(R / "data/items/bank-pilot.jsonl")}
    typ = {
        "mistral-base":   nll_map(R / f"data/typicality/typ-mistral-base{SUF}.jsonl"),
        "olmo2-base":     nll_map(R / f"data/typicality/typ-olmo2-base{SUF}.jsonl"),
        "olmo2-instruct": nll_map(R / f"data/typicality/typ-olmo2-instruct{SUF}.jsonl"),
        "zephyr-final":   nll_map(R / f"data/typicality/typ-zephyr-final{SUF}.jsonl"),
    }
    judge = jl(R / "data/judge/judge-olmo2-base.jsonl")

    def lin(i): return bank[i]["lineage"]
    def stg(i): return bank[i]["stage"]
    def pid(i): return bank[i]["prompt_id"]
    prompts = sorted({pid(i) for i in bank})

    # ---------- 1 generation ----------
    hr("1. GENERATION SUMMARY (data/runs/gen-*-pilot.jsonl)")
    print("%-14s %5s %6s %8s %20s" % ("stage", "n", "empty", "<40ch", "len min/med/max"))
    for (l, s), p in gen_files.items():
        rs = jl(p)
        L = sorted(len(r["text"].strip()) for r in rs)
        slots = len({(r["prompt_id"], r["params"]["k_index"]) for r in rs})
        print("%-14s %5d %6d %8d %8d/%4d/%4d   unique slots %d"
              % (f"{l}-{s}", len(rs), sum(1 for x in L if x == 0),
                 sum(1 for x in L if x < 40), L[0], L[len(L)//2], L[-1], slots))

    # ---------- 2 bank ----------
    hr("2. BANK COMPOSITION (data/items/bank-pilot.jsonl) — empty-only cut")
    by = defaultdict(int)
    for i in bank: by[(lin(i), stg(i))] += 1
    for k in sorted(by): print("   %-14s %4d" % ("-".join(k), by[k]))
    print("   TOTAL %d   lineages %s   prompts %d" %
          (len(bank), sorted({lin(i) for i in bank}), len(prompts)))
    allL = sorted(len(r["text"].strip()) for p in gen_files.values() for r in jl(p))
    print("   drop table over all %d generations (what each candidate cut removes):" % len(allL))
    for t in (1, 20, 40, 60, 100, 150, 200):
        print("     chars < %4d: %3d" % (t, sum(1 for x in allL if x < t)))

    # ---------- 3 shared-diet by stage ----------
    hr("3. SHARED-DIET TYPICALITY — mean per-token NLL by producing stage, per reader")
    def stage_table(ref):
        g = defaultdict(list)
        for i, v in typ[ref].items(): g[(lin(i), stg(i))].append(v)
        return g
    G = {ref: stage_table(ref) for ref in ("mistral-base", "olmo2-base")}
    print("%-14s %14s %14s" % ("producer", "read by mistral", "read by olmo2"))
    for l in ("olmo2", "zephyr"):
        for s in STAGES:
            if (l, s) in G["mistral-base"]:
                print("%-14s %14.3f %14.3f" % (f"{l}-{s}",
                      mean(G["mistral-base"][(l, s)]), mean(G["olmo2-base"][(l, s)])))

    # ---------- 4 positive control ----------
    hr("4. POSITIVE CONTROL — BASE-stage items only (each reader should find its own family's")
    print("   raw samples more predictable). Whole-family averaging is CONFOUNDED: aligned")
    print("   models emit low-entropy text ANY reader finds predictable, so it measures")
    print("   alignment, not kinship. This cut is the only apples-to-apples one.")
    for ref, own in (("mistral-base", "zephyr"), ("olmo2-base", "olmo2")):
        other = "olmo2" if own == "zephyr" else "zephyr"
        o = mean(G[ref][(own, "base")]); s_ = mean(G[ref][(other, "base")])
        print("   %-13s own %.3f  stranger %.3f  -> own lower? %s" % (ref, o, s_, o < s_))

    # ---------- 5 shared-diet collapse ----------
    hr("5. SHARED-DIET COLLAPSE — base -> aligned, with per-prompt robustness")
    for l in ("zephyr", "olmo2"):
        for ref in ("mistral-base", "olmo2-base"):
            g = G[ref]; b = mean(g[(l, "base")])
            parts = []
            for s in ALIGNED:
                if (l, s) in g:
                    m = mean(g[(l, s)]); parts.append("%s %.3f (%+.1f%%)" % (s, m, 100*(m-b)/b))
            # per prompt: aligned-mean minus base-mean within each prompt
            drops = 0; deltas = []
            for p in prompts:
                bb = [v for i, v in typ[ref].items() if lin(i) == l and stg(i) == "base" and pid(i) == p]
                aa = [v for i, v in typ[ref].items() if lin(i) == l and stg(i) in ALIGNED and pid(i) == p]
                d = mean(aa) - mean(bb); deltas.append(d); drops += d < 0
            print("   %-7s read by %-13s base %.3f -> %s" % (l, ref, b, " | ".join(parts)))
            print("           per-prompt drops %2d/10   deltas %s" % (drops, " ".join("%+.2f" % d for d in deltas)))

    # ---------- 6 own-diet Delta-ppl ----------
    hr("6. OWN-DIET DELTA-PPL — NLL(own aligned) - NLL(own base), by producing stage")
    print("   negative = the aligned model finds that text MORE familiar than its base did")
    pairs = {"olmo2": ("olmo2-instruct", "olmo2-base"), "zephyr": ("zephyr-final", "mistral-base")}
    for l, (al, ba) in pairs.items():
        D = {i: typ[al][i] - typ[ba][i] for i in bank}
        own = defaultdict(list); out = defaultdict(list)
        for i, d in D.items():
            (own if lin(i) == l else out)[stg(i)].append(d)
        print("\n   %s  [%s minus %s]" % (l.upper(), al, ba))
        own_means = {s: mean(own[s]) for s in STAGES if own[s]}
        print("     own-lineage : " + "  ".join("%s %+.3f" % (s, m) for s, m in own_means.items()))
        out_means = {s: mean(out[s]) for s in STAGES if out[s]}
        print("     out-lineage : " + "  ".join("%s %+.3f" % (s, m) for s, m in out_means.items()))
        rng_own = max(own_means.values()) - min(own_means.values())
        rng_out = max(out_means.values()) - min(out_means.values())
        print("     spread across stages, own %.3f  (a pure calibration offset would give ~0)" % rng_own)
        print("     specificity: own range %.3f vs out-of-lineage range %.3f  -> %.1fx" % (rng_own, rng_out, rng_own / rng_out))
        steps = list(own_means.items())
        big = max(((steps[k+1][0], steps[k+1][1] - steps[k][1]) for k in range(len(steps)-1)), key=lambda t: abs(t[1]))
        print("     biggest single step: -> %s  %+.3f" % big)
        # per-prompt: base must be the most-foreign stage
        ok = 0
        for p in prompts:
            row = {s: mean([D[i] for i in D if lin(i) == l and stg(i) == s and pid(i) == p]) for s in own_means}
            ok += row["base"] == max(row.values())
        print("     per-prompt: base is the most-foreign stage in %d/10 prompts" % ok)

    # ---------- 7 judge validation ----------
    hr("7. JUDGE PIPELINE VALIDATION — olmo2-base, data/judge/judge-olmo2-base.jsonl")
    print("   PIPELINE PROOF ONLY. The B4b competence gate does not exist; nothing below is evidence.")
    lins = {lin(r["item_x"]) for r in judge} | {lin(r["item_y"]) for r in judge}
    same = sum(1 for r in judge if bank[r["item_x"]]["source_model"] == bank[r["item_y"]]["source_model"])
    print("   pairs %d | lineages judged %s | self-lineage exclusion %s | same-source pairs %d | prompts %d"
          % (len(judge), sorted(lins), "HELD" if lins == {"zephyr"} else "BROKEN", same,
             len({r["prompt_id"] for r in judge})))
    px = [r["p_x"] for r in judge]
    print("   degeneracy: p_x min/med/max %.3f/%.3f/%.3f  sd %.4f  frac within .02 of .50 = %.2f"
          % (min(px), st.median(px), max(px), st.pstdev(px), sum(abs(p-.5) < .02 for p in px)/len(px)))
    spread = [abs(r["p_x_first_order"] - (1 - r["p_y_first_order"])) for r in judge]
    print("   position bias: mean |order disagreement| %.3f ; raw p(first slot) %.3f"
          % (mean(spread), mean([r["p_x_first_order"] for r in judge])))
    O = typ["olmo2-base"]
    xs = [O[r["item_x"]] - O[r["item_y"]] for r in judge]; ys = px
    mx, my = mean(xs), mean(ys)
    cov = sum((a-mx)*(b-my) for a, b in zip(xs, ys)); var = sum((a-mx)**2 for a in xs)
    r_ = cov / ((var * sum((b-my)**2 for b in ys)) ** .5)
    print("   typicality slope (p_x on rarity_x - rarity_y): beta %+.4f  r %+.3f  n %d   <- NOT INTERPRETED"
          % (cov/var, r_, len(judge)))


if __name__ == "__main__":
    main()
