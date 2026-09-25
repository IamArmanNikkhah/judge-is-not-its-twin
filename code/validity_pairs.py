#!/usr/bin/env python
"""B4b — build the judge competence gate: pairs whose answer is known.

Design Amendment A2 (2026-09-03, author-ratified: "80 percent and the three ruins").
For each prompt, N_INTACT real stories from the bank (>= MIN_CHARS, drawn round-robin
across the seven stage groups so no source dominates) are each paired with three ruins:

  shuffled   the same words in random order         — the harsh test; GATES
  stub       the first sentence only                — "creative" must mean more than "long";
                                                      doubles as a read on the OLMo stubs
  offprompt  an intact story answering a DIFFERENT   — is the judge reading the prompt at all
             prompt (same length band)

Ruined items get new ids (val-<ruin>__<orig>) and live outside the bank, so nothing here
leaks into the bias pairs. Deterministic under --seed. Writes:
  data/validity/validity-items.jsonl   intact copies + ruined items (judge.py --items)
  data/validity/validity-pairs.jsonl   {prompt_id, item_x=intact, item_y=ruined, ruin}
                                       (judge.py --pairs-file)

Usage: python validity_pairs.py [--root ..] [--seed 411] [--n-intact 6]
"""
import argparse, json, random, re
from collections import defaultdict
from pathlib import Path

MIN_CHARS = 400          # a stub-vs-stub pair tests nothing; intact means intact
RUINS = ("shuffled", "stub", "offprompt")
SENT = re.compile(r"(?<=[.!?])\s+")


def first_sentence(text):
    parts = SENT.split(text.strip(), maxsplit=2)
    s = parts[0]
    if len(s) < 20 and len(parts) > 1:      # "Silence." alone is not a sentence-sized stub
        s = parts[0] + " " + parts[1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--seed", type=int, default=411)
    ap.add_argument("--n-intact", type=int, default=6)
    args = ap.parse_args()
    R = Path(args.root)
    rng = random.Random(args.seed)

    bank = [json.loads(l) for l in open(R / "data/items/bank-pilot.jsonl")]
    long_items = [it for it in bank if len(it["text"]) >= MIN_CHARS]
    by_prompt = defaultdict(lambda: defaultdict(list))
    for it in long_items:
        by_prompt[it["prompt_id"]][(it["lineage"], it["stage"])].append(it)
    prompts = sorted(by_prompt)

    # intact picks: round-robin over stage groups so every source is represented
    intact = {}
    for pi, pid in enumerate(prompts):
        groups = sorted(by_prompt[pid])
        for g in groups:
            rng.shuffle(by_prompt[pid][g])
        # start the round-robin at a different group per prompt: 6 picks over 7 groups
        # from a fixed start never reach the seventh source (caught 2026-09-03: zephyr-sft
        # had zero intact items)
        picks, k = [], pi
        while len(picks) < args.n_intact:
            g = groups[k % len(groups)]
            if by_prompt[pid][g]:
                picks.append(by_prompt[pid][g].pop())
            k += 1
        intact[pid] = picks

    items_out, pairs_out = [], []
    for pid in prompts:
        others = [p for p in prompts if p != pid]
        for it in intact[pid]:
            items_out.append({**it, "role": "intact"})
            base_id = it["item_id"]
            # shuffled
            words = it["text"].split()
            rng.shuffle(words)
            ruined = {"shuffled": " ".join(words), "stub": first_sentence(it["text"])}
            # offprompt: a long story from another prompt, within +-30% of this one's length
            L = len(it["text"])
            pool = [o for o in long_items if o["prompt_id"] in others and 0.7 * L <= len(o["text"]) <= 1.3 * L]
            ruined["offprompt"] = rng.choice(pool)["text"] if pool else rng.choice(
                [o for o in long_items if o["prompt_id"] in others])["text"]
            for ruin in RUINS:
                rid = f"val-{ruin}__{base_id}"
                items_out.append({"item_id": rid, "prompt_id": pid, "text": ruined[ruin],
                                  "source_model": f"ruin:{ruin}", "lineage": "validity",
                                  "stage": ruin, "role": "ruined", "ruin": ruin, "source_item": base_id})
                pairs_out.append({"prompt_id": pid, "item_x": base_id, "item_y": rid, "ruin": ruin})

    out = R / "data/validity"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "validity-items.jsonl", "w") as f:
        for r in items_out: f.write(json.dumps(r) + "\n")
    with open(out / "validity-pairs.jsonl", "w") as f:
        for r in pairs_out: f.write(json.dumps(r) + "\n")
    n_int = sum(1 for r in items_out if r["role"] == "intact")
    srcs = defaultdict(int)
    for pid in prompts:
        for it in intact[pid]: srcs[(it["lineage"], it["stage"])] += 1
    print(f"prompts {len(prompts)}  intact {n_int}  ruined {len(items_out) - n_int}  pairs {len(pairs_out)}")
    print("intact by source:", dict(sorted(srcs.items())))
    stubs = [len(r["text"]) for r in items_out if r.get("ruin") == "stub"]
    print(f"stub length min/med/max {min(stubs)}/{sorted(stubs)[len(stubs)//2]}/{max(stubs)} chars")


if __name__ == "__main__":
    main()
