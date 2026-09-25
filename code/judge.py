#!/usr/bin/env python
"""Arm 2 — judge harness: base-executable pairwise forced choice via logits.

Protocol (configs/judging-protocol.md): for each pair (X, Y) the judge sees the
shared completion frame ending in "...the more creative response is Response",
and we read next-token logits for the "A"/"B" continuations. Every pair is run
in BOTH orders; preference = mean of the two order-runs (position bias handled
by construction). No sampling, no output parsing — a base model can do this.

Self-pair exclusion happens at PAIR CONSTRUCTION (build_pairs): a judge never
receives pairs containing generations from its own lineage (any stage) unless
--include-self-pairs is passed (sensitivity run only).

--pairs-file (added 2026-09-03 for the B4b competence gate): score an explicit
list of {prompt_id, item_x, item_y[, ruin]} instead of building pairs. Self-pair
exclusion does not apply — the gate tests competence, not bias, and the judge
seeing its own family's intact story vs. a shuffled copy of it is the point.
Extra pair fields (e.g. "ruin") are carried into the output record.

--use-chat-template (2026-09-03): same content as a user turn under the model's
chat template, with the frame's last line prefilled into the assistant turn.

LABEL READOUT (2026-09-03, after zephyr-sft refused to run): " A"/" B" are one
token under the OLMo and Mistral-v0.1 tokenizers but TWO under the HF-H4
tokenizers (a bare "▁" then "A"). The readout is now tokenizer-exact: tokenize
frame+" A" and frame+" B" in context, find where they diverge, and compare the
probability of each divergent suffix. Single-token suffixes = one forward pass
on the shared prefix (identical to the old readout where it applied); longer
suffixes = one pass per label, summed log-probs. Shared suffix tokens cancel.

Usage:
  python judge.py --model allenai/OLMo-2-1124-7B --judge-lineage olmo2 --judge-stage base \
      --items <items.jsonl> --prompts ../data/items/prompts.jsonl \
      --out ../data/runs/judge-olmo2-base.jsonl [--criterion creative|better] \
      [--pairs-per-prompt 12] [--include-self-pairs] [--seed 411] [--pairs-file <pairs.jsonl>]
      [--use-chat-template]
"""
import argparse, itertools, json, random, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FRAME = (
    "Here are two responses to the same writing prompt.\n"
    "Prompt: {prompt}\n\n"
    "Response A: {a}\n\n"
    "Response B: {b}\n\n"
    "The {phrase} response is Response"
)
CHAT_USER = (
    "Here are two responses to the same writing prompt.\n"
    "Prompt: {prompt}\n\n"
    "Response A: {a}\n\n"
    "Response B: {b}\n\n"
    "Which is the {phrase} response? Answer with A or B."
)
CHAT_PREFILL = "The {phrase} response is Response"
# criterion -> the words the judge reads. "better" is kept EXACTLY as Phase A ran it ("The more
# better response is Response", ungrammatical, found 2026-09-25) so old jobs reproduce; the
# grammatical probe is its own criterion.
PHRASES = {"creative": "more creative", "better": "more better", "better-grammatical": "better"}
MAX_LEN = 1024  # Phase A default; Phase B passes --max-len 2048 (PHASE-B-protocol.md Rule 2)


def fit_max_len(ia, ib, max_len):
    """Keep the tail: the readout lives at the end, so an over-long pair loses its frame HEAD
    (the instruction) — Phase A did this silently. Returns (ia, ib, truncated, longest)."""
    longest = max(len(ia), len(ib))
    if longest <= max_len:
        return ia, ib, False, longest
    cut = longest - max_len
    return ia[cut:], ib[cut:], True, longest


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_pairs(items, judge_lineage, pairs_per_prompt, include_self, rng):
    """Within-prompt pairs across different sources; self-lineage excluded by default."""
    by_prompt = {}
    for it in items:
        by_prompt.setdefault(it["prompt_id"], []).append(it)
    pairs = []
    for pid, group in sorted(by_prompt.items()):
        eligible = [it for it in group
                    if include_self or it.get("lineage") != judge_lineage]
        cross = [(x, y) for x, y in itertools.combinations(eligible, 2)
                 if x.get("source_model") != y.get("source_model")]
        rng.shuffle(cross)
        pairs.extend((pid, x, y, {}) for x, y in cross[:pairs_per_prompt])
    return pairs


def load_pairs_file(path, items):
    by_id = {it["item_id"]: it for it in items}
    pairs = []
    for line in open(path):
        p = json.loads(line)
        x, y = by_id[p["item_x"]], by_id[p["item_y"]]
        meta = {k: v for k, v in p.items() if k not in ("prompt_id", "item_x", "item_y")}
        pairs.append((p["prompt_id"], x, y, meta))
    return pairs


def build_text(tok, prompt_text, xa, xb, criterion, chat):
    if not chat:
        return FRAME.format(prompt=prompt_text, a=xa, b=xb, phrase=PHRASES[criterion])
    msgs = [{"role": "user", "content": CHAT_USER.format(prompt=prompt_text, a=xa, b=xb, phrase=PHRASES[criterion])}]
    head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return head + CHAT_PREFILL.format(phrase=PHRASES[criterion])


def _ids(tok, text, chat):
    return tok(text, add_special_tokens=not chat)["input_ids"]


@torch.no_grad()
def _suffix_logprob(model, ids, start, device):
    """Sum of log p(ids[t] | ids[:t]) for t >= start."""
    x = torch.tensor([ids], device=device)
    lp = torch.log_softmax(model(x).logits[0].float(), dim=-1)
    tgt = x[0, 1:]
    per = lp[:-1].gather(1, tgt[:, None])[:, 0]
    return float(per[start - 1:].sum())


@torch.no_grad()
def choice_prob(model, tok, prompt_text, xa, xb, criterion, device, chat=False, max_len=MAX_LEN):
    """Returns (p_a, mass_ab, top1, path, fit).
    p_a    A-vs-B renormalised probability.
    mass_ab how much of the next-token distribution at the divergence point sits on the two
            label tokens (single-token path) or the joint probability of both suffixes
            (multi-token path) — a readout that is not reading shows tiny mass.
    top1   the model's favourite next token at the divergence point.
    path   "1tok" or "ntok".
    fit    {"n_tokens": longest input, "truncated": bool} — recorded, never silent."""
    text = build_text(tok, prompt_text, xa, xb, criterion, chat)
    ia, ib = _ids(tok, text + " A", chat), _ids(tok, text + " B", chat)
    ia, ib, truncated, longest = fit_max_len(ia, ib, max_len)
    fit = {"n_tokens": longest, "truncated": truncated}
    L = 0
    while L < min(len(ia), len(ib)) and ia[L] == ib[L]:
        L += 1
    sa, sb = ia[L:], ib[L:]
    if len(sa) == 1 and len(sb) == 1:
        x = torch.tensor([ia[:L]], device=device)
        logits = model(x).logits[0, -1, :].float()
        pa = float(torch.softmax(torch.stack([logits[sa[0]], logits[sb[0]]]), dim=0)[0])
        full = torch.softmax(logits, dim=0)
        mass = float(full[sa[0]] + full[sb[0]])
        top1 = tok.decode([int(full.argmax())])
        return pa, mass, top1, "1tok", fit
    la = _suffix_logprob(model, ia, L, device)
    lb = _suffix_logprob(model, ib, L, device)
    pa = float(torch.softmax(torch.tensor([la, lb]), dim=0)[0])
    mass = float(torch.exp(torch.tensor(la)) + torch.exp(torch.tensor(lb)))
    x = torch.tensor([ia[:L]], device=device)
    top1 = tok.decode([int(model(x).logits[0, -1, :].argmax())])
    return pa, mass, top1, "ntok", fit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--judge-lineage", required=True)
    ap.add_argument("--judge-stage", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--criterion", default="creative", choices=sorted(PHRASES))
    ap.add_argument("--pairs-per-prompt", type=int, default=12)
    ap.add_argument("--include-self-pairs", action="store_true")
    ap.add_argument("--pairs-file", default=None, help="explicit pairs; bypasses build_pairs")
    ap.add_argument("--use-chat-template", action="store_true",
                    help="wrap the frame in the model's chat template (trained stages only; base models have none)")
    ap.add_argument("--seed", type=int, default=411)
    ap.add_argument("--max-len", type=int, default=MAX_LEN,
                    help="total input cap. 1024 reproduces Phase A; Phase B passes 2048 (protocol Rule 2)")
    ap.add_argument("--fail-on-truncation", action="store_true",
                    help="abort on the first pair that does not fit --max-len; Phase B runs pass this")
    args = ap.parse_args()

    device = pick_device()
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if device == "cuda":
        # direct-to-VRAM: Colab CPU RAM < fp16 weight size (OOM kill, 2026-08-31)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map={"": 0}
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16 if device != "cpu" else torch.float32
        ).to(device).eval()

    prompts = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(args.prompts)}
    items = [json.loads(l) for l in open(args.items)]
    if args.pairs_file:
        pairs = load_pairs_file(args.pairs_file, items)
    else:
        pairs = build_pairs(items, args.judge_lineage, args.pairs_per_prompt,
                            args.include_self_pairs, rng)
    if not pairs:
        sys.exit("no eligible pairs — check self-pair exclusion vs item sources")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crit = args.criterion
    paths = set()
    with open(out_path, "w") as out:
        for pid, x, y, meta in pairs:
            ptext = prompts[pid]
            # both orders; criterion prob is for the FIRST item of the pair (x)
            p_x_first, m1, t1, path1, fit1 = choice_prob(model, tok, ptext, x["text"], y["text"], crit, device,
                                                        chat=args.use_chat_template, max_len=args.max_len)
            p_y_first, m2, t2, path2, fit2 = choice_prob(model, tok, ptext, y["text"], x["text"], crit, device,
                                                        chat=args.use_chat_template, max_len=args.max_len)
            if args.fail_on_truncation and (fit1["truncated"] or fit2["truncated"]):
                sys.exit(f"pair {x['item_id']} / {y['item_id']} needs {max(fit1['n_tokens'], fit2['n_tokens'])} "
                         f"tokens > --max-len {args.max_len}; refusing to judge with the frame head cut off")
            paths.update((path1, path2))
            p_x = (p_x_first + (1.0 - p_y_first)) / 2.0
            out.write(json.dumps({
                "prompt_id": pid,
                "item_x": x["item_id"], "item_y": y["item_id"],
                "judge_model": args.model,
                "judge_lineage": args.judge_lineage,
                "judge_stage": args.judge_stage,
                "criterion": args.criterion,
                "p_x": round(p_x, 5),
                "p_x_first_order": round(p_x_first, 5),
                "p_y_first_order": round(p_y_first, 5),
                "self_pairs_included": args.include_self_pairs or bool(args.pairs_file),
                "frame": "chat" if args.use_chat_template else "completion",
                "readout": path1 if path1 == path2 else f"{path1}/{path2}",
                "mass_ab_first_order": round(m1, 5), "mass_ab_second_order": round(m2, 5),
                "top1_first_order": t1, "top1_second_order": t2,
                "max_len": args.max_len,
                "n_tokens_first_order": fit1["n_tokens"], "n_tokens_second_order": fit2["n_tokens"],
                "truncated": fit1["truncated"] or fit2["truncated"],
                **meta,
            }) + "\n")
            out.flush()
    print(f"judged {len(pairs)} pairs [readout {sorted(paths)}] -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
