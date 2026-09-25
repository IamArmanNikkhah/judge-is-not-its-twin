#!/usr/bin/env python
"""Arm 1 — generation harness.

Each checkpoint writes k completions per prompt. Output JSONL carries full
provenance (model, stage, sampling params, seed) so no run is ever ambiguous.

Usage:
  python generate.py --model allenai/OLMo-2-1124-7B --lineage olmo2 --stage base \
      --prompts ../data/items/prompts.jsonl --out ../data/runs/gen-olmo2-base.jsonl \
      [--k 20] [--max-new-tokens 220] [--prompt-limit 10] [--device auto]

Base checkpoints get the bare completion frame; chat stages get the same frame
(primary, cross-stage comparable) — chat-template runs are a separate sensitivity
pass (--use-chat-template), never mixed in one output file.
"""
import argparse, json, hashlib, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# One shared frame for every stage: plain completion, no instruction-following needed.
COMPLETION_FRAME = "Writing prompt: {prompt}\n\nResponse:"

# Phase B diversified arm (PHASE-B-protocol.md, Option B, ratified 2026-09-05): ONE
# unlikely story per call, so both arms live in the same length regime. The wording
# is elicitation only — typicality and judges score every item under COMPLETION_FRAME
# (the common evaluation frame), never under this text.
UNLIKELY_FRAME = (
    "Writing prompt: {prompt}\n\n"
    "Write a response that most writers would be unlikely to produce: one you would give a low "
    "probability, under 10 percent, among all the responses you could write. It must still be a "
    "complete story that fits the prompt.\n\n"
    "Response:"
)
# "decoding" (2026-09-06, after the Option B wording failed its manipulation check): the SAME
# ordinary frame, typicality moved by sampling temperature instead of by asking. The tag only
# keeps its item IDs apart from the ordinary arm; the dose lives in params and the run id.
ELICITATIONS = {"ordinary": COMPLETION_FRAME, "unlikely": UNLIKELY_FRAME, "decoding": COMPLETION_FRAME}


def build_frames(prompt_text, elicitation):
    """(what the model sees, what every scorer/judge sees). Identical for the ordinary arm."""
    if elicitation not in ELICITATIONS:
        raise ValueError(f"unknown elicitation {elicitation!r}; choose from {sorted(ELICITATIONS)}")
    return (ELICITATIONS[elicitation].format(prompt=prompt_text),
            COMPLETION_FRAME.format(prompt=prompt_text))


def item_id_for(run_id, prompt_id, lineage, stage, elicitation, seed, k):
    """Old IDs are unchanged for the ordinary arm; the diversified arm carries its tag so the
    two arms of one run can never collide or be mistaken for each other."""
    arm = "" if elicitation == "ordinary" else f"__{elicitation}"
    return (f"{run_id}__" if run_id else "") + f"{prompt_id}__{lineage}-{stage}{arm}__s{seed}k{k}"


def generation_stop_metadata(token_ids, eos_token_id, max_new_tokens):
    """Record stopping mechanics, not whether the text is a complete story."""
    eos_ids = [] if eos_token_id is None else (
        [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id))
    ended_eos = bool(token_ids and token_ids[-1] in eos_ids)
    at_cap = len(token_ids) >= max_new_tokens
    return {"generated_token_count": len(token_ids),  # includes any terminal special token
            "last_generated_token_id": token_ids[-1] if token_ids else None,
            "configured_eos_token_ids": eos_ids,
            "ended_with_eos": ended_eos, "token_cap_reached": at_cap,
            "stop_reason": "eos" if ended_eos else "token_cap" if at_cap else "other"}


def pick_device(arg):
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lineage", required=True)
    ap.add_argument("--stage", required=True, choices=["base", "sft", "dpo", "final", "smoke"])
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--k-start", type=int, default=0,
                    help="first k_index to write; a top-up run fills slots k_start..k_start+k-1 under the same run-id")
    ap.add_argument("--max-new-tokens", type=int, default=220)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=411)
    ap.add_argument("--run-id", default=None,
                    help="namespace new experiments so changed caps/elicitation cannot collide with old item IDs")
    ap.add_argument("--prompt-limit", type=int, default=0, help="use only first N prompts (minimal bank)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--use-chat-template", action="store_true")
    ap.add_argument("--elicitation", choices=sorted(ELICITATIONS), default="ordinary",
                    help="Phase B arm: 'unlikely' asks for one low-probability story per call; "
                         "scoring frame stays COMPLETION_FRAME either way")
    ap.add_argument("--resume-from", default=None,
                    help="existing partial JSONL: its records are copied through and their (prompt_id,k_index) slots skipped — free-tier VMs die mid-run, shrink the window")
    args = ap.parse_args()
    if args.use_chat_template and args.elicitation != "ordinary":
        sys.exit("--elicitation unlikely is a completion-frame arm; no chat-template variant exists")

    device = pick_device(args.device)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model)
    if device == "cuda":
        # Stream shards straight to VRAM — Colab VMs have less CPU RAM than the model
        # (12GB RAM vs 14GB fp16 weights → cgroup OOM kill, seen 2026-08-31).
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16, device_map={"": 0}
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.float16 if device != "cpu" else torch.float32
        ).to(device).eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.prompt_limit:
        prompts = prompts[: args.prompt_limit]

    done = {}
    if args.resume_from and Path(args.resume_from).exists():
        for l in open(args.resume_from):
            r = json.loads(l)
            done[(r["prompt_id"], r["params"]["k_index"])] = l
        print(f"resuming: {len(done)} records carried over", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(out_path, "w") as out:
        for l in done.values():
            out.write(l if l.endswith("\n") else l + "\n")
            n_written += 1
        for p in prompts:
            if args.use_chat_template:
                text = tok.apply_chat_template(
                    [{"role": "user", "content": p["text"]}],
                    tokenize=False, add_generation_prompt=True,
                )
                eval_frame = text
            else:
                text, eval_frame = build_frames(p["text"], args.elicitation)
            enc = tok(text, return_tensors="pt").to(device)
            for i in range(args.k_start, args.k_start + args.k):
                if (p["id"], i) in done:
                    continue
                with torch.no_grad():
                    gen = model.generate(
                        **enc,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tok.pad_token_id,
                    )
                generated_ids = gen[0][enc["input_ids"].shape[1]:].tolist()
                completion = tok.decode(generated_ids, skip_special_tokens=True).strip()
                rec = {
                    "item_id": item_id_for(args.run_id, p["id"], args.lineage, args.stage,
                                           args.elicitation, args.seed, i),
                    "prompt_id": p["id"],
                    "text": completion,
                    "source_model": args.model,
                    "lineage": args.lineage,
                    "stage": args.stage,
                    "frame": "chat" if args.use_chat_template else "completion",
                    "elicitation": args.elicitation,
                    "generation_frame": text,        # exactly what the model saw (provenance)
                    "evaluation_frame": eval_frame,  # what every scorer and judge sees (the ruler)
                    "run_id": args.run_id,
                    "termination": generation_stop_metadata(
                        generated_ids, model.generation_config.eos_token_id, args.max_new_tokens),
                    "params": {"temperature": args.temperature, "top_p": args.top_p,
                               "max_new_tokens": args.max_new_tokens, "seed": args.seed, "k_index": i},
                    "sha1": hashlib.sha1(completion.encode()).hexdigest()[:12],
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                out.write(json.dumps(rec) + "\n")
                out.flush()   # the watcher must be able to bank each record before a VM dies
                n_written += 1
            print(f"[{p['id']}] {args.k} completions", file=sys.stderr)
    print(f"wrote {n_written} records -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
