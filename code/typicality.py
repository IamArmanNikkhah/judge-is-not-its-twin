#!/usr/bin/env python
"""Typicality scorer (T1-reparametrized mechanism measures).

Two measures per item:
  shared_nll  — mean per-token NLL under one or more REFERENCE base models
                (leave-own-lineage-out rule lives in the run config, not here:
                the caller passes the reference models appropriate for the
                items being scored).
  Appending runs: each invocation scores items under ONE reference model and
  writes one JSONL; the merge step averages shared_nll across references and
  computes delta_ppl = nll(aligned) - nll(base) when both stages of a lineage
  have been scored (see merge_typicality.py-style logic in analyze.py).

Usage:
  python typicality.py --model EleutherAI/pythia-70m --model-tag ref1 \
      --items ../data/runs/gen-smoke.jsonl --out ../data/runs/typ-ref1.jsonl \
      [--frame completion-prefix --prompts ../data/items/prompts.jsonl]
Item files need: item_id, text (and prompt_id when --frame completion-prefix).

SCORING FRAMES (design amendment A1, 2026-09-03 — see design-twin-test-v0.md):
  standalone          raw text, no prefix. Typicality of the artifact alone.
                      Length-confounded: the first tokens have no context, so
                      their cost is amortised over item length and a 7-char
                      item is all cold-start. Found 2026-09-03 hiding OLMo's
                      shared-diet collapse (RESULTS §8). Kept for reproduction.
  completion-prefix   the exact generation frame ("Writing prompt: …\\n\\nResponse:")
                      is placed in front of the item and EXCLUDED from the loss.
                      Only item tokens are scored, each in the context it was
                      generated in. Measures typicality of the story AS AN
                      ANSWER to the prompt — the frame the judge also sees.
Prefix and item are tokenized separately and concatenated as ids, so no BPE
merge crosses the boundary — matching generation, where item tokens were
produced after the frame's last token. generate.py strips the completion, so
any leading whitespace token the model emitted after "Response:" is not
reconstructed; identical for every item, so it cancels in every comparison.
"""
import argparse, json, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Must stay byte-identical to generate.py's COMPLETION_FRAME.
COMPLETION_FRAME = "Writing prompt: {prompt}\n\nResponse:"
FRAMES = ("standalone", "completion-prefix")


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def mean_nll(model, tok, text, device, max_len=512, prefix=None):
    """Mean per-token NLL of `text`. With `prefix`, the prefix conditions the
    model but contributes nothing to the loss.
    Returns (nll, n_scored_tokens, per_token_nll, truncated). `truncated` is True when
    the item alone exceeded max_len and its tail was dropped — Phase A scored this
    silently (RESULTS-B0 §2); Phase B refuses to (PHASE-B-protocol.md Rule 2)."""
    if prefix is None:
        full = tok(text, return_tensors="pt")["input_ids"]
        ids = full[:, :max_len]
        n_prefix = 0
        truncated = full.shape[1] > max_len
    else:
        p_ids = tok(prefix, return_tensors="pt")["input_ids"]
        full = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        t_ids = full[:, :max_len]
        truncated = full.shape[1] > max_len
        ids = torch.cat([p_ids, t_ids], dim=1)
        n_prefix = p_ids.shape[1]
    if ids.shape[1] - n_prefix < 1 or ids.shape[1] < 2:
        return None, 0, [], truncated
    ids = ids.to(device)
    out = model(ids)
    logits = out.logits[:, :-1, :]
    targets = ids[:, 1:]
    # position j in `targets` is token j+1 of `ids`; score only item tokens.
    # In standalone mode the first token is never scored (it has no predecessor),
    # which is the cold-start asymmetry the prefix frame removes.
    start = max(n_prefix - 1, 0)
    logits = logits[:, start:, :]
    targets = targets[:, start:]
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).float()
    return float(per_token.mean()), int(targets.shape[1]), [round(float(v), 5) for v in per_token], truncated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-tag", required=True, help="short label for this reference, e.g. olmo2-base")
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frame", choices=FRAMES, default="standalone",
                    help="scoring frame; completion-prefix needs --prompts")
    ap.add_argument("--prompts", default=None, help="prompts.jsonl (id, text) for completion-prefix")
    ap.add_argument("--max-len", type=int, default=512,
                    help="item token cap. 512 reproduces Phase A; Phase B uses 1024 (protocol Rule 2)")
    ap.add_argument("--save-token-nll", action="store_true",
                    help="write per-token NLL arrays so window variants are offline (Phase A open item 5)")
    ap.add_argument("--fail-on-truncation", action="store_true",
                    help="abort instead of scoring a truncated item; Phase B runs pass this")
    ap.add_argument("--resume-from", default=None,
                    help="existing partial JSONL under the SAME reference AND frame: its rows are "
                         "copied through and their item_ids skipped. Added 2026-09-01 after judgeK "
                         "died 1160/1389 into a pass and the whole pass had to be redone.")
    args = ap.parse_args()

    prompts = {}
    if args.frame == "completion-prefix":
        if not args.prompts:
            sys.exit("--frame completion-prefix requires --prompts")
        prompts = {p["id"]: p["text"] for p in map(json.loads, open(args.prompts))}

    device = pick_device()
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

    done = {}
    if args.resume_from and Path(args.resume_from).exists():
        for line in open(args.resume_from):
            r = json.loads(line)
            # Mixing references OR frames inside one output file would silently corrupt
            # the merge — a row's NLL is only meaningful paired with the model and the
            # frame that made it.
            if r.get("ref_tag") != args.model_tag:
                sys.exit(f"resume file holds ref_tag={r.get('ref_tag')!r} but this run is "
                         f"{args.model_tag!r} — refusing to mix references in one output")
            if r.get("frame", "standalone") != args.frame:
                sys.exit(f"resume file holds frame={r.get('frame', 'standalone')!r} but this run is "
                         f"{args.frame!r} — refusing to mix frames in one output")
            done[r["item_id"]] = line
        print(f"resuming: {len(done)} rows carried over", file=sys.stderr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w") as out:
        for line in done.values():
            out.write(line if line.endswith("\n") else line + "\n")
            n += 1
        for line in open(args.items):
            item = json.loads(line)
            if item["item_id"] in done:
                continue
            prefix = None
            if args.frame == "completion-prefix":
                prefix = COMPLETION_FRAME.format(prompt=prompts[item["prompt_id"]])
            nll, n_tok, per_token, truncated = mean_nll(
                model, tok, item["text"], device, max_len=args.max_len, prefix=prefix)
            if truncated and args.fail_on_truncation:
                sys.exit(f"{item['item_id']} exceeds --max-len {args.max_len}; refusing to score a "
                         f"truncated item (protocol Rule 2). Raise the cap or exclude the item upstream.")
            if nll is None:
                continue
            row = {
                "item_id": item["item_id"],
                "ref_model": args.model,
                "ref_tag": args.model_tag,
                "frame": args.frame,
                "mean_nll": round(nll, 5),
                "n_tokens": n_tok,
                "n_chars": len(item["text"]),
                "max_len": args.max_len,
                "truncated": truncated,
            }
            if args.save_token_nll:
                row["token_nll"] = per_token
            out.write(json.dumps(row) + "\n")
            n += 1
    print(f"scored {n} items under {args.model_tag} [{args.frame}] -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
