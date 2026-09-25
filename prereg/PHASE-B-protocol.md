> **Public copy.** Cleaned from a dated working document. Rules, dates, thresholds and the expectations written before each run are unchanged. Conversational notes (who approved what, internal review logistics, compute scheduling) were removed. The original, with version history, is kept by the author.

# Phase B protocol: story completion, length, and the diversified arm

2026-09-05 ~22:3x CDT. Written BEFORE any generation, scoring or judging under it.

## The experiment this governs

One writer (`HuggingFaceH4/zephyr-7b-beta`, bare completion mode as in Phase A and the 09-05
smoke). For each story prompt it writes stories two ways: ORDINARY (the Phase A frame, ordinary
sampling) and DIVERSIFIED (same frame plus an instruction to write an unlikely response).
Pairs of one ordinary and one diversified story go to judges from different training stages,
opposite lineage only, both criteria, both display orders. The question: does a later-stage judge
prefer the ordinary story more often than an earlier one? This is the §11.4 tripwire. A positive
growing slope revives the sign-reversed reading; anything else leaves the Phase A headline standing.

## Rule 1: only finished stories count

A generation is COMPLETE when all of the following hold. Mechanical checks run on the raw record
(`generate.py` termination metadata plus text), never on hand-edited text.

- `termination.stop_reason == "eos"`.
- `generated_token_count < max_new_tokens`, strictly. EOS landing on the last allowed token is
  ambiguous and is excluded as `eos_at_cap`.
- Stripped text ends with `.`, `!` or `?`, optionally followed by closing quotes or brackets.
  Reason: the smoke's cap-hit ends with "and"; punctuation catches that class cheaply. It is a
  necessary check, not proof of a narrative ending.
- No leaked frame or chat text: `Writing prompt:`, `Response:`, `<|`, `[INST`, `###`. No unfilled
  placeholders (`[insert`, `[name]`, `{name}` and the like; the B0 screen found real ones).
- At least 100 words (Rule 3 floor).
- Not byte-identical to another story in the same arm and prompt (B0 found 12 identical pairs).

Anything failing is EXCLUDED, never trimmed and kept, and counted per arm and per reason.
Exclusion counts are reported before any judging, because the survivors of a cap are the
shorter, more conventional stories: exclusion is itself a typicality filter, and the diversified
arm will probably lose more. Tripwire: more than 10% excluded in either arm, or a gap of more than
5 points between arms, stops the pilot before judging and reopens the cap or the elicitation
wording. Reason for 10%: the smoke lost 1 of 4 at 512; the raised cap should make cap loss rare,
and a double-digit loss means the cap is still doing the ending.

A blind read of a sample (one pair per prompt, key held apart) happens after the mechanical
screen and before judging. It is a screen for coherence and prompt fit, not ground truth and not
a rater assignment.

## Rule 2: the budget must be non-binding

| Setting | Phase A | Phase B | Reason |
|---|---|---|---|
| Generation cap (`--max-new-tokens`) | 220 | 768 | Smoke at 512 lost 1 of 4 and had a natural ending at 510. 768 gives 50% headroom over the longest observed story. |
| Judge context (`judge.py MAX_LEN`) | 1,024 | 2,048 | Frame (~60 tokens) plus two 768-token stories is ~1,600. At 1,024 the judge truncates its own instruction on any pair over ~480 tokens each; two of four smoke stories were. |
| Typicality scorer (`typicality.py max_len`) | 512 | 1,024 | Prompt prefix (~35 tokens) plus 768 fits. |
| Per-token NLL arrays | not saved | saved | Phase A item 5: window variants become offline. |

Phase A numbers are untouched: Phase B judging is a new run on new items. Before launch, every
intended reader and judge tokenizes the actual inputs and the longest pair in both orders is
checked against these limits; a 7B in fp16 at 2k context fits a T4.

## Rule 3: matched length, no stubs

- Floor: 100 words. Reason: below that, coherence and craft cannot be rated; the Phase A bank has
  a 7-character "story".
- Pairing: within each prompt, sort both arms' complete stories by word count and pair adjacent
  ordinary/diversified stories without reuse (the B0 builder's shape, across arms instead of within
  one group). Reject a pair whose longer/shorter word ratio exceeds 1.25. Reason: Phase A §10 shows
  length bias in a criterion-dependent direction; 1.25 keeps a pair inside "about the same length"
  while leaving enough pairs from 4 samples per arm. Report pairs kept and pairs rejected.
- Log length gap stays in the analysis as a covariate even after matching.
- Pair orientation by stable hash, both display orders judged, opposite-lineage judges only,
  same pairs to every judge stage.

## Option B: one unlikely story per call

Diversified frame, one story per call, same sampling as ordinary (temperature 1.0, top-p 0.95):

```
Writing prompt: {prompt}

Write a response that most writers would be unlikely to produce: one you would give a low
probability, under 10 percent, among all the responses you could write. It must still be a
complete story that fits the prompt.

Response:
```

Ordinary frame: `generate.py COMPLETION_FRAME`, unchanged from Phase A.

Rejected alternative (A): five alternatives with probabilities in one call. It splits one budget
five ways, each story lands near a third of ordinary length, and Rule 3 pairing collapses.

Scoring frame: BOTH arms are scored and judged under the ordinary frame; the diversified
instruction is provenance, kept in the record, not part of the ruler or shown to judges. This is
the B0 "common evaluation frame" amendment (PHASE-B0-feasibility.md). Flagged: it changes A1's
wording from exact generation frame to common evaluation frame for these items only.

## Manipulation check, before any judging

Positive control first. Score every complete story with the two base readers used in B0
(`olmo2-base` for Zephyr writing). The diversified arm must be measurably less typical than the
ordinary arm (higher mean per-token NLL under the prefix frame, on matched pairs). If it is not,
the pilot has nothing to judge and Option B's wording is revised, not the judges.

Confirmation-bias check, written before data: a subtle fault in the completion screen that
rejects diversified stories more would shrink the contrast and push toward the null (the Phase A
headline). So per-arm exclusion counts are printed before the manipulation check, not after.

## Pilot scope

Proposed: the 12 `ff-*` story prompts only, 4 samples per arm per prompt, 96 generations, fresh
run namespace, per-record flush, no resume. At the smoke's ~16 s per 512-token story, 768 tokens
runs ~25 s, so ~40 minutes of generation, inside the ~60-minute VM-death horizon. Exact session
cap, supervisor and stop verification as in the smoke.

## Amendments after first pilot data (labelled, 2026-09-05 ~23:3x)

- Placeholder check narrowed AFTER seeing the pilot: the first pattern flagged "[Name withheld]"
  signing a letter inside the ff-03 story, which is a device, not an unfilled slot. Now only
  "[insert ...]", a bare "[name]"/"[title]"/"[character]"/"[your name]" or "{slot}" count. The
  affected item was a cap-hit anyway, so no kept/excluded verdict changed.

## Still open

- Meaningful-effect threshold, sample size for anything beyond this pilot, and the expert panel:
  not decided, not implied by this file.
- Whether excerpt-scoped work is ever revived: parked as a separate design with a different claim.

## Amendment B1 (2026-09-06, labelled; recorded here 2026-09-22): the treatment arm is decoding, not wording

Option B (the "under 10 percent probability" wording) FAILED its manipulation check on 2026-09-06:
paired delta −0.031 nats/token, 95% [−0.097, +0.037], 11/29 pairs in the intended direction, with
the Phase A stage gap visible on the same reader as positive control (RESULTS-B-pilot §Manipulation
check). The author chose the decoding lever the same day.

**Treatment arm = `decoding`:** the ordinary frame, ordinary top-p 0.95, sampling temperature
raised. Judges and scorers see the identical frame for both arms; only the sampling changed.

**Dose rule, pre-registered in the working log at 12:5x on 2026-09-06 before any dose data:** the
LOWEST temperature that passes the Rule 1 screen at ≥90% AND shows a detectable typicality shift
under `olmo2-base` (bootstrap interval above zero). Sweep planned at 1.3 / 1.6 / 1.9; the 3,000 s
budget cut it at 24 / 2 / 0 stories. **T = 1.3 satisfied the rule** (23/24 screen; delta +0.768,
95% [+0.595, +0.908], 19/20). The 1.6 dose was not needed for the rule and its two stories read as
salad (mean NLL 3.48). The arm was then filled to 12 × 4 at T = 1.3 (47 distinct stories; a
duplicate session under a reused seed was dropped, see RESULTS-B-pilot §T = 1.3 arm).

**Caveat carried into judging:** temperature raises per-token NLL under any reader by construction,
so the ruler moving is expected. Whether T = 1.3 stories are still coherent stories is a human
question; the blind read (4 pairs, key held apart) answers it and runs BEFORE judging.

Sampling seed is now part of the run spec and validated per record; any session that must produce
independent samples of an existing arm uses a new seed.

## Pre-registered read of the pilot judge run (written 2026-09-22, before any judge output exists)

Judges: the four OLMo-2 stages (base / sft / dpo / final), opposite lineage to the Zephyr writer.
Phase A competence gates (7/7 pass under "better") carry over: identical checkpoints, no re-gating.
Pairs: the 32 in `screen-v4-decoding-t1.3-arm/pairs.jsonl`, both criteria, both display orders,
`--max-len 2048 --fail-on-truncation`.

- The Phase A headline is REVIVED in its sign-reversed reading only if the judge preference for the
  ordinary (more typical) story is positive AND grows monotonically across judge stages under the
  "better" criterion, after conditioning on the judge's own value call (Amendment A3), with the
  bootstrap interval on the base-to-final slope above zero.
- A flat or non-monotone preference, or one that vanishes after conditioning on value, leaves the
  Phase A headline standing ("the generator drifts; the judge does not follow"), and the decoding
  manipulation is reported as a successful within-stage typicality manipulation that the judges
  did not track.
- A preference for the ordinary story at EVERY stage including base is a quality effect, not a
  drift effect; the blind-read coherence scores are the tiebreaker on whether hot stories are
  worse stories.
- 32 pairs is a pilot. No sample-size or effect-size claim follows from it either way.
