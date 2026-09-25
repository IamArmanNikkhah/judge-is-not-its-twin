# Judging protocol — base-executable forced choice (T1 correction #1)

**Why not "rate 1–10":** base models can't reliably follow scoring instructions, so instruction-style judging makes bias "appear" at SFT for competence reasons — the null predicts the headline. The protocol must be executable by every stage.

## Core measurement: pairwise forced choice via logits
For an item pair (X, Y) matched on prompt and length bin:
```
Prompt template (few-shot, completion-style — no chat template for base stages):
  "Here are two responses to the same writing prompt.
   Prompt: {prompt}
   Response A: {X}
   Response B: {Y}
   The more creative response is Response"
Measurement: P("A") vs P("B") from next-token logits. No sampling, no parsing.
```
- Position bias handled by construction: every pair evaluated in both orders; preference = mean of the two (the 2305.17926 swap, done at logit level).
- Same template across all stages and lineages; chat stages ALSO run the bare completion template (primary) and their chat template (sensitivity check) — cross-stage comparison always uses the shared bare template.
- Secondary probe, same trick with "better" replacing "more creative" — separates creativity-preference from generic quality-preference.

## Per-stage validity gate (the competence check)
Before any bias slope is interpreted: the stage must discriminate source-4 garbage (high-temperature base samples rated low-value by the anchor) from clearly-competent text at better than chance on held-out validity pairs. Stages failing the gate are reported but excluded from gradient claims. Expected: all stages pass the logit protocol (it needs no instruction-following); if a base stage still fails, the SFT→DPO cell carries the paper (T1 verdict).

## Judge-side bias estimate J
Per judge stage: logistic model of forced-choice outcome on Δtypicality(X,Y) + Δlength + Δvalue(anchor), aggregated over pairs. J = the typicality coefficient minus the anchor panel's coefficient on identical pairs (bias is a GAP vs the answer key, not a raw preference).

## Self-pair exclusion
A judge never sees pairs containing its own lineage's generations (any stage) in the primary analysis. Sensitivity run includes them to quantify the self-preference component (2404.13076 / 2410.21819 comparison point).

## P4 mechanism regressors (T1 correction #2)
- shared-diet rarity: mean NLL under leave-own-lineage-out base references (configs/lineages.yaml rule)
- own-diet delta: Δppl = NLL(own aligned stage) − NLL(own base) per item
- Pilot collinearity diagnostic: if |r(shared, Δppl)| > 0.85 on the pilot bank → fall back to disagreement-cell item selection (items where the two measures disagree by design, e.g. instruct-flavored text).
