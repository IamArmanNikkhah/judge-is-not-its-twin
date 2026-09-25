> **Public copy.** Cleaned from a dated working document. Rules, dates, thresholds and the expectations written before each run are unchanged. Conversational notes (who approved what, internal review logistics, compute scheduling) were removed. The original, with version history, is kept by the author.

# Twin Test — experiment design v1 (DRAFT, not locked)
*2026-08-31. A design review returned approve-with-corrections; both corrections are applied below. Lock still gated on: full methods re-read of 2410.21819 + 2608.23705, and checkpoint-availability verification. Own compute only.*

**Review corrections applied in this version:**
1. **Competence confound (P2):** base models may be unable to execute judging at all — bias "appearing at SFT" is then predicted by the null (capability, not bias, emerges there). Fix: base-executable judging protocol (forced-choice via logits / few-shot completion scoring) + per-stage judge-validity metric (discrimination on the source-4 garbage anchor); P2 slope reported CONDITIONAL on nonzero discriminative validity; the SFT→DPO cell (both stages competent) is the featured comparison.
2. **P4 collinearity (reparametrized):** raw own-perplexity vs reference-rarity are near-collinear across base models. Fix: shared-diet = rarity under held-out base references (unchanged); own-diet = **Δppl = ppl(own aligned) − ppl(own base checkpoint)** — isolates what post-training specifically made familiar. Pilot collinearity diagnostic with named threshold; fallback = disagreement-cell item selection (e.g. instruct-style text: low own-ppl, unremarkable base-rarity).

## Question and predictions
**Q:** Is the judge's typicality preference the same inherited bias as the generator's mode collapse?

- **P1 (twin correlation):** Across models, generator-side typicality bias G_M and judge-side typicality bias J_M correlate positively.
- **P2 (inheritance gradient — the causal lever):** Along public post-training trajectories (base → SFT → DPO/RLHF of the SAME lineage), J_M rises at the same stages where G_M rises. Prior work locates generator collapse at SFT (arXiv:2604.16027); prediction: judge bias appears there too. **Nobody has ever measured a judge at intermediate checkpoints.**
- **P3 (tail collapse, =G2):** Judge–human agreement is high on typical items (reproducing the field's ~80%) and collapses monotonically in the low-typicality tail — where creative value lives.
- **P4 (mechanism dissociation):** If judges penalize text that is atypical of the *shared* training distribution (rarity under reference corpora/base models), that supports inheritance; if they only penalize text atypical *to themselves* (own-perplexity beats reference-rarity as regressor), that's self-familiarity (Preference-Leakage territory). Competing-regressor design separates the two. Either result is informative; only P4a is the full twin claim.

## Design overview
Two measurement arms on the same item bank, then correlate.

**Arm 1 — Generator bias G_M.** For each model M: sample k≈20 generations per prompt over ~25 open-ended creative prompts (story openings, alternative uses, flash fiction), direct prompting, fixed temperature. G_M = typicality concentration of M's outputs vs the human reference distribution (fraction of outputs falling below the human distribution's τ-th typicality percentile; plus embedding-dispersion as secondary). Item-level **typicality is measured objectively**: mean log-prob under 2–3 held-out base reference models + corpus-rarity/embedding-percentile — never the judge's own perplexity (that's a P4 regressor, kept separate).

**Arm 2 — Judge bias J_M.** Each model M scores every item in the bank for creativity (and separately for quality/craft), blind, randomized order, position-safe single-item scoring. J_M = β on typicality from: (score_M − score_expert) ~ typicality + length + expert_value + fluency-errors. Judges never score their own generations (self-pairs excluded; sensitivity analysis includes them to quantify the self-preference component).

**Correlation.** Primary: Spearman(G_M, J_M) across the finished-model panel. Secondary and stronger: within-lineage stage gradients (P2) analyzed as paired trajectories; mixed-effects with family as grouping factor (n_models is small — the gradient design carries the inferential weight, not the cross-sectional N).

## Model roster
- **Trajectory arm (P2):** stacks with public intermediate checkpoints — OLMo (base/SFT/DPO/Instruct), Tülu-3 recipe checkpoints, Pythia+Zephyr-style pairs. 3 lineages × 3–4 stages, 7B-class. Own compute (RunPod/Lambda, ~$50–150 total; inference only).
- **Finished panel (P1):** ~12 API models across families and alignment intensities (GPT, Claude, Gemini, Llama-chat, Mistral, Qwen, DeepSeek…). API cost ~$200–400.

## Item bank (~250–300 items, length-binned)
1. **Human tail:** creative-writing pieces post-training-cutoff (freshly commissioned or verifiably recent — WritingPrompts is in every training set; contamination flag below).
2. **Model-typical:** direct-prompt outputs from panel models.
3. **Within-model typicality manipulation:** VS-prompted outputs from the *same* models (arXiv:2510.01171) — same capability, higher diversity; partially de-confounds typicality from quality.
4. **Base-model high-temperature samples:** the weird-and-often-bad tail — anchors the value axis so "judges correctly penalize garbage" is measurable rather than assumed.

## Ground truth (G3)
- **Expert panel:** 3–5 practicing writers (MFA/published), CAT protocol: rate NOVELTY and VALUE on separate scales, blind to source, own internalized standards (no rubric). ~250 items ≈ 4–6 hrs each. Budget $1–3K, or barter/co-credit — OPEN KNOB.
- **Crowd panel (cheap bonus):** Prolific raters on the same items → replicates Kaufman's expert–novice gap in our domain AND lets us show the field's usual oracle diverges from experts exactly in the tail (feeds the A1 argument).

## What kills it / what survives a null (Path B, stated up front)
- If J_M is flat on typicality against expert ground truth → thesis dead; the G2/P3 dataset still yields "judge-human agreement is tail-robust," an honest negative worth a short paper.
- If P1 correlation nulls but P3 tail collapse holds → the invisibility claim survives without the inheritance claim; paper re-scopes to G2+G4.
- If P4 resolves to own-perplexity only → the finding merges toward Preference Leakage; differentiation = typicality-specific, creativity-scoped extension. Weaker but publishable.

## Threats and controls
- **Weird≠good confound:** value rated separately by experts; all bias estimates conditioned on value. Source-4 items ensure the low-typicality × low-value cell is populated.
- **Length:** binned + regressed (LC-AlpacaEval lesson).
- **Contamination:** human items must post-date training cutoffs; commissioned or verified-recent. NO WritingPrompts as human anchor (it's in the corpora AND 2608.23705 already used it — differentiation bonus).
- **Fluency artifacts:** human typos/grammar tracked as covariate (2607.03025-class style effects).
- **Self-preference:** self-pairs excluded in primary; quantified in sensitivity.
- **Scoop check before lock:** re-read 2410.21819 + 2608.23705 methods sections to confirm neither buried a stage-wise or correlation analysis in results (Ginestet-class risk — a twin experiment hiding in someone's Table 3).

## Phasing (race discipline; revised after review)
- **Phase A (~2–3 wks, no humans):** minimal ~100-item bank (all four sources represented) → trajectory arm (P2) with the validity-conditional protocol, SFT→DPO cell featured → P4 pilot + collinearity diagnostic → crowd proxy anchor (gradient claims framed as anchor-differenced, which is anchor-robust). **Freeze analysis code on pilot data before full runs** (forking-paths protection for a solo paper). Output: workshop-grade P2 result (workshop target open).
- **Phase B (+3–4 wks), GATED on Phase A:** fires only if P2 gradient is nonzero and the P4 pilot passes. Full 250–300 item bank + expert panel → full paper (ICML/ACL-ARR 2027 class). The gate protects the $1–3K and the expert goodwill.
- Pre-lock verification items: (a) full methods/results re-read of 2410.21819 + 2608.23705 (abstract-level check done 2026-08-31, clean); (b) OLMo/Tülu/Pythia intermediate checkpoints actually downloadable at the listed stages.

## Open decisions (author's)
1. Expert-panel sourcing & budget ($1–3K vs barter/co-credit).
2. Model roster final cut (which 12 APIs; which 3 lineages).
3. Phase A workshop target: submit-or-skip.

---

## Amendment A1 — typicality scoring frame (2026-09-03, preregistered before the re-run)

**Finding that forced it.** The pilot scored every item standalone. The first tokens of a
standalone item have no context and are expensive; that cost is amortised over item length,
so mean per-token NLL is length-confounded. OLMo-2's aligned stages emit a tail of very short
completions (7–90 chars), which inflated their means and hid a shared-diet collapse that is
present on length-matched items (RESULTS-pilot §8, `data/analysis/robustness-2026-09-03.txt`).

**Decision.** The typicality measure is re-defined as the mean per-token NLL of the item
**conditioned on the exact generation frame** (`COMPLETION_FRAME` from generate.py placed in
front, excluded from the loss). This is `typicality.py --frame completion-prefix`. Rationale,
in order: (1) it removes the cold-start artifact at the source rather than by a length cut,
which would condition on the outcome; (2) it makes the quantity "typicality of this story as an
answer to this prompt," which is the frame the judge sees, so the P2/P4 regression of judge
preference on typicality is measured in one frame; (3) it has no free parameter, unlike
drop-first-k. The standalone frame is retained only to reproduce the pilot tables.

**What this changes.** Every shared-diet and Δppl number in RESULTS-pilot §2–§4 will move. The
pilot verdicts stand or fall on the re-scored numbers; nothing from the standalone frame ships.
The four reference passes (mistral-base, olmo2-base, olmo2-instruct, zephyr-final) are re-run
over the same 1,389-item bank; the ≥400-char cut remains a sensitivity pass. Own-diet Δppl on
long items only was unchanged under the standalone frame (robustness §R1b), so the P2
gradient is expected to survive; the OLMo shared-diet collapse is expected to appear at roughly
the length-matched magnitude (−7 to −10%). Those two expectations are written here before the
run so that a different outcome is a finding and not a fit.

## Amendment A2 — B4b judge competence gate (2026-09-03, fixed before the run)

**Purpose.** A judge that cannot do the task still yields probabilities, and a slope from a
non-judge means nothing. Worse, "bias appears with training" and "competence appears with
training" draw the same picture. Before any judge's typicality slope is read, the judge must
show it can tell an intact story from a ruined one.

**Items.** Per prompt, 6 intact stories (≥400 chars, round-robin across the seven stage
groups) each paired with three ruins of itself: **shuffled** (same words, random order),
**stub** (first sentence only), **offprompt** (an intact story answering a different prompt,
±30% length). 180 pairs, both orders, seed 411. Ruined items carry new ids and live in
`data/validity/`, outside the bank. Self-lineage exclusion does not apply (competence, not bias).

**Pass rule.** A judge stage PASSES iff, on shuffled pairs, it prefers the intact story in
≥ 80% of pairs and the Wilson 95% lower bound exceeds 50%. Stub and offprompt accuracies are
reported per stage and do not gate: a base model that ignores the prompt is a finding about
base models. Failing stages are reported but excluded from gradient claims. The 80% is a
judgment call, the author's, written here before any judge has been run.

**Expectations, written before the run.** All seven stages pass shuffled (the logit protocol
needs no instruction-following). Base stages are expected to sit near chance on offprompt and
to improve after SFT; stub is expected to be the discriminating secondary between base and
aligned stages. If a base stage FAILS shuffled, the SFT→DPO cell carries the paper (review correction 1).

## Amendment A3 — judging criteria after the B4b inversion (2026-09-03; fixed 2026-09-04 with the 80% gate and the 0.5 entanglement threshold as proposed, before any trajectory judge run)

**What A2 found.** Under the protocol's "more creative" forced choice, both base models pass the
shuffled gate (.95 / .88) and every trained stage fails it: OLMo sft/dpo/final at chance by a
slot-A freeze; zephyr-sft at .13 by confidently preferring the scrambled text; zephyr-dpo at .40.
A 2×2 diagnostic (frame: bare vs chat template; criterion: "creative" vs "better") on olmo2-sft
and zephyr-sft shows the frame changes nothing and the word changes everything: shuffled accuracy
under "better" .97 / 1.00 (OLMo) and .88 / .88 (Zephyr). Stub ≈ 1.0 everywhere; off-prompt
≈ .57–.67 everywhere (`data/analysis/gate-2026-09-03.txt`, `diag-2026-09-03.txt`).

**Reading.** Competence is established: every stage discriminates value when asked for the better
story. The "creative" failure is a criterion-comprehension breakdown in trained judges: asked for
creativity they reward maximal atypicality regardless of value (the weird≠good conflation this
design's source-4 items exist to expose). Note the direction: this is a judge rewarding
ATYPICALITY without value, which is the opposite sign to the typicality bias the paper tests;
it motivates value-conditioning, it is not an instance of the thesis.

**Decision (proposed).**
1. The competence gate runs under **"better"** on all seven stages, same 180 pairs, same seed,
   same rule (≥80% shuffled, Wilson low > 50%). A2's "creative" gate result is reported in full
   as a finding, not discarded.
2. Trajectory judge runs carry **both criteria**: "creative" is the primary taste probe, "better"
   the value axis and the protocol's preregistered secondary probe. Every bias estimate is
   conditioned on the value axis (design line 45; judging-protocol.md line 20 already say this).
3. **Per-stage non-degeneracy check on the measurement pairs themselves**, per criterion: a stage
   whose order-averaged p_x has sd < 0.07 or whose mean order-disagreement exceeds 0.45 on the
   real intact-vs-intact pairs is reported as non-executable for that criterion; its other
   criterion and the contrast still ship. Thresholds are set from the pilot base-stage run
   (sd .147, disagreement .143) and the observed freezes (disagreement .46–.68): halfway between,
   written here before any trajectory data exists.
4. Estimand added: per-pair **p_creative − p_better**, the creativity-specific component of judge
   preference, regressed on typicality alongside the two raw slopes.
5. **Limitation preregistered:** no stage reads the prompt (off-prompt ≈ .6). Claims are about
   typicality of the text, not prompt-relative fit.

**Expectations, written before the run.** All seven stages pass the "better" gate. On real pairs
no stage freezes (the freeze was triggered by scrambled text, which real pairs do not contain).
If a base stage fails "better", the review-correction fallback (SFT→DPO cell) reactivates and this amendment
says so. If the "creative" and "better" slopes are indistinguishable at every stage, the
creativity scoping collapses and the paper reframes around the criterion-breakdown finding.

**A3 addendum (2026-09-04, the author's objection): is "better" a quality axis or typicality in a costume?**
Trained judges' "better" is plausibly polish (grammar, coherence, vocabulary), and trained models
write polished, predictable text, so "better" may be entangled with typicality. Conditioning on
an entangled axis would partial out the effect under test. Falsifier, per judge stage, on the
real intact-vs-intact pairs: the correlation between p_better and Δtypicality of the pair. If
|r| exceeds 0.5 (the author's number, like the 80%), "better" is not an independent value
axis for that stage: the conditioning is dropped for that stage, the raw "creative" slope is
reported with the non-degeneracy check only, and the paper states plainly that value in Phase A
is a proxy. Under any outcome the paper says that no model-side test establishes that a judge
sees original thought; that is what the Phase B expert panel is for.
