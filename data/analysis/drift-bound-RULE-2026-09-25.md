# Drift bound: rule written before computing (2026-09-25 14:56 CDT)

Post hoc (not preregistered); the paper labels it so.

Question: how much could each trained judge's taste for predictable stories have GROWN beyond its own base judge's, given the data?

- Estimand: drift_s = -(b_s - b_base), where b is the preregistered-form slope (p_x ~ dtyp + dlen, item-base reader, all pairs, same code path as judge_trajectory.py). Positive drift = moved TOWARD the predictable story (the twin direction).
- Pairs: every stage within a family judged the identical 200 pairs (checked 14:56), so the bootstrap is PAIRED: each resample draws 10 prompts with replacement and refits base and stage s on the same draw.
- Bootstrap: 5000 resamples, cluster = prompt, seed 411 (as judge_trajectory.py).
- Reported bound: the 95th percentile of drift_s (one-sided 95% upper bound). Also the point estimate and the two-sided 95% interval.
- Primary criterion: "better" (the one every judge passes). Secondary: "creative".
- Translation into pick probability: bound x each family's writer drift from base to SFT under the stranger view (Table 1): OLMo-2 1.915 x 0.092 = 0.176 nats/token; Zephyr 2.105 - 1.581 = 0.524. That gives the largest shift in a judge's pick probability, from training-grown taste alone, between a base story and an SFT story that differ by that much predictability.
- No equivalence verdict: no smallest-effect-of-interest was set in advance, so the paper reports the bound, never "equivalent". Wording must avoid "the data rule out" (do-not-reintroduce list).
- Output: data/analysis/drift-bound-2026-09-25.txt, script code/drift_bound.py. Whatever it shows goes in the paper.

## Amendment 1 (2026-09-25 14:57 CDT, after the first run, translation only)
The translation above used the wrong ruler. OLMo-2 judges read ZEPHYR stories and Zephyr judges read OLMo-2 stories, and dtyp is scored by each story's OWN family base model, so Table 1's stranger view (the other family's reader) is not in the slope's units. Replaced with the mean |dtyp| of the 200 pairs each judge family actually judged (same reader as the slope): OLMo-2 judges 0.400 nats/token, Zephyr judges 0.330. The bound itself (per nat), the bootstrap and the estimand are unchanged; the first run's output is kept in git history.

## Amendment 2 (2026-09-25 15:21 CDT, after the v5 verifier)
The bound was computed only with the item-base reader, which scores base-stage stories with their own writer (the self-scoring the paper flags). Add the judge-family reader (Sens3 in trajectory-2026-09-04.txt; no self-scoring) as a second run, same estimand, same bootstrap, same seed; ruler = mean |dtyp| of the judged pairs under THAT reader. The paper reports both readers' bounds and leads with the larger. Added after an independent check of the paper.
