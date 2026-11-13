# Preregistration

**Status: DRAFT, pending the author's sign-off on the margins.** The dev run is done
(`results/dev/`). This file is committed and pushed before any model is scored on the
test set, and the git history is the timestamp. Any later change is listed under Deviations, with its reason.

## Question

Is a smaller or quantized model non-inferior to a larger one on MedQA (4 options), within
a margin fixed in advance? "No significant difference" is not the claim: with few items
nothing differs from anything. The burden of proof is on the cheaper model.

## Fixed before the test run

- Data: the 1,273 test items of `GBaker/MedQA-USMLE-4-options` at revision `0fb93dd`.
  The dataset has no dev split, so dev is 500 train items drawn with seed 0. Dev is used
  for the pilot, the prompt and the cascade threshold. Test items are used for nothing
  else than the one confirmatory run.
- Models: `Qwen/Qwen3-1.7B` (`70d244c`), `Qwen/Qwen3-4B` (`1cfa9a7`), `Qwen/Qwen3-8B`
  (`b968826`). 4-bit means bitsandbytes NF4 with float16 compute. The 8B in float16 does
  not fit a 15 GB T4, so every comparison with the 8B uses 4-bit on both sides.
- Scoring: `kaggle/run.py` as of this commit. One prompt, thinking disabled, the reply
  forced to start with `Answer:`. The prediction is the letter with the highest
  next-token probability among A, B, C and D. This is greedy decoding restricted to the
  four options, so every item gets an answer and nothing is unparseable. The models do not
  reason before answering; accuracies are for this protocol only.
- Cascade: answer with the 4B (4-bit); when the probability of its chosen letter is below
  a threshold, use the 8B's answer instead. Threshold rule, applied to dev only: the
  lowest of the 0%, 5%, ..., 95% quantiles of the 4B's dev confidence whose cascade
  accuracy is within 1 point of the 8B's. Threshold: **0.996**, which sends 50% of dev
  items to the 8B (cascade 62.2%, 8B 63.0% on dev). The first draft of this rule used a
  fixed grid from 0.30 to 1.00 in steps of 0.05. The 4B is overconfident (45% of dev
  items fall between 0.95 and 1.00), so that grid jumped from 30% to 75% of items sent
  to the 8B. The grid was changed to quantiles after seeing dev and before any test run.
- Margins, written before the dev run and not changed afterwards:
  - H1, 4 points. The 4B has half the parameters of the 8B. For about half the memory
    and compute, a team would accept at most one more wrong answer in 25 questions.
  - H2 and H3, 2 points. Quantization and a cascade are sold as changes that keep
    quality, so the tolerated loss is smaller: one more wrong answer in 50.

## Confirmatory hypotheses

- H1, size: 4B is non-inferior to 8B (both 4-bit), margin 4 points.
- H2, quantization: 4B in 4-bit is non-inferior to 4B in float16, margin 2 points.
- H3, cascade: the 4B-then-8B cascade is non-inferior to the 8B alone, margin 2 points.

## Analysis

`analyze.py test` as of this commit, with `llm-judge-stats` 0.3.0.

- Outcome: correct or not, per item. Paired difference cheaper minus reference.
- `noninferiority()`: two-sided 95% Agresti-Min CI for the paired difference and the
  one-sided p-value from the same statistic (H0: the loss is at least the margin).
- The three p-values are Holm-adjusted. A hypothesis is confirmed when its adjusted
  p < 0.025. Without the adjustment this is the same as the lower CI limit being above
  minus the margin; with it the rule is stricter. CIs are reported unadjusted.
- Sensitivity control: 1.7B (4-bit) against 8B (4-bit) with `compare()`. The 1.7B must
  come out lower and distinguishable. If it does not, the setup cannot tell models
  apart and no non-inferiority claim is made.
- Every difference is reported with its CI, whether non-inferiority is shown or not. A
  failed hypothesis is reported as the largest loss the CI rules out.

## Sample size

The test set is fixed at 1,273 items. With no true difference, the items needed are
7.85 x discordance / margin^2 (one-sided alpha 0.025, power 0.80):

| Discordance | Margin 2 points | Margin 4 points |
|---|---|---|
| 6% | 1,178 | 295 |
| 20% | 3,925 | 982 |
| 30% | 5,888 | 1,472 |

Power on the 1,273 test items with the discordance measured on dev, assuming no true
difference:

| Hypothesis | Dev discordance | Dev difference | Power |
|---|---|---|---|
| H1 size | 26.4% | -6.4 points | 0.79 |
| H2 quantization | 12.0% | -1.2 points | 0.54 |
| H3 cascade | 8.4% | -0.8 points | 0.69 |

These are upper bounds. On dev the 4B is 6.4 points below the 8B (56.6% against 63.0%),
beyond the 4-point margin, so H1 is expected to fail. The cascade threshold was picked on
the same dev items, so its dev difference is optimistic. If the cheaper model is truly worse, power is
lower: with a true loss of 1 point and 20% discordance, H1 would need about 1,745 items.
No hypothesis reaches 0.80 power. All three are run and reported whatever the outcome.

## Secondary (reported, not used for the main claims)

Accuracy per configuration with Wilson CIs; equivalence (TOST) with the same margins;
share of items on which the cascade calls the 8B; seconds and peak GPU memory per
configuration.

## Stopping rules

- One test run per configuration. No interim looks, no rerun with another prompt.
- If a configuration fails to load on the T4 in the dev run, it is replaced before the
  test run and the change is listed under Deviations.

## Deviations

None.
