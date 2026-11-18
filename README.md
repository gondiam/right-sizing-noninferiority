# right-sizing-noninferiority

Can you switch to the cheaper model? Treated as a non-inferiority trial: the margin is
fixed in advance and the cheaper model has to prove it loses less than that.

**Status: done.** The plan was fixed in [PREREGISTRATION.md](PREREGISTRATION.md) (commit
`76b0b91`) and the test set was scored once, from that commit.

## Results

None of the three cheaper options is shown to be non-inferior. The sensitivity control
passes, so the setup can tell models apart and the failures are not a broken eval.

| Hypothesis | Difference (95% CI) | Margin | Verdict |
|---|---|---|---|
| H1 size: 4B vs 8B (both 4-bit) | -9.2 points (-12.0 to -6.4) | 4 | Inferior |
| H2 quantization: 4B 4-bit vs fp16 | -3.3 points (-5.1 to -1.5) | 2 | Not shown |
| H3 cascade: 4B, then 8B when unsure, vs 8B | -2.4 points (-3.9 to -0.8) | 2 | Not shown |
| Control: 1.7B vs 8B (both 4-bit) | -23.4 points (-26.6 to -20.2) | | Lower, as required |

Holm-adjusted, one-sided alpha 0.025, 1,273 paired items. Equivalence (TOST) is not shown
for any of the three either.

- **H1** fails outright: the whole CI lies beyond the 4-point margin.
- **H2 and H3** are not shown, but they are not clean ties either. Both CIs exclude zero,
  so each option is measurably worse; whether the loss exceeds 2 points is left open.
- A "no significant difference" test would have been wrong in the other direction here:
  every cheaper option is significantly worse, and only non-inferiority asks the
  question a team actually has, which is whether the loss is small enough to accept.

| Configuration | Accuracy (95% CI) | Seconds on a T4 | Peak GPU memory |
|---|---|---|---|
| Qwen3-1.7B, 4-bit | 40.4% (37.7 to 43.1) | 201 | 4.7 GB |
| Qwen3-4B, 4-bit | 54.6% (51.9 to 57.3) | 384 | 7.0 GB |
| Qwen3-4B, fp16 | 57.9% (55.2 to 60.6) | 196 | 11.6 GB |
| Qwen3-8B, 4-bit | 63.8% (61.1 to 66.4) | 594 | 10.6 GB |
| Cascade (4B 4-bit, 8B on 49.2% of items) | 61.4% (58.7 to 64.1) | | |

Exploratory, not preregistered: on this GPU 4-bit was not cheaper in time. The 4B in
4-bit took about twice as long as in fp16; it only saved memory. The cascade sent 49.2%
of items to the 8B, so running it as scored here (the 4B on every item, then the 8B on
the uncertain half) would take about 384 + 0.49 x 594 = 676 seconds, more than the 8B
alone. Accuracies hold for this protocol only: one prompt, no reasoning, the answer read
from the next-token probabilities of A to D.

## Why

"We ran both models and saw no significant difference" is the usual argument for moving
to a smaller model. It proves nothing: with a small eval, no two models differ
significantly. Clinical trials settled this decades ago (ICH E9 and E10) by reversing the
burden of proof. The null hypothesis is that the cheaper option is worse by at least a
margin, and the data have to reject it.

## Design

- Data: MedQA-USMLE, 4 options, 1,273 test items (CC-BY-4.0).
- Models: Qwen3 1.7B, 4B and 8B on a free Kaggle T4. No API is called.
- Three hypotheses: a smaller model (margin 4 points), 4-bit quantization (2 points) and
  a confidence-based cascade (2 points). A 1.7B model is the sensitivity control: it must
  come out inferior, or the setup cannot tell models apart.
- Statistics: `noninferiority()` from
  [llm-judge-stats](https://github.com/gondiam/llm-judge-stats) 0.3.0, Holm-adjusted.

## Run

```
kaggle/push.sh dev               # score the dev items on a Kaggle T4
uv run python analyze.py dev     # discordance, power, cascade threshold
kaggle/push.sh test              # once, after the preregistration is pushed
uv run python analyze.py test
```
