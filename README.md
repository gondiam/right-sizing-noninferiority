# right-sizing-noninferiority

Can you switch to the cheaper model? Treated as a non-inferiority trial: the margin is
fixed in advance and the cheaper model has to prove it loses less than that.

**Status: work in progress.** The analysis plan is in [PREREGISTRATION.md](PREREGISTRATION.md).
No model has been scored on the test set yet, and there are no results.

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
