"""Pilot (dev) and confirmatory (test) analysis of results/<split>/<config>.jsonl.

uv run python analyze.py dev    # discordance, power on the 1,273 test items, cascade threshold
uv run python analyze.py test   # the three hypotheses; needs THRESHOLD fixed from dev
"""

import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
from llm_judge_stats import adjust_pvalues, compare, equivalence, noninferiority
from scipy.stats import binomtest

RESULTS = Path(__file__).parent / "results"
ALPHA, N_TEST = 0.025, 1273
# hypothesis: (reference, cheaper, margin). "cascade" is built from 4b-4bit and 8b-4bit.
HYPOTHESES = {
    "H1 size: 4B vs 8B": ("8b-4bit", "4b-4bit", 0.04),
    "H2 quantization: 4B 4-bit vs fp16": ("4b-fp16", "4b-4bit", 0.02),
    "H3 cascade: 4B then 8B vs 8B": ("8b-4bit", "cascade", 0.02),
}
SLACK = 0.01  # the cascade may lose at most 1 point to the 8B on dev
THRESHOLD: float | None = 0.996  # from `analyze.py dev`, fixed before the test run


def load(split: str, config: str) -> tuple[np.ndarray, np.ndarray]:
    """Per item: answered correctly (bool) and the probability of the chosen letter."""
    rows = [json.loads(x) for x in (RESULTS / split / f"{config}.jsonl").open()]
    return np.array([r["pred"] == r["gold"] for r in rows]), np.array(
        [max(r["probs"]) for r in rows]
    )


def cascade(conf: np.ndarray, small: np.ndarray, large: np.ndarray, t: float) -> np.ndarray:
    """Keep the small model's answer when it is confident, ask the large one otherwise."""
    return np.where(conf >= t, small, large)


def choose_threshold(conf: np.ndarray, small: np.ndarray, large: np.ndarray) -> float:
    """Lowest threshold (fewest large-model calls) within SLACK of the large model on dev.

    Candidates are the 0%, 5%, ..., 95% quantiles of the small model's confidence.
    """
    for t in np.quantile(conf, np.arange(0, 1, 0.05)).round(4).tolist():
        if cascade(conf, small, large, t).mean() >= large.mean() - SLACK:
            return t
    return 1.01  # always escalate


def power(n: int, margin: float, discordance: float) -> float:
    """Chance of showing non-inferiority with n paired items if there is no true difference."""
    return NormalDist().cdf(margin * math.sqrt(n / discordance) - NormalDist().inv_cdf(1 - ALPHA))


def systems(split: str, t: float) -> dict[str, np.ndarray]:
    out = {c: load(split, c)[0] for c in ("1.7b-4bit", "4b-4bit", "4b-fp16", "8b-4bit")}
    conf = load(split, "4b-4bit")[1]
    out["cascade"] = cascade(conf, out["4b-4bit"], out["8b-4bit"], t)
    print(f"Cascade threshold {t}: the 8B is called on {(conf < t).mean():.1%} of items.\n")
    return out


def accuracies(s: dict[str, np.ndarray]) -> None:
    for name, x in s.items():
        ci = binomtest(int(x.sum()), len(x)).proportion_ci(method="wilson")
        print(
            f"  {name:<10} {x.mean():.1%}  (95% CI {ci.low:.1%} to {ci.high:.1%}, {len(x)} items)"
        )
    print()


def dev() -> None:
    small, conf = load("dev", "4b-4bit")
    t = choose_threshold(conf, small, load("dev", "8b-4bit")[0])
    s = systems("dev", t)
    accuracies(s)
    print(f"Power on the {N_TEST} test items if there is no true difference:")
    for name, (ref, cheap, margin) in HYPOTHESES.items():
        disc = float((s[ref] != s[cheap]).mean())
        diff = s[cheap].mean() - s[ref].mean()
        print(
            f"  {name}: dev discordance {disc:.1%}, dev difference {diff * 100:+.1f} points, "
            f"margin {margin * 100:.0f} points, power {power(N_TEST, margin, disc):.2f}"
        )


def test() -> None:
    if THRESHOLD is None:
        sys.exit("fix THRESHOLD from the dev run and commit the preregistration first")
    s = systems("test", THRESHOLD)
    accuracies(s)
    res = {
        name: noninferiority(s[ref], s[cheap], margin, names=(ref, cheap))
        for name, (ref, cheap, margin) in HYPOTHESES.items()
    }
    adjusted = adjust_pvalues([r.p_value for r in res.values()], "holm")
    for (name, r), p in zip(res.items(), adjusted, strict=True):
        verdict = "NON-INFERIOR" if p < ALPHA else r.verdict.upper()
        print(f"{name}\n{r}\nHolm-adjusted one-sided p = {p:.4f} -> {verdict}\n")
    print("Sensitivity control (must be distinguishable, with the 1.7B lower):")
    print(compare(s["8b-4bit"], s["1.7b-4bit"], names=("8b-4bit", "1.7b-4bit")), "\n")
    print("Secondary: equivalence (TOST) with the same margins")
    for name, (ref, cheap, margin) in HYPOTHESES.items():
        print(f"  {name}: {equivalence(s[ref], s[cheap], margin).verdict}")


def selftest() -> None:
    conf = np.array([0.9, 0.9, 0.4, 0.4])
    small, large = np.array([1, 0, 0, 0], bool), np.array([1, 1, 1, 0], bool)
    assert cascade(conf, small, large, 0.5).tolist() == [True, False, True, False]
    assert choose_threshold(conf, small, large) == 1.01  # only full escalation gets within 1 point
    assert choose_threshold(conf, large, large) == 0.4  # equal models: never escalate
    assert abs(power(982, 0.04, 0.2) - 0.80) < 0.005
    print("selftest passed")


if __name__ == "__main__":
    {"dev": dev, "test": test, "selftest": selftest}[sys.argv[1]]()
