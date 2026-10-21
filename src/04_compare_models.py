"""
Step 4b: Compare the baseline and model 2 with paired statistical tests.

Loads the CV fold scores and test-set predictions saved by
02_baseline_model.py and 03_train_model2.py -- no retraining happens here.

Two paired tests, because they answer slightly different questions and
neither one substitutes for the other:

1. Paired t-test on matched CV fold accuracies (same 5 folds for both
   models) -> is the *typical* difference in accuracy across resamples
   of the training data statistically distinguishable from zero?
2. McNemar's test on the held-out test set (same 556 examples for both
   models) -> on this one particular sample, does one model get
   significantly more of the disagreement cases right than the other?

Implemented directly with scipy (no statsmodels dependency, since this
environment has no internet access to install it).
"""

import numpy as np
import pandas as pd
from scipy import stats
from config import RESULTS

BASELINE_CV_CSV = RESULTS / "baseline_cv_scores.csv"
MODEL2_CV_CSV = RESULTS / "model2_cv_scores.csv"
BASELINE_TEST_CSV = RESULTS / "baseline_test_predictions.csv"
MODEL2_TEST_CSV = RESULTS / "model2_test_predictions.csv"
ALPHA = 0.05

def mcnemar_test(b, c, exact_threshold=25):
    """
    McNemar's test on a paired 2x2 contingency table.
    b = count where model A correct, model B wrong
    c = count where model B correct, model A wrong
    Uses the exact binomial form for small samples (b + c < exact_threshold),
    otherwise the chi-square approximation with continuity correction.
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    if n < exact_threshold:
        result = stats.binomtest(min(b, c), n, 0.5, alternative="two-sided")
        return float(n), result.pvalue
    statistic = (abs(b - c) - 1) ** 2 / n
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    return statistic, p_value

def main():
    base_cv = pd.read_csv(BASELINE_CV_CSV)
    m2_cv = pd.read_csv(MODEL2_CV_CSV)
    base_test = pd.read_csv(BASELINE_TEST_CSV)
    m2_test = pd.read_csv(MODEL2_TEST_CSV)

    print("=" * 60)
    print("CV ACCURACY BY FOLD")
    print("=" * 60)
    for f, b, m in zip(base_cv["fold"], base_cv["accuracy"], m2_cv["accuracy"]):
        print(f"Fold {f}: baseline={b:.3f}  model2={m:.3f}")
    print()

    # --- Test 1: paired t-test on matched CV fold accuracies ---
    diffs = m2_cv["accuracy"].values - base_cv["accuracy"].values
    t_stat, p_value = stats.ttest_rel(m2_cv["accuracy"], base_cv["accuracy"])
    print("=" * 60)
    print("PAIRED T-TEST on matched CV fold accuracies (model2 - baseline)")
    print("=" * 60)
    print(f"mean difference: {diffs.mean():+.4f}")
    print(f"t-statistic: {t_stat:.3f},  p-value: {p_value:.4f}")
    if p_value < ALPHA:
        print(f"-> significant at alpha={ALPHA}: model2's CV accuracy differs from baseline's")
    else:
        print(f"-> NOT significant at alpha={ALPHA}: no evidence the two models differ in CV accuracy")
    print()

    # --- Held-out test set metrics side by side ---
    print("=" * 60)
    print(f"HELD-OUT TEST SET (n={len(base_test)})")
    print("=" * 60)
    base_correct = (base_test["y_pred"] == base_test["y_true"]).values
    m2_correct = (m2_test["y_pred"] == m2_test["y_true"]).values
    print(f"accuracy: baseline={base_correct.mean():.3f}  model2={m2_correct.mean():.3f}")
    print()

    # --- Test 2: McNemar's test on held-out test set (paired, same items) ---
    both_correct = np.sum(base_correct & m2_correct)
    base_only = np.sum(base_correct & ~m2_correct)
    m2_only = np.sum(~base_correct & m2_correct)
    both_wrong = np.sum(~base_correct & ~m2_correct)

    statistic, p_value = mcnemar_test(base_only, m2_only)

    print("=" * 60)
    print("MCNEMAR'S TEST on held-out test set (paired predictions)")
    print("=" * 60)
    print(f"both correct: {both_correct}  |  baseline only correct: {base_only}  |  model2 only correct: {m2_only}  |  both wrong: {both_wrong}")
    n_disc = base_only + m2_only
    test_kind = "exact binomial" if n_disc < 25 else "chi-square, continuity-corrected"
    print(f"discordant pairs: {n_disc}  |  test used: {test_kind}  |  p-value: {p_value:.4f}")
    if p_value < ALPHA:
        winner = "model2" if m2_only > base_only else "baseline"
        print(f"-> significant at alpha={ALPHA}: {winner} makes significantly fewer test-set errors")
    else:
        print(f"-> NOT significant at alpha={ALPHA}: the disagreement pattern is consistent with chance")

if __name__ == "__main__":
    main()
