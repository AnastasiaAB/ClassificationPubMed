"""
Step 4: Compare every model against the baseline with paired statistical tests.

Reads the per-fold CV scores and held-out test predictions saved by the training scripts
(02, 03, 07, 09); nothing is retrained here. Models whose result files are missing are skipped.

Primary question: does any challenger reliably beat the baseline?  That is one family of
comparisons (each challenger vs. the baseline), so p-values are Holm-corrected within each test
type. With 3 challengers, uncorrected p < 0.05 would give a ~14% chance of at least one false
"significant" result even if no model differed at all.

Two paired tests, because they answer different questions:
1. Paired t-test on matched per-fold CV accuracies (same 5 folds for every model): is the typical
   gap across resamples of the training data distinguishable from zero?
2. McNemar's test on the held-out test set (same examples for every model): among examples where
   the two models disagree, does one win more often than chance?

Also reported: Wilson 95% intervals for test accuracy, and a calibration / error profile
(how confident each model is when it is wrong) plus how many models miss each test example.
Implemented with scipy only (no statsmodels dependency).
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, log_loss
from config import RESULTS

MODELS = {  # file prefix -> display name
    "baseline": "Baseline (TF-IDF + LogReg)",
    "model2": "TF-IDF + SVD + GradBoost",
    "mlp": "TF-IDF + MLP (TensorFlow)",
    "bert": "Frozen BiomedBERT + LogReg",
}
REFERENCE = "baseline"
ALPHA = 0.05


def mean_ci(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    half = stats.sem(values) * stats.t.ppf((1 + confidence) / 2, n - 1)
    return values.mean(), half


def wilson_ci(k, n, confidence=0.95):
    """Wilson score interval for a proportion k/n."""
    z = stats.norm.ppf((1 + confidence) / 2)
    p = k / n
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def holm_adjust(pvals):
    """Holm step-down adjusted p-values (controls family-wise error rate)."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def mcnemar_test(b, c, exact_threshold=25):
    """McNemar's test. b = reference right & challenger wrong, c = challenger right & reference wrong.
    Exact binomial when b + c < exact_threshold, else chi-square with continuity correction."""
    n = b + c
    if n == 0:
        return 1.0
    if n < exact_threshold:
        return stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return float(1 - stats.chi2.cdf((abs(b - c) - 1) ** 2 / n, df=1))


def main():
    cv, test = {}, {}
    for key in MODELS:
        cv_path, test_path = RESULTS / f"{key}_cv_scores.csv", RESULTS / f"{key}_test_predictions.csv"
        if cv_path.exists() and test_path.exists():
            cv[key], test[key] = pd.read_csv(cv_path), pd.read_csv(test_path)
        else:
            print(f"(skipping {key}: result files not found)")
    if REFERENCE not in cv:
        raise SystemExit("baseline results are required (run 02_baseline_model.py)")
    challengers = [k for k in cv if k != REFERENCE]

    # Paired comparisons are only valid if every model saw the identical test set
    ref = test[REFERENCE]
    for key in challengers:
        assert np.array_equal(ref["y_true"].to_numpy(), test[key]["y_true"].to_numpy()) \
            and ref["text"].equals(test[key]["text"]), f"{key} was evaluated on a different test set"
    n_test = len(ref)
    y_true = ref["y_true"].to_numpy()

    # ---- 1. Cross-validation summary ----
    print("=" * 78)
    print("CROSS-VALIDATION (5-fold on the training set): mean [95% t-interval]")
    print("=" * 78)
    for key in cv:
        acc_m, acc_h = mean_ci(cv[key]["accuracy"])
        print(f"{MODELS[key]:30s} acc {acc_m:.3f} [{acc_m - acc_h:.3f}, {acc_m + acc_h:.3f}]   "
              f"f1 {cv[key]['f1'].mean():.3f}   auc {cv[key]['roc_auc'].mean():.3f}")

    # ---- 2. Held-out test summary ----
    print()
    print("=" * 78)
    print(f"HELD-OUT TEST SET (n={n_test}): accuracy with Wilson 95% interval")
    print("=" * 78)
    pred = {k: test[k]["y_pred"].to_numpy() for k in test}
    proba = {k: test[k]["y_proba"].to_numpy() for k in test}
    correct = {k: pred[k] == y_true for k in test}
    for key in test:
        k_ok = int(correct[key].sum())
        lo, hi = wilson_ci(k_ok, n_test)
        print(f"{MODELS[key]:30s} acc {k_ok / n_test:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"f1 {f1_score(y_true, pred[key]):.3f}   auc {roc_auc_score(y_true, proba[key]):.3f}   "
              f"errors {n_test - k_ok}")

    # ---- 3. Paired t-test on CV accuracy, challenger vs baseline ----
    rows = []
    for key in challengers:
        a, b = cv[key]["accuracy"].to_numpy(), cv[REFERENCE]["accuracy"].to_numpy()
        diff = a - b
        t_stat, p = stats.ttest_rel(a, b)
        d_mean, d_half = mean_ci(diff)
        rows.append((key, d_mean, d_mean - d_half, d_mean + d_half, int((diff > 0).sum()), t_stat, p))
    p_adj = holm_adjust([r[-1] for r in rows])
    print()
    print("=" * 78)
    print(f"PAIRED T-TEST on CV accuracy, each model vs. {REFERENCE} (difference = model - baseline)")
    print("=" * 78)
    print(f"{'model':30s} {'mean diff':>10s} {'95% CI of diff':>20s} {'folds >':>8s} {'p':>8s} {'Holm p':>8s}")
    for (key, d, lo, hi, wins, t_stat, p), pa in zip(rows, p_adj):
        print(f"{MODELS[key]:30s} {d * 100:+9.2f}p {f'[{lo * 100:+.2f}, {hi * 100:+.2f}]':>20s} "
              f"{wins:>6d}/5 {p:8.4f} {pa:8.4f}")

    # ---- 4. McNemar on the held-out test set, challenger vs baseline ----
    rows = []
    for key in challengers:
        b = int((correct[REFERENCE] & ~correct[key]).sum())   # baseline right, challenger wrong
        c = int((~correct[REFERENCE] & correct[key]).sum())   # challenger right, baseline wrong
        rows.append((key, b, c, mcnemar_test(b, c)))
    p_adj = holm_adjust([r[-1] for r in rows])
    print()
    print("=" * 78)
    print(f"MCNEMAR'S TEST on the test set, each model vs. {REFERENCE}")
    print("=" * 78)
    print(f"{'model':30s} {'only baseline right':>20s} {'only model right':>17s} {'p':>8s} {'Holm p':>8s}")
    for (key, b, c, p), pa in zip(rows, p_adj):
        print(f"{MODELS[key]:30s} {b:>20d} {c:>17d} {p:8.4f} {pa:8.4f}")

    # ---- 5. Calibration / error profile ----
    print()
    print("=" * 78)
    print("ERROR PROFILE (confidence = max(p, 1-p); 'confident' = confidence >= 0.9)")
    print("=" * 78)
    print(f"{'model':30s} {'errors':>6s} {'conf. on errors':>16s} {'confident errors':>17s} {'Brier':>7s} {'log loss':>9s}")
    for key in test:
        conf = np.maximum(proba[key], 1 - proba[key])
        wrong = ~correct[key]
        print(f"{MODELS[key]:30s} {int(wrong.sum()):>6d} {conf[wrong].mean():>16.3f} "
              f"{int((conf[wrong] >= 0.9).sum()):>10d}/{int(wrong.sum()):<5d}"
              f"{brier_score_loss(y_true, proba[key]):>7.3f} {log_loss(y_true, proba[key]):>9.3f}")

    # ---- 6. How many models miss each example ----
    n_wrong = sum((~correct[k]).astype(int) for k in test)
    print()
    print("=" * 78)
    print(f"HOW MANY OF THE {len(test)} MODELS MISCLASSIFY EACH TEST EXAMPLE")
    print("=" * 78)
    for m in range(len(test) + 1):
        print(f"misclassified by exactly {m}: {int((n_wrong == m).sum())}")
    print(f"\nTest examples all {len(test)} models misclassify: {int((n_wrong == len(test)).sum())} "
          f"(row positions saved to results/missed_by_all_models.csv)")
    # Row positions only (no abstract text), so the file is safe to commit; look the rows up in
    # any *_test_predictions.csv (git-ignored) to read them
    missed = pd.DataFrame({"test_row": np.flatnonzero(n_wrong == len(test)),
                           "y_true": y_true[n_wrong == len(test)]})
    missed.to_csv(RESULTS / "missed_by_all_models.csv", index=False)


if __name__ == "__main__":
    main()
