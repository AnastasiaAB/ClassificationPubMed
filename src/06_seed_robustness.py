"""
Step 6: How sensitive is the baseline-vs-model2 CV comparison to the random seed?

The seed controls the train/test split, the CV fold assignment, and the models'
internal randomness. For each seed we rerun the full 5-fold paired comparison and
record the mean accuracy difference (model2 - baseline) and the paired t-test p-value.

Why this matters: with only 5 folds the p-value is itself noisy, so a single seed's
"significant / not significant" verdict shouldn't be treated as a fixed fact. The
direction and rough size of the effect is the more robust thing to report.

Output: results/seed_robustness.csv
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from config import DB_PATH, RESULTS

SEEDS = [42, 0, 7, 123, 999]
N_FOLDS = 5
ALPHA = 0.05

def tfidf():
    return TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")

def build_baseline(seed):
    return Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])

def build_model2(seed):
    return Pipeline([
        ("tfidf", tfidf()),
        ("svd", TruncatedSVD(n_components=200, random_state=seed)),
        ("clf", HistGradientBoostingClassifier(random_state=seed)),
    ])

def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT title, abstract, label_binary FROM abstracts", conn)
    conn.close()
    df["text"] = df["title"].fillna("") + " " + df["abstract"]
    X, y = df["text"], df["label_binary"]

    rows = []
    for seed in SEEDS:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        X_tr, y_tr = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        base_scores, m2_scores = [], []
        for tr_idx, val_idx in cv.split(X_tr, y_tr):
            base = build_baseline(seed).fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            m2 = build_model2(seed).fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            base_scores.append(accuracy_score(y_tr.iloc[val_idx], base.predict(X_tr.iloc[val_idx])))
            m2_scores.append(accuracy_score(y_tr.iloc[val_idx], m2.predict(X_tr.iloc[val_idx])))

        diffs = np.array(m2_scores) - np.array(base_scores)
        _, p_val = stats.ttest_rel(m2_scores, base_scores)
        rows.append({
            "seed": seed,
            "baseline_cv_acc": np.mean(base_scores),
            "model2_cv_acc": np.mean(m2_scores),
            "mean_diff": diffs.mean(),
            "p_value": p_val,
            "significant": p_val < ALPHA,
        })
        print(f"seed={seed:4d}  mean_diff={diffs.mean():+.4f}  p={p_val:.4f}  "
              f"{'significant' if p_val < ALPHA else 'not significant'}")

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "seed_robustness.csv", index=False)
    print(f"\nModel2 worse on average in {(out['mean_diff'] < 0).sum()}/{len(out)} seeds; "
          f"significant at alpha={ALPHA} in {out['significant'].sum()}/{len(out)}")
    print(f"Saved -> {RESULTS / 'seed_robustness.csv'}")

if __name__ == "__main__":
    main()
