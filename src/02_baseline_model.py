"""
Step 3: Baseline model with cross-validation and confidence intervals.

Model: TF-IDF (title + abstract) -> Logistic Regression
Data:  pubmed.db (table `abstracts`, produced by 01_clean_and_load.py)

Methodology:
- Hold out a stratified 20% test set up front and DO NOT touch it until
  the final check at the bottom. All model development uses only the
  training set.
- On the training set, run stratified 5-fold CV to get a distribution of
  scores per metric (accuracy, F1, ROC-AUC), then report mean +/- 95% CI
  using a t-distribution over the 5 fold scores. This is the number to
  quote as "baseline performance" and to beat with a second model later.
- Refit on the full training set and report a single confirmatory score
  on the held-out test set.
"""

import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from config import DB_PATH, RESULTS, MODELS

RANDOM_STATE = 42
N_FOLDS = 5

def mean_ci(scores, confidence=0.95):
    """Mean and 95% CI half-width using a t-distribution over fold scores."""
    scores = np.asarray(scores)
    n = len(scores)
    mean = scores.mean()
    sem = stats.sem(scores)
    half_width = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, half_width

def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT title, abstract, label, label_binary FROM abstracts", conn)
    conn.close()

    # Combine title + abstract as the text feature (title carries signal too,
    # and costs nothing to include)
    df["text"] = df["title"].fillna("") + " " + df["abstract"]
    X = df["text"]
    y = df["label_binary"]  # 1 = category_b, 0 = category_a

    # Held-out test set, untouched until the end
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            min_df=2,
            stop_words="english",
        )),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    # --- Cross-validation on training set only ---
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    print("=" * 60)
    print(f"BASELINE MODEL: TF-IDF + Logistic Regression")
    print(f"{N_FOLDS}-fold stratified CV on training set (n={len(X_train)})")
    print("=" * 60)
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        mean, half_width = mean_ci(scores)
        fold_str = ", ".join(f"{s:.3f}" for s in scores)
        print(f"{metric:10s}: folds = [{fold_str}]")
        print(f"{'':10s}  mean = {mean:.3f}  |  95% CI = [{mean - half_width:.3f}, {mean + half_width:.3f}]")
    print()

    # Save per-fold CV scores so a later comparison script (e.g. against a
    # second model) can run paired tests without retraining this model
    cv_scores_df = pd.DataFrame({
        "fold": range(1, N_FOLDS + 1),
        **{metric: cv_results[f"test_{metric}"] for metric in scoring},
    })
    cv_scores_df.to_csv(RESULTS / "baseline_cv_scores.csv", index=False)
    print("Saved per-fold CV scores -> baseline_cv_scores.csv")
    print()

    # --- Refit on full training set, confirm on held-out test set ---
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print(f"HELD-OUT TEST SET (n={len(X_test)}, never used above)")
    print("=" * 60)
    print(f"accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"f1:        {f1_score(y_test, y_pred):.3f}")
    print(f"roc_auc:   {roc_auc_score(y_test, y_proba):.3f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["category_a", "category_b"]))

    # Save fitted pipeline + splits for reuse when building/comparing model 2
    import joblib
    joblib.dump(pipeline, MODELS / "baseline_pipeline.joblib")
    pd.DataFrame({"text": X_test, "y_true": y_test, "y_pred": y_pred, "y_proba": y_proba}) \
        .to_csv(RESULTS / "baseline_test_predictions.csv", index=False)
    print("Saved fitted pipeline -> baseline_pipeline.joblib")
    print("Saved test predictions -> baseline_test_predictions.csv")

if __name__ == "__main__":
    main()
