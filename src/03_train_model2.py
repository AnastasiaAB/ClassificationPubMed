"""
Step 4a: Train the second model.

Model 2: TF-IDF -> TruncatedSVD (dense LSA embedding) -> HistGradientBoostingClassifier

A deliberately different modeling approach from the baseline (sparse
bag-of-words + linear model): this compresses the TF-IDF space into a
dense, lower-dimensional semantic embedding and fits a nonlinear,
tree-based ensemble on top. No internet access is available in this
environment, so pretrained embeddings (BERT etc.) aren't an option --
this is the most meaningfully "harder" comparison achievable with
purely local, scikit-learn-native tools.

Uses the SAME train/test split and SAME CV fold assignments as
02_baseline_model.py (same random_state) so results are directly comparable
in 04_compare_models.py.

Outputs:
- model2_pipeline.joblib        (fitted pipeline, refit on full training set)
- model2_test_predictions.csv   (predictions on the held-out test set)
- model2_cv_scores.csv          (per-fold CV scores, matched to baseline's folds)
"""

import sqlite3
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from config import DB_PATH, RESULTS, MODELS

RANDOM_STATE = 42
N_FOLDS = 5

def mean_ci(scores, confidence=0.95):
    scores = np.asarray(scores)
    n = len(scores)
    mean = scores.mean()
    sem = stats.sem(scores)
    half_width = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, half_width

def build_model2_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")),
        ("svd", TruncatedSVD(n_components=200, random_state=RANDOM_STATE)),
        ("clf", HistGradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])

def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT title, abstract, label_binary FROM abstracts", conn)
    conn.close()

    df["text"] = df["title"].fillna("") + " " + df["abstract"]
    X = df["text"]
    y = df["label_binary"]

    # Same split as 02_baseline_model.py (same random_state) -> same test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    pipeline = build_model2_pipeline()

    # --- Cross-validation on training set only (same fold assignments as baseline) ---
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    print("=" * 60)
    print("MODEL 2: TF-IDF -> TruncatedSVD -> HistGradientBoostingClassifier")
    print(f"{N_FOLDS}-fold stratified CV on training set (n={len(X_train)})")
    print("=" * 60)
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        mean, half_width = mean_ci(scores)
        fold_str = ", ".join(f"{s:.3f}" for s in scores)
        print(f"{metric:10s}: folds = [{fold_str}]")
        print(f"{'':10s}  mean = {mean:.3f}  |  95% CI = [{mean - half_width:.3f}, {mean + half_width:.3f}]")
    print()

    cv_scores_df = pd.DataFrame({
        "fold": range(1, N_FOLDS + 1),
        **{metric: cv_results[f"test_{metric}"] for metric in scoring},
    })
    cv_scores_df.to_csv(RESULTS / "model2_cv_scores.csv", index=False)
    print("Saved per-fold CV scores -> model2_cv_scores.csv")
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

    joblib.dump(pipeline, MODELS / "model2_pipeline.joblib")
    pd.DataFrame({"text": X_test, "y_true": y_test, "y_pred": y_pred, "y_proba": y_proba}) \
        .to_csv(RESULTS / "model2_test_predictions.csv", index=False)
    print("Saved fitted pipeline -> model2_pipeline.joblib")
    print("Saved test predictions -> model2_test_predictions.csv")

if __name__ == "__main__":
    main()
