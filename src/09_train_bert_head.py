"""
Step 9: Classifier on frozen BiomedBERT embeddings.

Model 4: frozen BiomedBERT embedding (768-d) -> StandardScaler -> Logistic Regression

Same protocol as the other models:
- same 80/20 stratified split (same random_state) -> same held-out test set
- same 5-fold stratified CV assignment on the training set -> paired fold scores
- default hyperparameters, nothing tuned (same policy as the baseline)

Requires data/processed/bert_embeddings.npz from 08_extract_bert_embeddings.py.
Embeddings are matched to abstracts by PMID, so row order cannot silently misalign them.

Outputs: results/bert_cv_scores.csv, results/bert_test_predictions.csv, models/bert_head.joblib
"""

import sqlite3
import numpy as np
import pandas as pd
import joblib
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from config import DB_PATH, RESULTS, MODELS, RANDOM_STATE, EMBEDDINGS_PATH

N_FOLDS = 5


def mean_ci(scores, confidence=0.95):
    scores = np.asarray(scores)
    n = len(scores)
    mean = scores.mean()
    half_width = stats.sem(scores) * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, half_width


def main():
    # Same query shape/row order as the other scripts, plus pmid for alignment
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT pmid, title, abstract, label_binary FROM abstracts", conn)
    conn.close()
    assert df["pmid"].is_unique, "duplicate PMIDs in the database"
    df["text"] = df["title"].fillna("") + " " + df["abstract"]

    emb = np.load(EMBEDDINGS_PATH)
    row_of_pmid = pd.Series(np.arange(len(emb["pmids"])), index=emb["pmids"])
    missing = set(df["pmid"]) - set(row_of_pmid.index)
    if missing:
        raise ValueError(f"{len(missing)} abstracts have no embedding; rerun 08_extract_bert_embeddings.py")
    X = emb["embeddings"][row_of_pmid.loc[df["pmid"]].to_numpy()]
    y = df["label_binary"]
    print(f"Embeddings: {X.shape} from {emb['model_name']} (max_length={int(emb['max_length'])}, "
          f"pooling={emb['pooling']})")

    # train_test_split depends only on n and the stratify labels, so this selects exactly the
    # same rows as the text-based scripts
    X_train, X_test, text_train, text_test, y_train, y_test = train_test_split(
        X, df["text"], y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Sanity check: confirm the test set matches the baseline's (when its predictions exist)
    base_path = RESULTS / "baseline_test_predictions.csv"
    if base_path.exists():
        base = pd.read_csv(base_path)
        assert np.array_equal(base["y_true"].to_numpy(), y_test.to_numpy()), \
            "test set differs from the baseline's; comparisons would not be paired"
        print("Verified: test set is identical to the baseline's.")

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"accuracy": "accuracy", "f1": "f1", "roc_auc": "roc_auc"}
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    print("=" * 60)
    print("MODEL 4: frozen BiomedBERT embeddings -> Logistic Regression")
    print(f"{N_FOLDS}-fold stratified CV on training set (n={len(X_train)})")
    print("=" * 60)
    for metric in scoring:
        scores = cv_results[f"test_{metric}"]
        mean, hw = mean_ci(scores)
        print(f"{metric:10s}: folds = [{', '.join(f'{s:.3f}' for s in scores)}]")
        print(f"{'':10s}  mean = {mean:.3f}  |  95% CI = [{mean - hw:.3f}, {mean + hw:.3f}]")
    print()

    pd.DataFrame({
        "fold": range(1, N_FOLDS + 1),
        **{m: cv_results[f"test_{m}"] for m in scoring},
    }).to_csv(RESULTS / "bert_cv_scores.csv", index=False)
    print(f"Saved per-fold CV scores -> {RESULTS / 'bert_cv_scores.csv'}\n")

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

    joblib.dump(pipeline, MODELS / "bert_head.joblib")
    pd.DataFrame({"text": text_test.values, "y_true": y_test.values, "y_pred": y_pred, "y_proba": y_proba}) \
        .to_csv(RESULTS / "bert_test_predictions.csv", index=False)
    print(f"Saved test predictions -> {RESULTS / 'bert_test_predictions.csv'}")


if __name__ == "__main__":
    main()
