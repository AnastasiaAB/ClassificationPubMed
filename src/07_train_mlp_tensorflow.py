"""
Step 7: Neural network (TensorFlow / Keras) on TF-IDF features.

Model 3: TF-IDF (same vectorizer settings as the baseline) -> Dense(256, ReLU)
         -> Dropout(0.5) -> Dense(1, sigmoid)

A different model family from the linear baseline (nonlinear hidden layer, trained
by gradient descent) while keeping the input representation identical, so any
difference comes from the classifier rather than the features.

Same protocol as 02_baseline_model.py and 03_train_model2.py:
- same 80/20 stratified split (same random_state) -> same held-out test set
- same 5-fold stratified CV assignment on the training set -> paired fold scores
- TF-IDF is fit inside each fold on that fold's training text only
- early stopping uses a stratified validation slice carved out of each training
  fold, NEVER the CV validation fold or the test set

Outputs (same formats as the other models, so the comparison scripts can load them):
- results/mlp_cv_scores.csv
- results/mlp_test_predictions.csv
- models/mlp_model.keras, models/mlp_tfidf.joblib
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow's startup logging

import sqlite3
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from config import DB_PATH, RESULTS, MODELS, RANDOM_STATE

N_FOLDS = 5
HIDDEN_UNITS = 256
DROPOUT = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 30
PATIENCE = 3          # stop after this many epochs without validation-loss improvement
VAL_FRACTION = 0.1    # share of each training set held out for early stopping

# Make runs reproducible (seeds Python, NumPy and TensorFlow; forces deterministic ops)
tf.keras.utils.set_random_seed(RANDOM_STATE)
tf.config.experimental.enable_op_determinism()


def mean_ci(scores, confidence=0.95):
    scores = np.asarray(scores)
    n = len(scores)
    mean = scores.mean()
    half_width = stats.sem(scores) * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, half_width


def make_vectorizer():
    return TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2, stop_words="english")


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(HIDDEN_UNITS, activation="relu"),
        tf.keras.layers.Dropout(DROPOUT),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fit_model(train_text, y_train):
    """Fit TF-IDF + MLP on the given training data only. Returns (vectorizer, model, epochs_run)."""
    tf.keras.utils.set_random_seed(RANDOM_STATE)  # same starting point for every fit
    vec = make_vectorizer()
    X = vec.fit_transform(train_text).toarray().astype("float32")
    y = np.asarray(y_train).astype("float32")

    # Inner stratified split used only for early stopping
    fit_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=VAL_FRACTION, stratify=y, random_state=RANDOM_STATE
    )
    model = build_model(X.shape[1])
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )
    history = model.fit(
        X[fit_idx], y[fit_idx],
        validation_data=(X[val_idx], y[val_idx]),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=0,
    )
    return vec, model, len(history.history["loss"])


def predict_proba(vec, model, text):
    X = vec.transform(text).toarray().astype("float32")
    return model.predict(X, verbose=0).ravel()


def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT title, abstract, label_binary FROM abstracts", conn)
    conn.close()
    df["text"] = df["title"].fillna("") + " " + df["abstract"]
    X, y = df["text"], df["label_binary"]

    # Same split as the other models -> same held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    # --- 5-fold CV on the training set (same fold assignment as the other models) ---
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    print("=" * 60)
    print("MODEL 3: TF-IDF -> MLP (TensorFlow/Keras)")
    print(f"{N_FOLDS}-fold stratified CV on training set (n={len(X_train)})")
    print("=" * 60)
    for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        vec, model, epochs_run = fit_model(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        proba = predict_proba(vec, model, X_train.iloc[val_idx])
        pred = (proba >= 0.5).astype(int)
        y_val = y_train.iloc[val_idx]
        row = {
            "fold": fold,
            "accuracy": accuracy_score(y_val, pred),
            "f1": f1_score(y_val, pred),
            "roc_auc": roc_auc_score(y_val, proba),
        }
        fold_rows.append(row)
        print(f"fold {fold}: acc={row['accuracy']:.3f}  f1={row['f1']:.3f}  "
              f"auc={row['roc_auc']:.3f}  (stopped after {epochs_run} epochs)")
        tf.keras.backend.clear_session()

    cv_df = pd.DataFrame(fold_rows)
    print()
    for metric in ["accuracy", "f1", "roc_auc"]:
        mean, hw = mean_ci(cv_df[metric])
        print(f"{metric:9s} mean = {mean:.3f}  |  95% CI = [{mean - hw:.3f}, {mean + hw:.3f}]")
    cv_df.to_csv(RESULTS / "mlp_cv_scores.csv", index=False)
    print(f"\nSaved per-fold CV scores -> {RESULTS / 'mlp_cv_scores.csv'}\n")

    # --- Refit on the full training set, confirm on the held-out test set ---
    vec, model, epochs_run = fit_model(X_train, y_train)
    proba = predict_proba(vec, model, X_test)
    pred = (proba >= 0.5).astype(int)

    print("=" * 60)
    print(f"HELD-OUT TEST SET (n={len(X_test)}, never used above)  [stopped after {epochs_run} epochs]")
    print("=" * 60)
    print(f"accuracy:  {accuracy_score(y_test, pred):.3f}")
    print(f"f1:        {f1_score(y_test, pred):.3f}")
    print(f"roc_auc:   {roc_auc_score(y_test, proba):.3f}")
    print()
    print(classification_report(y_test, pred, target_names=["category_a", "category_b"]))

    model.save(MODELS / "mlp_model.keras")
    joblib.dump(vec, MODELS / "mlp_tfidf.joblib")
    pd.DataFrame({"text": X_test.values, "y_true": y_test.values, "y_pred": pred, "y_proba": proba}) \
        .to_csv(RESULTS / "mlp_test_predictions.csv", index=False)
    print(f"Saved model -> {MODELS / 'mlp_model.keras'}")
    print(f"Saved test predictions -> {RESULTS / 'mlp_test_predictions.csv'}")


if __name__ == "__main__":
    main()
