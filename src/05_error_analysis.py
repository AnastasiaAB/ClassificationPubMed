"""
Step 5: Error analysis for both models, and where their errors overlap.

Input:  model2_test_predictions.csv, baseline_test_predictions.csv
Output: baseline_error_analysis.png, model2_error_analysis.png   (confusion matrix + probability distribution)
        error_overlap_analysis.png (where the two models agree/disagree)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from config import RESULTS, FIGURES

BASELINE_CSV = RESULTS / "baseline_test_predictions.csv"
MODEL2_CSV = RESULTS / "model2_test_predictions.csv"

def plot_single_model_errors(df, model_name, output_path):
    y_true, y_pred, y_proba = df["y_true"], df["y_pred"], df["y_proba"]
    correct = y_true == y_pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    cm = confusion_matrix(y_true, y_pred)
    labels = ["category_a\n(cancer)", "category_b\n(cardio)"]
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, color=color)
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {model_name} (n={len(df)})")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    ax2.hist(y_proba[correct], bins=30, alpha=0.6, label="Correct", color="#2c7fb8")
    ax2.hist(y_proba[~correct], bins=30, alpha=0.8, label="Misclassified", color="#d95f02")
    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    ax2.set_xlabel("Predicted P(category_b)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Predicted Probability Distribution — {model_name}")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {output_path}")

    n_errors = (~correct).sum()
    print(f"{model_name} misclassified: {n_errors} of {len(df)} ({n_errors/len(df)*100:.1f}%)")
    print(f"  true=category_b, pred=category_a: {((y_true==1)&(y_pred==0)).sum()}")
    print(f"  true=category_a, pred=category_b: {((y_true==0)&(y_pred==1)).sum()}")
    print()

def plot_error_overlap(base_df, m2_df, output_path):
    base_correct = (base_df["y_true"] == base_df["y_pred"]).values
    m2_correct = (m2_df["y_true"] == m2_df["y_pred"]).values

    both_correct = np.sum(base_correct & m2_correct)
    base_only_wrong = np.sum(base_correct & ~m2_correct)   # baseline right, model2 wrong
    m2_only_wrong = np.sum(~base_correct & m2_correct)     # model2 right, baseline wrong
    both_wrong = np.sum(~base_correct & ~m2_correct)

    categories = ["Both correct", "Only model2\nwrong", "Only baseline\nwrong", "Both wrong"]
    counts = [both_correct, base_only_wrong, m2_only_wrong, both_wrong]
    colors = ["#2c7fb8", "#d95f02", "#7570b3", "#e7298a"]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(categories, counts, color=colors)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                 str(count), ha="center", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_title(f"Prediction Agreement — Held-out Test Set (n={len(base_df)})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved -> {output_path}")
    print(f"Both correct: {both_correct} | Only model2 wrong: {base_only_wrong} | "
          f"Only baseline wrong: {m2_only_wrong} | Both wrong: {both_wrong}")

def main():
    base_df = pd.read_csv(BASELINE_CSV)
    m2_df = pd.read_csv(MODEL2_CSV)

    plot_single_model_errors(base_df, "Baseline (TF-IDF + LogReg)", FIGURES / "baseline_error_analysis.png")
    plot_single_model_errors(m2_df, "Model 2 (SVD + Gradient Boosting)", FIGURES / "model2_error_analysis.png")
    plot_error_overlap(base_df, m2_df, FIGURES / "error_overlap_analysis.png")

if __name__ == "__main__":
    main()
