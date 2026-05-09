"""
Step 2 — Evaluate fine-tuned models on the held-out test set.

Requires step1_finetune.py to have been run first (needs the saved checkpoints
and the test_split.csv file).

Outputs saved to results/:
  results/test_predictions_{model}.csv   ← per-sample predictions + probabilities
  results/metrics_summary.csv            ← F1, Precision, Recall, Accuracy, FPR, AUC
  results/confusion_matrix_{model}.png   ← confusion matrix heatmap
  results/roc_curves.png                 ← ROC curves for all three models
  results/metrics_comparison.png         ← bar chart comparing models head-to-head

Usage:
  python step2_evaluate.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works without a display
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_auc_score, roc_curve,
    classification_report,
)
from torch.optim import AdamW
from transformers import (
    BertTokenizer,         BertForSequenceClassification,
    RobertaTokenizer,      RobertaForSequenceClassification,
    DistilBertTokenizer,   DistilBertForSequenceClassification,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — must match step1_finetune.py
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR  = "results"
MODELS_DIR   = "models"
MAX_LEN      = 128
BATCH_SIZE   = 128  # evaluation only — no gradients stored, can use large batch
N_BOOTSTRAP  = 1000  # number of bootstrap samples for confidence intervals

MODELS = {
    "bert":       "bert-base-uncased",
    "roberta":    "roberta-base",
    "distilbert": "distilbert-base-uncased",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
test_path = f"{RESULTS_DIR}/test_split.csv"
if not os.path.exists(test_path):
    raise FileNotFoundError(
        f"{test_path} not found. Run step1_finetune.py first to generate the test split."
    )

test_df = pd.read_csv(test_path)
print(f"Test set: {len(test_df)} samples  →  {test_df['label'].sum()} FLIP  |  {(test_df['label']==0).sum()} HOLD")

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET CLASS  (same as step1 — needed to feed data to the model)
# ═══════════════════════════════════════════════════════════════════════════════
class RebuttaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts     = df["rebuttal_text"].tolist()
        self.labels    = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  BOOTSTRAP CONFIDENCE INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════
def bootstrap_ci(y_true, y_pred, metric_fn, n=N_BOOTSTRAP, ci=0.95):
    """
    Estimate a 95% confidence interval for a metric using bootstrap resampling.

    How bootstrap works:
      1. Randomly sample (with replacement) from the test set n times.
      2. Compute the metric on each sample.
      3. The 2.5th and 97.5th percentiles of those n scores = the 95% CI.

    This tells you: "if we had slightly different test examples, how much
    would the metric change?" — a measure of result reliability.
    """
    scores = []
    rng = np.random.default_rng(seed=42)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)

    for _ in range(n):
        idx = rng.integers(0, n_samples, size=n_samples)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return lo, hi

# ═══════════════════════════════════════════════════════════════════════════════
#  RUN EVALUATION FOR ONE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_model(model_key: str, model_name: str, test_df: pd.DataFrame) -> dict:
    print(f"\n{'═' * 60}")
    print(f"  Evaluating: {model_key.upper()}  ({model_name})")
    print(f"{'═' * 60}")

    # ── Load tokenizer + model architecture ───────────────────────────────────
    if model_key == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model     = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    elif model_key == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model     = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model     = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # ── Load the fine-tuned weights saved by step1 ────────────────────────────
    ckpt_path = f"{MODELS_DIR}/{model_key}_best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found. Run step1_finetune.py first.")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()   # disable dropout — important for deterministic predictions

    # ── DataLoader ────────────────────────────────────────────────────────────
    dataset = RebuttaDataset(test_df, tokenizer, MAX_LEN)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds   = []
    all_probs   = []   # probability of FLIP (class 1) — needed for ROC-AUC
    all_true    = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs  = model(input_ids=input_ids, attention_mask=attention_mask)
            probs    = torch.softmax(outputs.logits, dim=1)  # convert logits → probabilities
            preds    = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())      # P(FLIP)
            all_true.extend(labels.cpu().numpy())

    y_true = np.array(all_true)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    # ── Compute metrics ───────────────────────────────────────────────────────
    macro_f1  = f1_score(y_true, y_pred, average="macro")
    flip_f1   = f1_score(y_true, y_pred, average=None)[1]         # F1 for FLIP class only
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy  = accuracy_score(y_true, y_pred)
    auc       = roc_auc_score(y_true, y_prob)

    # False Positive Rate: fraction of actual HOLDs incorrectly predicted as FLIP
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # 95% confidence intervals via bootstrap
    f1_lo,  f1_hi  = bootstrap_ci(y_true, y_pred, lambda a, b: f1_score(a, b, average="macro"))
    acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, accuracy_score)

    print(f"  Macro F1   : {macro_f1:.4f}  (95% CI: [{f1_lo:.4f}, {f1_hi:.4f}])")
    print(f"  FLIP F1    : {flip_f1:.4f}")
    print(f"  Precision  : {precision:.4f}")
    print(f"  Recall     : {recall:.4f}")
    print(f"  Accuracy   : {accuracy:.4f}  (95% CI: [{acc_lo:.4f}, {acc_hi:.4f}])")
    print(f"  FPR        : {fpr_val:.4f}")
    print(f"  ROC-AUC    : {auc:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["HOLD", "FLIP"]))

    # ── Save per-sample predictions ───────────────────────────────────────────
    preds_df = test_df.copy()
    preds_df["predicted_label"] = y_pred
    preds_df["prob_flip"]       = y_prob
    preds_df["correct"]         = (y_pred == y_true).astype(int)
    preds_df.to_csv(f"{RESULTS_DIR}/test_predictions_{model_key}.csv", index=False)

    return {
        "model":         model_key,
        "macro_f1":      round(macro_f1, 4),
        "flip_f1":       round(flip_f1, 4),
        "precision":     round(precision, 4),
        "recall":        round(recall, 4),
        "accuracy":      round(accuracy, 4),
        "fpr":           round(fpr_val, 4),
        "roc_auc":       round(auc, 4),
        "f1_ci_lo":      round(f1_lo, 4),
        "f1_ci_hi":      round(f1_hi, 4),
        "acc_ci_lo":     round(acc_lo, 4),
        "acc_ci_hi":     round(acc_hi, 4),
        "confusion_matrix": cm.tolist(),
        "y_true":        y_true,
        "y_prob":        y_prob,
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm: list, model_key: str) -> None:
    """Save a confusion matrix heatmap for one model."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_arr, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["HOLD (pred)", "FLIP (pred)"],
        yticklabels=["HOLD (true)", "FLIP (true)"],
    )
    ax.set_title(f"Confusion Matrix — {model_key.upper()}", fontsize=13)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()
    path = f"{RESULTS_DIR}/confusion_matrix_{model_key}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(results: list[dict]) -> None:
    """Save a single figure with ROC curves for all three models overlaid."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"bert": "#e74c3c", "roberta": "#2ecc71", "distilbert": "#3498db"}

    for r in results:
        fpr_arr, tpr_arr, _ = roc_curve(r["y_true"], r["y_prob"])
        ax.plot(
            fpr_arr, tpr_arr,
            label=f"{r['model'].upper()}  (AUC = {r['roc_auc']:.3f})",
            color=colors[r["model"]], linewidth=2,
        )

    # Diagonal = random baseline
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = f"{RESULTS_DIR}/roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_metrics_comparison(summary_df: pd.DataFrame) -> None:
    """Bar chart comparing key metrics across all three models."""
    metrics_to_plot = ["macro_f1", "flip_f1", "roc_auc", "accuracy"]
    metric_labels   = ["Macro F1", "FLIP F1", "ROC-AUC", "Accuracy"]
    colors          = ["#e74c3c", "#2ecc71", "#3498db"]
    model_names     = summary_df["model"].str.upper().tolist()

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 5))

    for ax, metric, label in zip(axes, metrics_to_plot, metric_labels):
        values = summary_df[metric].tolist()
        bars   = ax.bar(model_names, values, color=colors, width=0.5)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=10,
            )

        ax.set_title(label, fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", labelsize=9)

    plt.suptitle("Model Comparison on Sycophancy Detection", fontsize=14, y=1.02)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/metrics_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    all_results = []

    for model_key, model_name in MODELS.items():
        result = evaluate_model(model_key, model_name, test_df)
        all_results.append(result)

    # ── Save metrics summary CSV ───────────────────────────────────────────────
    summary_cols = [
        "model", "macro_f1", "flip_f1", "precision", "recall",
        "accuracy", "fpr", "roc_auc", "f1_ci_lo", "f1_ci_hi",
        "acc_ci_lo", "acc_ci_hi",
    ]
    summary_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    summary_path = f"{RESULTS_DIR}/metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nMetrics summary saved to {summary_path}")
    print(summary_df.to_string(index=False))

    # ── Generate all plots ─────────────────────────────────────────────────────
    print("\nGenerating plots...")
    for r in all_results:
        plot_confusion_matrix(r["confusion_matrix"], r["model"])
    plot_roc_curves(all_results)
    plot_metrics_comparison(summary_df)

    print(f"\n{'═' * 60}")
    print("Evaluation complete.")
    print(f"All outputs in: {RESULTS_DIR}/")
    print("Ready for Step 3: Integrated Gradients")
    print(f"{'═' * 60}")
