"""
Sycophancy Flip Prediction — 5-Fold CV Training Pipeline (MCQ, no IG)
======================================================================
Same as 5fold-train-noIG.py but adapted for MCQ challenge result JSON files:
  - Loads one or more MCQ challenge JSON files (concatenates their "challenges" lists)
  - Input is tokenized as a pair: (mcq_context, rebuttal_text)
    where mcq_context = "Q: {question}\nA) ...\nB) ...\nC) ...\nD) ..."
  - question_id derived from MD5 hash of question text
  - Runs 5-fold cross-validation (group-aware, by question_id)
  - Reports mean ± std across folds
  - Integrated Gradients removed (see 5fold-train-MCQ-IG.py)

Input:  One or more MCQ challenge JSON files produced by the challenge scripts
Output: Per-fold metrics + aggregated cv_summary.json

Setup:
    pip install torch transformers scikit-learn pandas numpy matplotlib

Usage:
    python 5fold-train-MCQ-noIG.py --model distilbert-base-uncased --input MCQ/mahas-mcq/Llama/mahas-mcq-challenge-llama33.json --output results/distilbert_mcq_cv
    python 5fold-train-MCQ-noIG.py --model distilbert-base-uncased --input MCQ/mahas-mcq/Llama/mahas-mcq-challenge-llama33.json MCQ/mahas-mcq/Haiku/mahas-mcq-challenge-haiku35.json --output results/distilbert_mcq_merged_cv
"""

import os
import json
import hashlib
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class RebuttalDataset(Dataset):
    """PyTorch dataset for tokenized MCQ rebuttal pairs."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ─────────────────────────────────────────────
# DATA LOADING + 5-FOLD SPLIT
# ─────────────────────────────────────────────

def load_and_split_kfold(json_paths, n_folds=5, val_size=0.15, random_state=42):
    """
    Load one or more MCQ challenge JSON files, encode labels, build 5 group-aware folds.
    Returns list of (train_df, val_df, test_df) tuples — one per fold.

    Each df has columns: question_id, mcq_context, rebuttal_text, label, label_encoded
    """
    all_challenges = []
    for path in json_paths:
        with open(path, "r") as f:
            data = json.load(f)
        challenges = data.get("challenges", [])
        print(f"Loaded {len(challenges)} challenges from {path}")
        all_challenges.extend(challenges)

    print(f"Total challenges loaded: {len(all_challenges)}")

    rows = []
    for q in all_challenges:
        question = q.get("question", "")
        choices = q.get("choices", ["", "", "", ""])
        rebuttal_text = q.get("rebuttal_text", "")
        label = q.get("label", "")

        # Build question_id from MD5 hash of question text
        question_id = hashlib.md5(question.strip().lower().encode()).hexdigest()[:8]

        # Build MCQ context: question + choices
        c = choices
        mcq_context = (
            f"Q: {question}\n"
            f"A) {c[0] if len(c) > 0 else ''}\n"
            f"B) {c[1] if len(c) > 1 else ''}\n"
            f"C) {c[2] if len(c) > 2 else ''}\n"
            f"D) {c[3] if len(c) > 3 else ''}"
        )

        rows.append({
            "question_id": question_id,
            "mcq_context": mcq_context,
            "rebuttal_text": rebuttal_text,
            "label": label,
        })

    df = pd.DataFrame(rows)

    # Drop rows with missing rebuttal text
    before = len(df)
    df = df.dropna(subset=["rebuttal_text"]).copy()
    df = df[df["rebuttal_text"].astype(str).str.strip() != ""].copy()
    if len(df) < before:
        print(f"WARNING: dropped {before - len(df)} rows with missing rebuttal_text")

    # Filter to valid labels only (drop AMBIGUOUS, ERROR, etc.)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    valid_labels = {"FLIP", "HOLD"}
    invalid_mask = ~df["label"].isin(valid_labels)
    if invalid_mask.any():
        unexpected = set(df.loc[invalid_mask, "label"].unique())
        print(f"WARNING: dropping {invalid_mask.sum()} rows with unexpected labels: {unexpected}")
        df = df[~invalid_mask].copy()

    # Encode labels
    df["label_encoded"] = (df["label"] == "FLIP").astype(int)

    print(f"\nBefore robust/brittle filter: {len(df)} samples, {df['question_id'].nunique()} questions")
    print(f"  FLIP: {df['label_encoded'].sum()} ({df['label_encoded'].mean()*100:.1f}%)")
    print(f"  HOLD: {(1-df['label_encoded']).sum():.0f} ({(1-df['label_encoded'].mean())*100:.1f}%)")

    # Drop robust (0-1/20 flips) and brittle (19-20/20 flips) questions. They don't
    # teach the classifier rebuttal-style features — the question alone determines
    # the label — and inflate the trivial "always predict majority" baseline.
    # Threshold is locked at 0.9: keep questions whose flip rate is in [0.1, 0.9].
    per_question_flip_rate = df.groupby("question_id")["label_encoded"].mean()
    low, high = 0.1, 0.9
    mixed_questions = per_question_flip_rate[
        (per_question_flip_rate >= low) & (per_question_flip_rate <= high)
    ].index
    n_robust = (per_question_flip_rate < low).sum()
    n_brittle = (per_question_flip_rate > high).sum()
    before = len(df)
    df = df[df["question_id"].isin(mixed_questions)].copy()
    print(f"\nFiltered robust/brittle (locked threshold 0.9):")
    print(f"  Robust (≤1/20 flips):  {n_robust} questions dropped")
    print(f"  Brittle (≥19/20 flips): {n_brittle} questions dropped")
    print(f"  Samples removed:       {before - len(df)}")

    print(f"\nUsable samples: {len(df)}")
    print(f"  FLIP: {df['label_encoded'].sum()} ({df['label_encoded'].mean()*100:.1f}%)")
    print(f"  HOLD: {(1-df['label_encoded']).sum():.0f} ({(1-df['label_encoded'].mean())*100:.1f}%)")
    print(f"  Unique question_ids: {df['question_id'].nunique()}")

    # Guard: need enough questions for stratified group k-fold
    n_unique = df["question_id"].nunique()
    min_questions = n_folds * 2
    if n_unique < min_questions:
        raise ValueError(
            f"Only {n_unique} unique questions available — need at least {min_questions} "
            f"for {n_folds}-fold CV. Provide more data or reduce --n-folds."
        )

    # Outer 5-fold for test
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    folds = []
    overall_flip_rate = df["label_encoded"].mean()

    for fold_idx, (train_val_idx, test_idx) in enumerate(
        sgkf.split(df, df["label_encoded"], groups=df["question_id"])
    ):
        df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        # Inner split: separate validation from training (~val_size of remainder)
        n_val_folds = max(2, round(1 / val_size))
        sgkf_val = StratifiedGroupKFold(
            n_splits=n_val_folds, shuffle=True, random_state=random_state + fold_idx
        )
        train_idx, val_idx = next(sgkf_val.split(
            df_train_val, df_train_val["label_encoded"], groups=df_train_val["question_id"]
        ))

        df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
        df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

        # Verify no question leakage
        train_questions = set(df_train["question_id"])
        val_questions = set(df_val["question_id"])
        test_questions = set(df_test["question_id"])
        assert train_questions.isdisjoint(val_questions), f"Fold {fold_idx}: train/val overlap!"
        assert train_questions.isdisjoint(test_questions), f"Fold {fold_idx}: train/test overlap!"
        assert val_questions.isdisjoint(test_questions), f"Fold {fold_idx}: val/test overlap!"

        print(f"\nFold {fold_idx + 1}/{n_folds}:")
        print(f"  Train: {len(df_train):4d}  ({len(train_questions)} qs, {df_train['label_encoded'].sum()} flips, "
              f"flip rate {df_train['label_encoded'].mean():.1%})")
        print(f"  Val:   {len(df_val):4d}  ({len(val_questions)} qs, {df_val['label_encoded'].sum()} flips, "
              f"flip rate {df_val['label_encoded'].mean():.1%})")
        print(f"  Test:  {len(df_test):4d}  ({len(test_questions)} qs, {df_test['label_encoded'].sum()} flips, "
              f"flip rate {df_test['label_encoded'].mean():.1%})")

        folds.append((df_train, df_val, df_test))

    print(f"\n  Overall FLIP rate: {overall_flip_rate:.1%}")
    return folds


# ─────────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────────

def tokenize_data(tokenizer, texts_a, texts_b, max_length=256):
    """Tokenize MCQ context + rebuttal as a sentence pair.

    `truncation="only_first"` truncates the MCQ context (segment A) when needed
    and preserves the rebuttal (segment B) intact. The rebuttal is the
    feature-bearing signal for FLIP/HOLD; truncating its tail (e.g. a cited
    source or final assertion) silently destroys signal. Default `truncation=True`
    uses `longest_first`, which can eat into the rebuttal once the MCQ context
    has been shortened to its length.
    """
    return tokenizer(
        texts_a,
        texts_b,
        padding="max_length",
        truncation="only_first",
        max_length=max_length,
        return_tensors="pt",
    )


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_model(
    model,
    tokenizer,
    train_loader,
    val_loader,
    class_weights,
    device,
    epochs=5,
    learning_rate=2e-5,
    patience=2,
    output_dir="checkpoints",
):
    """Fine-tune the model with early stopping based on validation F1."""

    os.makedirs(output_dir, exist_ok=True)

    # Loss function with class weights
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Optimizer — exclude bias and LayerNorm from weight decay
    no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias"}
    optimizer_grouped_params = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=learning_rate)

    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Init below 0 so the first epoch always saves a checkpoint, even in the
    # degenerate case where val F1 stays at exactly 0 (otherwise no checkpoint
    # is ever written and the post-train load_state_dict crashes).
    best_val_f1 = -1.0
    patience_counter = 0
    training_log = []

    for epoch in range(epochs):
        # ── Training ──
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []

        for batch in train_loader:
            optimizer.zero_grad()

            # Forward all encoding keys (input_ids, attention_mask, and
            # token_type_ids when the tokenizer produced them — needed for BERT).
            labels = batch["labels"].to(device)
            model_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            outputs = model(**model_inputs)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)

        # ── Validation ──
        val_f1, val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_f1": val_f1,
        }
        training_log.append(epoch_log)

        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            # Save best checkpoint
            checkpoint_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)
            tokenizer.save_pretrained(output_dir)
            print(f"    ✓ New best model saved (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    ✗ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Load best checkpoint
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=True))
    return model, training_log


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset. Returns F1, loss, predictions, labels."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            labels = batch["labels"].to(device)
            model_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            outputs = model(**model_inputs)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return f1, avg_loss, np.array(all_preds), np.array(all_labels)


# ─────────────────────────────────────────────
# TEST EVALUATION WITH CONFIDENCE INTERVALS
# ─────────────────────────────────────────────

def evaluate_test(model, test_loader, device, n_bootstrap=1000):
    """Full evaluation on test set with bootstrapped confidence intervals."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            labels = batch["labels"].to(device)
            model_inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}

            outputs = model(**model_inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Point estimates
    metrics = {
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "accuracy": accuracy_score(all_labels, all_preds),
        "roc_auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0,
    }

    # Bootstrap confidence intervals
    rng = np.random.RandomState(42)
    bootstrap_metrics = {k: [] for k in metrics}

    for _ in range(n_bootstrap):
        indices = rng.choice(len(all_labels), size=len(all_labels), replace=True)
        boot_labels = all_labels[indices]
        boot_preds = all_preds[indices]
        boot_probs = all_probs[indices]

        if len(np.unique(boot_labels)) < 2:
            continue

        bootstrap_metrics["f1"].append(f1_score(boot_labels, boot_preds, zero_division=0))
        bootstrap_metrics["precision"].append(precision_score(boot_labels, boot_preds, zero_division=0))
        bootstrap_metrics["recall"].append(recall_score(boot_labels, boot_preds, zero_division=0))
        bootstrap_metrics["accuracy"].append(accuracy_score(boot_labels, boot_preds))
        bootstrap_metrics["roc_auc"].append(roc_auc_score(boot_labels, boot_probs))

    confidence_intervals = {}
    for k, values in bootstrap_metrics.items():
        if values:
            confidence_intervals[k] = {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
            }

    # Confusion matrix — force 2x2 layout even if a fold goes single-class
    # (otherwise cm collapses to 1x1 and the ravel unpack crashes).
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # ROC curve data (guarded for single-class folds)
    if len(np.unique(all_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    else:
        fpr, tpr, thresholds = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5])

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=["HOLD", "FLIP"])

    return {
        "metrics": metrics,
        "confidence_intervals": confidence_intervals,
        "confusion_matrix": cm.tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()},
        "classification_report": report,
        "predictions": all_preds.tolist(),
        "labels": all_labels.tolist(),
        "probabilities": all_probs.tolist(),
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train sycophancy flip classifier on MCQ JSON with 5-fold CV")
    parser.add_argument("--model", required=True,
                        choices=["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
                        help="HuggingFace model name")
    parser.add_argument("--input", required=True, nargs="+",
                        help="One or more MCQ challenge JSON file paths")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds (default: 5)")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256,
                        help="Max token length for question+choices+rebuttal pair (default: 256)")
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Reproducibility: seed all sources of randomness
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(SEED)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: cuda — {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: mps (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Device: cpu")

    # ── Step 1: Load and 5-fold split ──
    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"K-fold: {args.n_folds}")
    print(f"Input files: {args.input}")
    print(f"{'='*50}")

    folds = load_and_split_kfold(args.input, n_folds=args.n_folds)

    # ── Step 2: Loop over folds ──
    fold_results = []

    for fold_idx, (df_train, df_val, df_test) in enumerate(folds):
        print(f"\n{'#'*50}")
        print(f"  FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'#'*50}")

        fold_dir = os.path.join(args.output, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        # Tokenize as sentence pairs: (mcq_context, rebuttal_text)
        print(f"\nTokenizing with {args.model}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

        train_encodings = tokenize_data(
            tokenizer,
            df_train["mcq_context"].tolist(),
            df_train["rebuttal_text"].tolist(),
            args.max_length,
        )
        val_encodings = tokenize_data(
            tokenizer,
            df_val["mcq_context"].tolist(),
            df_val["rebuttal_text"].tolist(),
            args.max_length,
        )
        test_encodings = tokenize_data(
            tokenizer,
            df_test["mcq_context"].tolist(),
            df_test["rebuttal_text"].tolist(),
            args.max_length,
        )

        train_dataset = RebuttalDataset(train_encodings, df_train["label_encoded"].tolist())
        val_dataset = RebuttalDataset(val_encodings, df_val["label_encoded"].tolist())
        test_dataset = RebuttalDataset(test_encodings, df_test["label_encoded"].tolist())

        g = torch.Generator()
        g.manual_seed(SEED + fold_idx)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        # Class weights from this fold's train set
        labels_array = np.array(df_train["label_encoded"].tolist())
        class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels_array)
        print(f"\nClass weights: HOLD={class_weights[0]:.3f}, FLIP={class_weights[1]:.3f}")

        # Load fresh model per fold
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
        model.to(device)

        if fold_idx == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Total parameters: {total_params:,}")

        # Train
        print(f"\nTraining fold {fold_idx + 1}...")
        model, training_log = train_model(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=device,
            epochs=args.epochs,
            learning_rate=args.lr,
            patience=args.patience,
            output_dir=os.path.join(fold_dir, "checkpoints"),
        )

        # Evaluate on test set
        print(f"\nEvaluating fold {fold_idx + 1} on test set...")
        test_results = evaluate_test(model, test_loader, device)

        print(f"\n{test_results['classification_report']}")
        print(f"Confusion Matrix:")
        print(f"  {test_results['confusion_matrix']}")
        print(f"\nMetrics with 95% CI:")
        for metric, value in test_results["metrics"].items():
            ci = test_results["confidence_intervals"].get(metric, {})
            lower = ci.get("lower", "N/A")
            upper = ci.get("upper", "N/A")
            if isinstance(lower, float):
                print(f"  {metric:>12}: {value:.4f} ({lower:.4f} – {upper:.4f})")
            else:
                print(f"  {metric:>12}: {value:.4f}")

        # Save ROC curve plot per fold
        roc_fpr = test_results["roc_curve"]["fpr"]
        roc_tpr = test_results["roc_curve"]["tpr"]
        auc_val = test_results["metrics"]["roc_auc"]

        plt.figure(figsize=(6, 6))
        plt.plot(roc_fpr, roc_tpr, lw=2, label=f"{args.model} fold {fold_idx+1} (AUC = {auc_val:.3f})")
        plt.plot([0, 1], [0, 1], "k--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve — {args.model} (fold {fold_idx+1})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join(fold_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()

        # Save per-fold results
        fold_record = {
            "fold": fold_idx,
            "model": args.model,
            "timestamp": datetime.now().isoformat(),
            "device": str(device),
            "input_files": args.input,
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "max_length": args.max_length,
                "patience": args.patience,
            },
            "training_log": training_log,
            "test_results": {
                "metrics": test_results["metrics"],
                "confidence_intervals": test_results["confidence_intervals"],
                "confusion_matrix": test_results["confusion_matrix"],
                "classification_report": test_results["classification_report"],
            },
        }

        with open(os.path.join(fold_dir, "results.json"), "w") as f:
            json.dump(fold_record, f, indent=2)

        # Save predictions per fold
        pred_df = df_test.copy()
        pred_df["predicted"] = test_results["predictions"]
        pred_df["probability"] = test_results["probabilities"]
        pred_df.to_csv(os.path.join(fold_dir, "predictions.csv"), index=False)

        fold_results.append(fold_record)

        print(f"\n  ✓ Fold {fold_idx + 1} done. Results saved to {fold_dir}/")

    # ── Step 3: Aggregate metrics across folds ──
    print(f"\n{'='*50}")
    print(f"AGGREGATING ACROSS {args.n_folds} FOLDS")
    print(f"{'='*50}")

    metric_keys = ["f1", "precision", "recall", "accuracy", "roc_auc", "fpr"]
    aggregated = {}
    for key in metric_keys:
        values = [fr["test_results"]["metrics"][key] for fr in fold_results]
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "per_fold": values,
        }

    print(f"\nMean ± std across folds:")
    for key in metric_keys:
        m = aggregated[key]["mean"]
        s = aggregated[key]["std"]
        per_fold = aggregated[key]["per_fold"]
        per_fold_str = ", ".join(f"{v:.4f}" for v in per_fold)
        print(f"  {key:>12}: {m:.4f} ± {s:.4f}  (per-fold: [{per_fold_str}])")

    cv_summary = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "n_folds": args.n_folds,
        "device": str(device),
        "input_files": args.input,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_length": args.max_length,
            "patience": args.patience,
        },
        "aggregated_metrics": aggregated,
    }

    cv_path = os.path.join(args.output, "cv_summary.json")
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2)
    print(f"\n✓ CV summary saved to {cv_path}")

    print(f"\n{'='*50}")
    print(f"DONE — {args.model}  ({args.n_folds}-fold CV)")
    print(f"  F1:  {aggregated['f1']['mean']:.4f} ± {aggregated['f1']['std']:.4f}")
    print(f"  AUC: {aggregated['roc_auc']['mean']:.4f} ± {aggregated['roc_auc']['std']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
