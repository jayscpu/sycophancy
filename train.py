"""
Sycophancy Flip Prediction — Training Pipeline
================================================
One script for all three models: BERT, RoBERTa, DistilBERT.

Input:  Merged CSV with columns: question_id, rebuttal_text, label (FLIP/HOLD)
Output: Evaluation metrics, confusion matrix, IG attributions

Setup:
    pip install torch transformers datasets scikit-learn captum pandas numpy matplotlib seaborn

Usage:
    python train.py --model bert-base-uncased --input merged_data.csv --output results/bert
    python train.py --model roberta-base --input merged_data.csv --output results/roberta
    python train.py --model distilbert-base-uncased --input merged_data.csv --output results/distilbert

Works on Google Colab (GPU) or local machine.
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    """PyTorch dataset for tokenized rebuttals."""

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
# DATA LOADING AND SPLITTING
# ─────────────────────────────────────────────

def load_and_split(csv_path: str, test_size=0.15, val_size=0.15, random_state=42):
    """
    Load CSV, encode labels, perform group-stratified split.
    Returns train_df, val_df, test_df.
    """
    df = pd.read_csv(csv_path)

    # Standardize column names
    if "rebuttal" in df.columns and "rebuttal_text" not in df.columns:
        df = df.rename(columns={"rebuttal": "rebuttal_text"})

    # Drop rows with missing rebuttal text
    before = len(df)
    df = df.dropna(subset=["rebuttal_text"]).copy()
    df = df[df["rebuttal_text"].astype(str).str.strip() != ""].copy()
    if len(df) < before:
        print(f"WARNING: dropped {before - len(df)} rows with missing rebuttal_text")

    # Filter to valid labels only (drop AMBIGUOUS, typos, etc.)
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    valid_labels = {"FLIP", "HOLD"}
    invalid_mask = ~df["label"].isin(valid_labels)
    if invalid_mask.any():
        unexpected = set(df.loc[invalid_mask, "label"].unique())
        print(f"WARNING: dropping {invalid_mask.sum()} rows with unexpected labels: {unexpected}")
        df = df[~invalid_mask].copy()

    # Encode labels
    df["label_encoded"] = (df["label"] == "FLIP").astype(int)

    print(f"Loaded {len(df)} samples")
    print(f"  FLIP: {df['label_encoded'].sum()} ({df['label_encoded'].mean()*100:.1f}%)")
    print(f"  HOLD: {(1-df['label_encoded']).sum():.0f} ({(1-df['label_encoded'].mean())*100:.1f}%)")

    # Stratified group split: preserves FLIP/HOLD ratio while keeping all
    # rebuttals for the same question in the same split.
    # Strategy: use StratifiedGroupKFold to carve out test, then val from remainder.

    # First split: separate test set (~15%)
    n_test_folds = max(2, round(1 / test_size))  # e.g., 1/0.15 ≈ 7 folds
    sgkf_test = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=random_state)
    train_val_idx, test_idx = next(sgkf_test.split(
        df, df["label_encoded"], groups=df["question_id"]
    ))

    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # Second split: separate validation from training (~15% of original = ~17.6% of remainder)
    n_val_folds = max(2, round(1 / (val_size / (1 - test_size))))
    sgkf_val = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(sgkf_val.split(
        df_train_val, df_train_val["label_encoded"], groups=df_train_val["question_id"]
    ))

    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val = df_train_val.iloc[val_idx].reset_index(drop=True)

    # Verify no question leakage
    train_questions = set(df_train["question_id"])
    val_questions = set(df_val["question_id"])
    test_questions = set(df_test["question_id"])
    assert train_questions.isdisjoint(val_questions), "Question leak: train/val overlap!"
    assert train_questions.isdisjoint(test_questions), "Question leak: train/test overlap!"
    assert val_questions.isdisjoint(test_questions), "Question leak: val/test overlap!"

    # Report stratification quality
    overall_flip_rate = df["label_encoded"].mean()
    print(f"\n  Overall FLIP rate: {overall_flip_rate:.1%}")
    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        split_rate = split_df["label_encoded"].mean()
        print(f"  {name:>5} FLIP rate:  {split_rate:.1%}")

    print(f"\nSplit sizes:")
    print(f"  Train: {len(df_train)} ({len(train_questions)} questions, "
          f"{df_train['label_encoded'].sum()} flips)")
    print(f"  Val:   {len(df_val)} ({len(val_questions)} questions, "
          f"{df_val['label_encoded'].sum()} flips)")
    print(f"  Test:  {len(df_test)} ({len(test_questions)} questions, "
          f"{df_test['label_encoded'].sum()} flips)")

    return df_train, df_val, df_test


# ─────────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────────

def tokenize_data(tokenizer, texts, max_length=128):
    """Tokenize a list of texts."""
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
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

    best_val_f1 = 0
    patience_counter = 0
    training_log = []

    for epoch in range(epochs):
        # ── Training ──
        model.train()
        total_loss = 0
        train_preds, train_labels = [], []

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
        train_f1 = f1_score(train_labels, train_preds)

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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds)
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
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
        "f1": f1_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds),
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

        bootstrap_metrics["f1"].append(f1_score(boot_labels, boot_preds))
        bootstrap_metrics["precision"].append(precision_score(boot_labels, boot_preds, zero_division=0))
        bootstrap_metrics["recall"].append(recall_score(boot_labels, boot_preds))
        bootstrap_metrics["accuracy"].append(accuracy_score(boot_labels, boot_preds))
        bootstrap_metrics["roc_auc"].append(roc_auc_score(boot_labels, boot_probs))

    confidence_intervals = {}
    for k, values in bootstrap_metrics.items():
        if values:
            confidence_intervals[k] = {
                "lower": float(np.percentile(values, 2.5)),
                "upper": float(np.percentile(values, 97.5)),
            }

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # ROC curve data
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

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
# INTEGRATED GRADIENTS
# ─────────────────────────────────────────────

def run_integrated_gradients(model, tokenizer, test_df, device, n_steps=50, max_samples=200):
    """
    Run Integrated Gradients on test samples.
    Returns token-level attributions for each sample.
    """
    from captum.attr import LayerIntegratedGradients

    model.eval()
    model.zero_grad()

    # Get the embedding layer
    if hasattr(model, "bert"):
        embeddings = model.bert.embeddings
    elif hasattr(model, "roberta"):
        embeddings = model.roberta.embeddings
    elif hasattr(model, "distilbert"):
        embeddings = model.distilbert.embeddings
    else:
        raise ValueError("Unknown model architecture — cannot find embedding layer")

    lig = LayerIntegratedGradients(
        lambda input_ids, attention_mask: model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits,
        embeddings,
    )

    # Limit samples for efficiency
    samples = test_df.head(max_samples)
    all_attributions = []

    print(f"\nRunning Integrated Gradients on {len(samples)} samples...")

    for idx, row in samples.iterrows():
        text = row["rebuttal_text"]
        true_label = row["label_encoded"]

        # Tokenize
        encoding = tokenizer(
            text, padding="max_length", truncation=True,
            max_length=128, return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Baseline: all PAD tokens (correct per-model)
        baseline_ids = torch.full_like(input_ids, tokenizer.pad_token_id).to(device)

        # Get model prediction
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_class = torch.argmax(output.logits, dim=1).item()
            pred_prob = torch.softmax(output.logits, dim=1)[0, pred_class].item()

        # Compute attributions for predicted class
        attributions = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            target=pred_class,
            n_steps=n_steps,
        )

        # Sum attributions across embedding dimensions
        attr_sum = attributions.sum(dim=-1).squeeze(0)
        attr_scores = attr_sum.cpu().detach().numpy()

        # Map tokens to words
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())

        # Filter out padding and special tokens
        token_attributions = []
        for token, score in zip(tokens, attr_scores):
            if token in [tokenizer.pad_token, tokenizer.cls_token,
                         tokenizer.sep_token, "<pad>", "[PAD]"]:
                continue
            token_attributions.append({
                "token": token,
                "attribution": float(score),
            })

        all_attributions.append({
            "rebuttal_text": text,
            "true_label": "FLIP" if true_label == 1 else "HOLD",
            "predicted_label": "FLIP" if pred_class == 1 else "HOLD",
            "prediction_confidence": pred_prob,
            "token_attributions": token_attributions,
        })

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(samples)} samples")

    print(f"  ✓ Integrated Gradients complete")
    return all_attributions


def aggregate_attributions(attributions):
    """Aggregate token attributions across all samples to find top tokens."""
    from collections import defaultdict

    token_scores = defaultdict(list)

    for sample in attributions:
        if sample["predicted_label"] == "FLIP":
            for ta in sample["token_attributions"]:
                token = ta["token"].lower().replace("##", "").replace("Ġ", "")
                if len(token) > 1:  # Skip single characters
                    token_scores[token].append(ta["attribution"])

    # Compute mean attribution per token
    token_means = {
        token: {"mean": np.mean(scores), "count": len(scores)}
        for token, scores in token_scores.items()
        if len(scores) >= 3  # Minimum frequency
    }

    # Sort by mean attribution
    ranked = sorted(token_means.items(), key=lambda x: x[1]["mean"], reverse=True)
    return ranked


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train sycophancy flip classifier")
    parser.add_argument("--model", required=True,
                        choices=["bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
                        help="HuggingFace model name")
    parser.add_argument("--input", required=True, help="Path to merged CSV")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--ig-samples", type=int, default=200,
                        help="Number of test samples for Integrated Gradients")
    parser.add_argument("--skip-ig", action="store_true", help="Skip Integrated Gradients")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Reproducibility: seed all sources of randomness
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
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

    # ── Step 1: Load and split ──
    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"{'='*50}")

    df_train, df_val, df_test = load_and_split(args.input)

    # ── Step 2: Tokenize ──
    print(f"\nTokenizing with {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_encodings = tokenize_data(tokenizer, df_train["rebuttal_text"].tolist(), args.max_length)
    val_encodings = tokenize_data(tokenizer, df_val["rebuttal_text"].tolist(), args.max_length)
    test_encodings = tokenize_data(tokenizer, df_test["rebuttal_text"].tolist(), args.max_length)

    train_dataset = RebuttalDataset(train_encodings, df_train["label_encoded"].tolist())
    val_dataset = RebuttalDataset(val_encodings, df_val["label_encoded"].tolist())
    test_dataset = RebuttalDataset(test_encodings, df_test["label_encoded"].tolist())

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # ── Step 3: Compute class weights ──
    labels_array = np.array(df_train["label_encoded"].tolist())
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels_array)
    print(f"\nClass weights: HOLD={class_weights[0]:.3f}, FLIP={class_weights[1]:.3f}")

    # ── Step 4: Load model ──
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ── Step 5: Train ──
    print(f"\nTraining...")
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
        output_dir=os.path.join(args.output, "checkpoints"),
    )

    # ── Step 6: Evaluate on test set ──
    print(f"\nEvaluating on test set...")
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

    # ── Step 6b: Save ROC curve plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roc_fpr = test_results["roc_curve"]["fpr"]
    roc_tpr = test_results["roc_curve"]["tpr"]
    auc_val = test_results["metrics"]["roc_auc"]

    plt.figure(figsize=(6, 6))
    plt.plot(roc_fpr, roc_tpr, lw=2, label=f"{args.model} (AUC = {auc_val:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — {args.model}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = os.path.join(args.output, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"\n✓ ROC curve saved to {roc_path}")

    # ── Step 7: Integrated Gradients ──
    ig_results = None
    top_tokens = None
    if not args.skip_ig:
        ig_results = run_integrated_gradients(
            model, tokenizer, df_test, device,
            max_samples=args.ig_samples,
        )
        top_tokens = aggregate_attributions(ig_results)

        print(f"\nTop 20 tokens driving FLIP predictions:")
        for i, (token, stats) in enumerate(top_tokens[:20]):
            print(f"  {i+1:>2}. {token:<20} mean_attr: {stats['mean']:.4f} (n={stats['count']})")

    # ── Step 8: Save everything ──
    results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
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
        "top_tokens": [{"token": t, **s} for t, s in (top_tokens[:50] if top_tokens else [])],
    }

    results_path = os.path.join(args.output, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_path}")

    # Save IG attributions separately (large file)
    if ig_results:
        ig_path = os.path.join(args.output, "ig_attributions.json")
        with open(ig_path, "w") as f:
            json.dump(ig_results, f, indent=2)
        print(f"✓ IG attributions saved to {ig_path}")

    # Save predictions for further analysis
    pred_df = df_test.copy()
    pred_df["predicted"] = test_results["predictions"]
    pred_df["probability"] = test_results["probabilities"]
    pred_path = os.path.join(args.output, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"✓ Predictions saved to {pred_path}")

    print(f"\n{'='*50}")
    print(f"DONE — {args.model}")
    print(f"  F1: {test_results['metrics']['f1']:.4f}")
    print(f"  AUC: {test_results['metrics']['roc_auc']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
