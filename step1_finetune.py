"""
Step 1 — Fine-tune BERT, RoBERTa, and DistilBERT for sycophancy detection.

Task: Binary classification on rebuttal text.
  FLIP = 1  →  model was fooled by the rebuttal (sycophantic)
  HOLD = 0  →  model held its ground (robust)

Outputs (all saved so you can resume if anything crashes):
  models/bert_best.pt        ← best BERT checkpoint
  models/roberta_best.pt     ← best RoBERTa checkpoint
  models/distilbert_best.pt  ← best DistilBERT checkpoint
  results/train_split.csv    ← training rows
  results/val_split.csv      ← validation rows
  results/test_split.csv     ← held-out test rows (NOT touched during training)
  results/training_metrics.csv  ← loss + val F1 per epoch per model

Usage:
  python step1_finetune.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import json
import os
import random
import warnings
warnings.filterwarnings("ignore")

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import (
    BertTokenizer,         BertForSequenceClassification,
    RobertaTokenizer,      RobertaForSequenceClassification,
    DistilBertTokenizer,   DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — change these values to experiment
# ═══════════════════════════════════════════════════════════════════════════════
# All 6 result files — each person tested different questions, so we concatenate directly
DATA_FILES = {
    "loly":    "challenge_gemini3flash.json",
    "maha":    "mahas-results.json",
    "jay":     "jays-challenge.json",
    "munirah": "munera-challenge.json",
    "roaa":    "roaas-challenge.json",
    "khuzama": "khuzama-challenge.json",
}

OUTPUT_DIR    = "models"    # where model checkpoints are saved
RESULTS_DIR   = "results"   # where CSVs are saved

SEED          = 42
MAX_LEN       = 128   # token length; rebuttals are short so 128 is plenty
BATCH_SIZE    = 64    # RTX 4090 has 24GB VRAM — 64 fits easily and trains faster
LEARNING_RATE = 2e-5  # safe default for fine-tuning BERT-family models
WEIGHT_DECAY  = 0.01  # AdamW regularisation (prevents overfitting)
NUM_EPOCHS    = 5
PATIENCE      = 2     # early stopping: stop if val F1 doesn't improve for 2 epochs

# Train / val / test proportions (must sum to 1.0)
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15  # held out — NEVER used during training or early stopping

# Models to train (key = short name used for file names, value = HuggingFace ID)
MODELS = {
    "bert":       "bert-base-uncased",
    "roberta":    "roberta-base",
    "distilbert": "distilbert-base-uncased",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """Fix all random seeds so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# Auto-detect the best available device:
#   CUDA  → NVIDIA GPU
#   MPS   → Apple Silicon (M1/M2/M3/M4) GPU
#   CPU   → fallback (slower but works)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1-A: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

def build_dataframe(data_files: dict) -> pd.DataFrame:
    """
    Load and concatenate all 6 result JSON files into one clean DataFrame.

    Because each person tested different questions, rebuttal_ids repeat across
    files (Q000_R00 appears in every file) but refer to different content.
    We add a 'person' column to keep records uniquely identifiable.

    Drops AMBIGUOUS labels and records with no label field.
    Maps  FLIP → 1  |  HOLD → 0
    """
    records = []
    for person, path in data_files.items():
        with open(path, "r") as f:
            challenges = json.load(f)["challenges"]

        n_before = len(records)
        for item in challenges:
            label = item.get("label")
            if label not in ("FLIP", "HOLD"):
                continue
            records.append({
                "person":        person,
                "rebuttal_id":   item["rebuttal_id"],
                "batch":         item.get("batch", ""),
                "rebuttal_text": item["rebuttal_text"],
                "label":         1 if label == "FLIP" else 0,
            })
        added = len(records) - n_before
        print(f"  {person:<10}: {added} usable records")

    df = pd.DataFrame(records)
    n_flip = df["label"].sum()
    n_hold = (df["label"] == 0).sum()
    print(f"\nTotal usable: {len(df)}  →  {n_flip} FLIP  |  {n_hold} HOLD")
    print(f"Class ratio : 1 FLIP per {n_hold / n_flip:.1f} HOLDs")
    return df


print("Loading datasets:")
df = build_dataframe(DATA_FILES)

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1-B: STRATIFIED TRAIN / VAL / TEST SPLIT
# ═══════════════════════════════════════════════════════════════════════════════
# "stratified" means each split has the same FLIP/HOLD ratio as the full dataset.
# This prevents any split from being accidentally all-HOLD or all-FLIP.

train_df, temp_df = train_test_split(
    df,
    test_size=VAL_RATIO + TEST_RATIO,
    stratify=df["label"],
    random_state=SEED,
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,                    # split the temp half-half → 15% / 15%
    stratify=temp_df["label"],
    random_state=SEED,
)

print(f"\nSplit sizes:")
print(f"  Train : {len(train_df):4d}  ({train_df['label'].sum()} FLIP)")
print(f"  Val   : {len(val_df):4d}  ({val_df['label'].sum()} FLIP)")
print(f"  Test  : {len(test_df):4d}  ({test_df['label'].sum()} FLIP)  ← held out")

# Save splits — important so Step 2 evaluation uses the exact same test set
train_df.to_csv(f"{RESULTS_DIR}/train_split.csv", index=False)
val_df.to_csv(  f"{RESULTS_DIR}/val_split.csv",   index=False)
test_df.to_csv( f"{RESULTS_DIR}/test_split.csv",  index=False)
print(f"\nSplits saved to {RESULTS_DIR}/")

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1-C: CLASS WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════
# Because HOLD outnumbers FLIP ~8:1, an unweighted model learns to just predict
# HOLD all the time and still gets high accuracy — that's useless for us.
# Giving the loss function a higher weight for FLIP forces the model to pay
# more attention to the minority class.

class_weights = compute_class_weight(
    class_weight="balanced",         # automatically sets weight ∝ 1 / class_frequency
    classes=np.array([0, 1]),
    y=train_df["label"].values,
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"\nClass weights  →  HOLD: {class_weights[0]:.3f}  |  FLIP: {class_weights[1]:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1-D: PYTORCH DATASET
# ═══════════════════════════════════════════════════════════════════════════════

class RebuttaDataset(Dataset):
    """
    Wraps our DataFrame so PyTorch can iterate over it in batches.

    The tokenizer converts raw text into three tensors:
      input_ids      → integer token IDs the model understands
      attention_mask → 1 for real tokens, 0 for padding
      label          → 0 or 1
    """

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
            padding="max_length",   # pad shorter texts with zeros
            truncation=True,        # cut texts longer than max_len
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }

# ═══════════════════════════════════════════════════════════════════════════════
#  STEP 1-E: TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_model(
    model_key: str,
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> list[dict]:
    """
    Fine-tune one transformer model and save the best checkpoint.

    Returns a list of per-epoch metric dicts (for plotting / comparison later).
    """
    print(f"\n{'═' * 60}")
    print(f"  Training: {model_key.upper()}  ({model_name})")
    print(f"{'═' * 60}")

    # ── Load tokenizer + model from HuggingFace Hub ───────────────────────────
    # First run: downloads ~440 MB and caches it. Subsequent runs use the cache.
    if model_key == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model     = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    elif model_key == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model     = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:  # distilbert
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model     = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    model.to(device)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = RebuttaDataset(train_df, tokenizer, MAX_LEN)
    val_ds   = RebuttaDataset(val_df,   tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # AdamW is the standard choice for transformer fine-tuning.
    # weight_decay adds L2 regularisation (discourages large weights → less overfitting).
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # ── Learning-rate scheduler ───────────────────────────────────────────────
    # Linearly warms up the LR for the first 10% of steps, then linearly decays.
    # This prevents the model from making large, unstable updates early on.
    total_steps   = len(train_loader) * NUM_EPOCHS
    warmup_steps  = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    # CrossEntropyLoss with class weights so FLIP mistakes cost more.
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # ── Training state ────────────────────────────────────────────────────────
    best_val_f1    = 0.0
    patience_count = 0
    metrics_log    = []
    best_ckpt_path = f"{OUTPUT_DIR}/{model_key}_best.pt"

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):

        # ── Training phase ────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = loss_fn(outputs.logits, labels)
            loss.backward()

            # Gradient clipping prevents the "exploding gradient" problem
            # (gradients can grow very large in deep networks and destabilise training)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # ── Validation phase ──────────────────────────────────────────────────
        model.eval()
        val_preds, val_true = [], []

        with torch.no_grad():   # no gradient tracking needed during validation
            for batch in val_loader:
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds   = torch.argmax(outputs.logits, dim=1)

                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        # Macro F1 treats both classes equally regardless of their size.
        # This is better than accuracy when classes are imbalanced.
        val_f1 = f1_score(val_true, val_preds, average="macro")
        print(f"  Epoch {epoch}/{NUM_EPOCHS}  |  Train Loss: {avg_train_loss:.4f}  |  Val Macro-F1: {val_f1:.4f}")

        metrics_log.append({
            "model":      model_key,
            "epoch":      epoch,
            "train_loss": round(avg_train_loss, 5),
            "val_f1":     round(val_f1, 5),
        })

        # ── Early stopping + checkpoint ───────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1    = val_f1
            patience_count = 0
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  ✓ New best  →  saved checkpoint  (F1 = {best_val_f1:.4f})")
        else:
            patience_count += 1
            print(f"  No improvement  ({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print(f"  Early stopping triggered.")
                break

    print(f"\n  Best Val F1 for {model_key.upper()}: {best_val_f1:.4f}")
    print(f"  Checkpoint: {best_ckpt_path}")
    return metrics_log

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN — train all three models sequentially
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_metrics = []

    for model_key, model_name in MODELS.items():
        epoch_metrics = train_one_model(model_key, model_name, train_df, val_df)
        all_metrics.extend(epoch_metrics)

    # Save per-epoch training history for all models
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = f"{RESULTS_DIR}/training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n{'═' * 60}")
    print("All models trained.")
    print(f"Training history → {metrics_path}")
    print(f"Best checkpoints → {OUTPUT_DIR}/{{bert,roberta,distilbert}}_best.pt")
    print(f"Test set (held out) → {RESULTS_DIR}/test_split.csv")
    print("Ready for Step 2: Evaluation")
    print(f"{'═' * 60}")
