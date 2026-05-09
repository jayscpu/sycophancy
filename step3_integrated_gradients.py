"""
Step 3 — Integrated Gradients (IG) for Explainability.

What this does:
  For each test sample, IG asks: "which tokens in the rebuttal pushed the model
  toward predicting FLIP or HOLD, and by how much?"
  We then group those token attributions by linguistic batch (A/B/C/D) and
  compare patterns across all three fine-tuned models.

Requires: step1_finetune.py and step2_evaluate.py to have been run first.
Install:  pip install captum

Outputs saved to results/:
  results/ig_attributions_{model}.csv   ← per-token attributions for every test sample
  results/ig_batch_summary.csv          ← mean attribution per batch per model
  results/ig_top_tokens_{model}.csv     ← highest-attribution tokens per model
  results/ig_batch_heatmap.png          ← heatmap comparing batches × models
  results/ig_token_barplots.png         ← top influential tokens per model

Usage:
  python step3_integrated_gradients.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns

from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer,       BertForSequenceClassification,
    RobertaTokenizer,    RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
)

try:
    from captum.attr import LayerIntegratedGradients
except ImportError:
    raise ImportError(
        "Captum is not installed. Run:  pip install captum\n"
        "Then re-run this script."
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
RESULTS_DIR  = "results"
MODELS_DIR   = "models"
MAX_LEN      = 128
BATCH_SIZE   = 8    # smaller batches for IG — it's memory-intensive
N_STEPS      = 50   # integration steps (higher = more accurate but slower; 50 is a good balance)
TARGET_CLASS = 1    # we care about attributions toward FLIP (class 1)

MODELS = {
    "bert":       "bert-base-uncased",
    "roberta":    "roberta-base",
    "distilbert": "distilbert-base-uncased",
}

# Batch labels → human-readable dimension names
# (adjust these if you know the exact mapping for your dataset)
BATCH_LABELS = {
    "A": "Batch A",
    "B": "Batch B",
    "C": "Batch C",
    "D": "Batch D",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  SETUP
# ═══════════════════════════════════════════════════════════════════════════════
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Note: Captum has limited MPS support — fall back to CPU for IG
    print("MPS detected but Captum requires CPU for Integrated Gradients. Using CPU.")
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

print(f"Device for IG: {device}")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LOAD TEST SET
# ═══════════════════════════════════════════════════════════════════════════════
test_path = f"{RESULTS_DIR}/test_split.csv"
if not os.path.exists(test_path):
    raise FileNotFoundError("Run step1_finetune.py first to generate test_split.csv")

test_df = pd.read_csv(test_path)
print(f"Test set: {len(test_df)} samples")

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: get the embedding layer for each model type
#  (IG needs to attribute scores back to the input embeddings, not the tokens)
# ═══════════════════════════════════════════════════════════════════════════════
def get_embedding_layer(model, model_key: str):
    if model_key == "bert":
        return model.bert.embeddings
    elif model_key == "roberta":
        return model.roberta.embeddings
    else:  # distilbert
        return model.distilbert.embeddings

# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: forward function that Captum will differentiate
# ═══════════════════════════════════════════════════════════════════════════════
def make_forward_fn(model, model_key: str):
    """
    Returns a function that takes token embeddings and returns
    the logit for the target class (FLIP).

    Captum computes gradients with respect to these embeddings,
    then integrates them along a straight path from a baseline
    (all-zero embeddings = "no information") to the actual input.
    """
    def forward_fn(embeddings, attention_mask):
        # Each model stores its transformer differently
        if model_key == "bert":
            out = model.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
            pooled = out.pooler_output
            logits = model.classifier(pooled)
        elif model_key == "roberta":
            out = model.roberta(inputs_embeds=embeddings, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0, :]   # CLS token
            logits = model.classifier(pooled)
        else:  # distilbert
            out = model.distilbert(inputs_embeds=embeddings, attention_mask=attention_mask)
            hidden = out.last_hidden_state[:, 0, :]
            hidden = model.pre_classifier(hidden)
            hidden = torch.relu(hidden)
            logits = model.classifier(hidden)
        return logits[:, TARGET_CLASS]   # scalar per sample: P(FLIP) logit

    return forward_fn

# ═══════════════════════════════════════════════════════════════════════════════
#  CORE: compute IG attributions for all test samples
# ═══════════════════════════════════════════════════════════════════════════════
def compute_ig_attributions(model_key: str, model_name: str, test_df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'═' * 60}")
    print(f"  Integrated Gradients: {model_key.upper()}")
    print(f"{'═' * 60}")

    # ── Load model + tokenizer ────────────────────────────────────────────────
    if model_key == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model     = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    elif model_key == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model     = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model     = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    ckpt_path = f"{MODELS_DIR}/{model_key}_best.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    embedding_layer = get_embedding_layer(model, model_key)
    forward_fn      = make_forward_fn(model, model_key)

    # LayerIntegratedGradients applies IG at the embedding layer
    lig = LayerIntegratedGradients(forward_fn, embedding_layer)

    records = []

    for _, row in test_df.iterrows():
        text  = row["rebuttal_text"]
        label = int(row["label"])
        batch = row.get("batch", "")

        # Tokenize this single sample
        enc = tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Baseline: all PAD tokens (represents "no information")
        baseline_ids = torch.zeros_like(input_ids)

        # Compute attributions
        # n_steps controls accuracy: more steps = better approximation of the integral
        attributions, _ = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask,),
            n_steps=N_STEPS,
            return_convergence_delta=True,
        )

        # attributions shape: (1, seq_len, embedding_dim)
        # Summarise across embedding dimension → one score per token
        attr_scores = attributions.squeeze(0).sum(dim=-1).detach().cpu().numpy()

        # Decode token strings
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())

        # Store one record per token (skipping padding)
        for token, score in zip(tokens, attr_scores):
            if token in (tokenizer.pad_token, "[PAD]", "<pad>"):
                break
            records.append({
                "rebuttal_id":  row["rebuttal_id"],
                "batch":        batch,
                "true_label":   label,
                "token":        token,
                "attribution":  float(score),
                "abs_attribution": abs(float(score)),
            })

    attr_df = pd.DataFrame(records)
    out_path = f"{RESULTS_DIR}/ig_attributions_{model_key}.csv"
    attr_df.to_csv(out_path, index=False)
    print(f"  Attributions saved: {out_path}  ({len(attr_df)} token records)")
    return attr_df

# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS: summarise attributions by batch
# ═══════════════════════════════════════════════════════════════════════════════
def summarise_by_batch(attr_df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    """
    For each batch (A/B/C/D), compute the mean absolute attribution.
    Higher = tokens in rebuttals from that batch are more influential.
    """
    summary = (
        attr_df.groupby("batch")["abs_attribution"]
        .mean()
        .reset_index()
        .rename(columns={"abs_attribution": f"mean_attr_{model_key}"})
    )
    return summary


def top_tokens(attr_df: pd.DataFrame, model_key: str, n: int = 20) -> pd.DataFrame:
    """Return the n tokens with the highest mean absolute attribution across all samples."""
    # Ignore special tokens
    special = {"[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>", "Ġ"}
    filtered = attr_df[~attr_df["token"].isin(special)]

    top = (
        filtered.groupby("token")["abs_attribution"]
        .mean()
        .sort_values(ascending=False)
        .head(n)
        .reset_index()
    )
    top["model"] = model_key
    return top

# ═══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_batch_heatmap(batch_pivot: pd.DataFrame) -> None:
    """
    Heatmap: rows = batches, columns = models.
    Color intensity = mean absolute attribution.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        batch_pivot, annot=True, fmt=".4f", cmap="YlOrRd",
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Mean Token Attribution by Batch × Model\n(higher = more influential)", fontsize=13)
    ax.set_xlabel("Model")
    ax.set_ylabel("Batch")
    plt.tight_layout()
    path = f"{RESULTS_DIR}/ig_batch_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_top_tokens(all_top_tokens: list[pd.DataFrame]) -> None:
    """Horizontal bar charts: top 15 tokens per model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    colors = {"bert": "#e74c3c", "roberta": "#2ecc71", "distilbert": "#3498db"}

    for ax, top_df in zip(axes, all_top_tokens):
        model_key = top_df["model"].iloc[0]
        top15 = top_df.head(15)
        ax.barh(
            top15["token"][::-1],
            top15["abs_attribution"][::-1],
            color=colors[model_key],
        )
        ax.set_title(f"{model_key.upper()} — Top Tokens", fontsize=12)
        ax.set_xlabel("Mean |Attribution|")
        ax.tick_params(axis="y", labelsize=9)

    plt.suptitle("Most Influential Tokens for FLIP Prediction", fontsize=14)
    plt.tight_layout()
    path = f"{RESULTS_DIR}/ig_token_barplots.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    all_batch_summaries = []
    all_top_token_dfs   = []

    for model_key, model_name in MODELS.items():
        attr_df = compute_ig_attributions(model_key, model_name, test_df)

        batch_summary = summarise_by_batch(attr_df, model_key)
        all_batch_summaries.append(batch_summary)

        top_df = top_tokens(attr_df, model_key)
        top_df.to_csv(f"{RESULTS_DIR}/ig_top_tokens_{model_key}.csv", index=False)
        all_top_token_dfs.append(top_df)

    # ── Merge batch summaries → pivot table (batch × model) ───────────────────
    merged = all_batch_summaries[0]
    for s in all_batch_summaries[1:]:
        merged = merged.merge(s, on="batch", how="outer")
    merged.to_csv(f"{RESULTS_DIR}/ig_batch_summary.csv", index=False)
    print(f"\nBatch summary:\n{merged.to_string(index=False)}")

    # Pivot for heatmap: index=batch, columns=model
    model_cols = [c for c in merged.columns if c.startswith("mean_attr_")]
    pivot = merged.set_index("batch")[model_cols]
    pivot.columns = [c.replace("mean_attr_", "").upper() for c in pivot.columns]

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_batch_heatmap(pivot)
    plot_top_tokens(all_top_token_dfs)

    print(f"\n{'═' * 60}")
    print("Integrated Gradients complete.")
    print(f"All outputs in: {RESULTS_DIR}/")
    print(f"{'═' * 60}")
    print()
    print("HOW TO INTERPRET:")
    print("  ig_batch_heatmap.png   → which batch style triggers sycophancy most")
    print("  ig_token_barplots.png  → which specific words drive FLIP predictions")
    print("  ig_batch_summary.csv   → raw numbers for your paper's Table 3")
