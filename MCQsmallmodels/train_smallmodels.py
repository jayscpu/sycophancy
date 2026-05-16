"""
Sycophancy Flip Prediction — Training Pipeline (small models)
================================================
One script for all three small models: Logistic Regression, XGBoost, SVM (RBF kernel).

Mirrors the structure of train.py (BERT/RoBERTa/DistilBERT) but uses
hand-crafted linguistic features instead of token embeddings, so the
output coefficients / feature importances are directly interpretable.

Input:  Merged CSV with columns: question_id, rebuttal_text, label (FLIP/HOLD)
Output: Evaluation metrics, confusion matrix, feature-importance ranking

Setup:
    pip install pandas numpy scikit-learn xgboost vaderSentiment textstat matplotlib

Usage:
    python train_smallmodels.py --model logreg  --input merged_data.csv --output results/logreg
    python train_smallmodels.py --model xgboost --input merged_data.csv --output results/xgboost
    python train_smallmodels.py --model svm-rbf --input merged_data.csv --output results/svm

Works on Google Colab or local machine (CPU only — no GPU required).
"""

import os
import json
import random
import argparse
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb
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
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from datetime import datetime


# ─────────────────────────────────────────────
# LEXICONS for hand-crafted linguistic features
# ─────────────────────────────────────────────

HEDGING = {
    "maybe", "perhaps", "might", "could be", "i think", "i believe", "i guess",
    "not sure", "i'm not sure", "seems", "appears", "kind of", "sort of",
    "possibly", "probably", "i suppose",
}
CERTAINTY = {
    "definitely", "absolutely", "100%", "certainly", "clearly", "obviously",
    "without doubt", "no doubt", "for sure", "undoubtedly", "indeed", "of course",
}
POLITENESS = {
    "please", "could you", "would you mind", "thank you", "thanks", "sorry",
    "appreciate", "kindly", "if you don't mind",
}
AGGRESSION = {
    "wrong", "incorrect", "you're wrong", "that's wrong", "no,", "stop",
    "duh", "seriously", "come on", "ridiculous", "nonsense",
}
SOURCES = {
    "study", "studies", "research", "scientist", "scientists", "expert", "experts",
    "paper", "journal", "article", "source", "i read", "i heard", "wikipedia",
    "google", "internet", "online", "documentary", "book",
}
ANECDOTE = {
    "my teacher", "my professor", "in school", "in college", "when i was",
    "my friend", "my mom", "my dad", "my parents", "i learned", "i remember",
    "grandfather", "grandmother",
}
AUTHORITY = {
    "experts say", "scientists agree", "everyone knows", "commonly known",
    "well known", "established fact", "proven", "well-documented",
}
COMMAND = {
    "check your facts", "look it up", "verify", "reconsider", "think again",
    "fact-check", "double check", "google it", "search it",
}


def _count_phrases(text_lower: str, lexicon: set[str]) -> int:
    """Count how many lexicon phrases appear in the text (substring match)."""
    return sum(1 for phrase in lexicon if phrase in text_lower)


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
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

_VADER = SentimentIntensityAnalyzer()


def extract_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a numeric feature matrix from a DataFrame of rebuttals.
    Returns (features_df, feature_names).

    Feature groups:
      style       — length, punctuation, casing
      readability — Flesch reading ease / grade
      lexicon     — counts of hedging, certainty, politeness, aggression,
                    source-citation, personal-anecdote, authority, command phrases
      sentiment   — VADER pos/neg/neu/compound
      meta        — one-hot of batch (A/B/C/D) when present
    """
    feats = pd.DataFrame(index=df.index)
    text = df["rebuttal_text"].astype(str)
    text_lower = text.str.lower()

    # — Style / length
    feats["len_chars"] = text.str.len()
    feats["len_words"] = text.str.split().str.len()
    feats["n_sentences"] = text.apply(lambda s: max(1, len(re.findall(r"[.!?]+", s))))
    feats["avg_word_len"] = feats["len_chars"] / feats["len_words"].clip(lower=1)
    feats["n_question_marks"] = text.str.count(r"\?")
    feats["n_exclamations"] = text.str.count("!")
    feats["n_commas"] = text.str.count(",")
    feats["caps_ratio"] = text.apply(lambda s: sum(1 for c in s if c.isupper()) / max(1, len(s)))
    feats["starts_lower"] = text.apply(lambda s: int(bool(s) and s[0].islower()))
    feats["has_digit"] = text.str.contains(r"\d", regex=True).astype(int)
    feats["ends_with_q"] = text.str.rstrip().str.endswith("?").astype(int)

    # — Readability
    feats["flesch_reading_ease"] = text.apply(lambda s: textstat.flesch_reading_ease(s) if s.strip() else 0.0)
    feats["flesch_grade"] = text.apply(lambda s: textstat.flesch_kincaid_grade(s) if s.strip() else 0.0)

    # — Lexicon counts (one feature per category)
    feats["n_hedging"] = text_lower.apply(lambda s: _count_phrases(s, HEDGING))
    feats["n_certainty"] = text_lower.apply(lambda s: _count_phrases(s, CERTAINTY))
    feats["n_politeness"] = text_lower.apply(lambda s: _count_phrases(s, POLITENESS))
    feats["n_aggression"] = text_lower.apply(lambda s: _count_phrases(s, AGGRESSION))
    feats["n_sources"] = text_lower.apply(lambda s: _count_phrases(s, SOURCES))
    feats["n_anecdote"] = text_lower.apply(lambda s: _count_phrases(s, ANECDOTE))
    feats["n_authority"] = text_lower.apply(lambda s: _count_phrases(s, AUTHORITY))
    feats["n_command"] = text_lower.apply(lambda s: _count_phrases(s, COMMAND))

    # — Sentiment (VADER)
    vader_scores = text.apply(lambda s: _VADER.polarity_scores(s))
    feats["vader_pos"] = vader_scores.apply(lambda d: d["pos"])
    feats["vader_neg"] = vader_scores.apply(lambda d: d["neg"])
    feats["vader_neu"] = vader_scores.apply(lambda d: d["neu"])
    feats["vader_compound"] = vader_scores.apply(lambda d: d["compound"])

    # — Meta: one-hot of batch
    if "batch" in df.columns:
        for b in ["A", "B", "C", "D"]:
            feats[f"batch_{b}"] = (df["batch"].astype(str).str.upper() == b).astype(int)

    # — MCQ-only: does the rebuttal contain the targeted wrong_answer verbatim?
    if "wrong_answer" in df.columns:
        def _contains_wa(row):
            wa = str(row.get("wrong_answer") or "").lower().strip()
            if not wa:
                return 0
            pattern = r"\b" + re.escape(wa) + r"\b"
            return int(bool(re.search(pattern, str(row["rebuttal_text"]).lower())))
        feats["contains_wa"] = df.apply(_contains_wa, axis=1)

    return feats, feats.columns.tolist()


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: np.ndarray,
    output_dir: str = "checkpoints",
):
    """Fit one of: logreg, xgboost, svm-rbf. Returns (clf, training_log).

    Uses class_weights to handle class imbalance. The val split is kept
    around so XGBoost can use it for early stopping (and so the training
    log mirrors the BERT script's per-epoch shape).
    """
    os.makedirs(output_dir, exist_ok=True)
    training_log = []

    # sklearn-style class_weight dict
    cw = {0: float(class_weights[0]), 1: float(class_weights[1])}

    if model_name == "logreg":
        clf = LogisticRegression(
            C=1.0, max_iter=2000, class_weight=cw, random_state=42, n_jobs=-1
        )
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)
        training_log.append({
            "epoch": 1,
            "train_f1": float(f1_score(y_train, train_pred, zero_division=0)),
            "val_f1":   float(f1_score(y_val, val_pred, zero_division=0)),
        })

    elif model_name == "xgboost":
        # scale_pos_weight handles imbalance for xgboost
        spw = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
        clf = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=20,
        )
        clf.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        if hasattr(clf, "evals_result_"):
            evals = clf.evals_result()
            train_logloss = evals["validation_0"]["logloss"]
            val_logloss = evals["validation_1"]["logloss"]
            for i, (tr, va) in enumerate(zip(train_logloss, val_logloss)):
                training_log.append({
                    "epoch": i + 1,
                    "train_logloss": float(tr),
                    "val_logloss": float(va),
                })
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)
        training_log.append({
            "epoch": "final",
            "train_f1": float(f1_score(y_train, train_pred, zero_division=0)),
            "val_f1":   float(f1_score(y_val, val_pred, zero_division=0)),
            "best_iteration": int(getattr(clf, "best_iteration", -1)),
        })

    elif model_name == "svm-rbf":
        clf = SVC(
            C=1.0, kernel="rbf", gamma="scale", class_weight=cw,
            probability=True, random_state=42,
        )
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        val_pred = clf.predict(X_val)
        training_log.append({
            "epoch": 1,
            "train_f1": float(f1_score(y_train, train_pred, zero_division=0)),
            "val_f1":   float(f1_score(y_val, val_pred, zero_division=0)),
        })

    else:
        raise ValueError(f"Unknown model: {model_name}")

    last = training_log[-1]
    print(f"  Train F1: {last['train_f1']:.4f} | Val F1: {last['val_f1']:.4f}")

    return clf, training_log


def evaluate(clf, X, y):
    """Evaluate fitted classifier on a dataset. Returns F1, predictions, labels."""
    preds = clf.predict(X)
    f1 = f1_score(y, preds, zero_division=0)
    return f1, preds, y


# ─────────────────────────────────────────────
# TEST EVALUATION WITH CONFIDENCE INTERVALS
# ─────────────────────────────────────────────

def evaluate_test(clf, X_test, y_test, n_bootstrap=1000):
    """Full evaluation on test set with bootstrapped confidence intervals."""
    all_preds = clf.predict(X_test)
    all_probs = clf.predict_proba(X_test)[:, 1]
    all_labels = np.asarray(y_test)

    all_preds = np.asarray(all_preds)
    all_probs = np.asarray(all_probs)

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

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    # ROC curve data
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=["HOLD", "FLIP"], zero_division=0)

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
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────

def run_feature_importance(model_name, clf, X_test, y_test, feature_names, n_repeats=20):
    """
    Extract model-appropriate feature importance — the analogue of
    Integrated Gradients for non-neural models.

      logreg  — standardized coefficients (signed: positive => predicts FLIP)
      xgboost — gain-based feature_importances_ (unsigned)
      svm-rbf — permutation importance on test set (unsigned)

    Returns a list of dicts ranked by abs(importance), highest first.
    """
    print(f"\nComputing feature importance for {model_name}...")

    if model_name == "logreg":
        coefs = clf.coef_[0]  # shape (n_features,)
        records = [
            {"feature": name, "importance": float(coef), "direction": "FLIP" if coef > 0 else "HOLD"}
            for name, coef in zip(feature_names, coefs)
        ]

    elif model_name == "xgboost":
        importances = clf.feature_importances_
        records = [
            {"feature": name, "importance": float(imp), "direction": None}
            for name, imp in zip(feature_names, importances)
        ]

    elif model_name == "svm-rbf":
        # Permutation importance — slow but model-agnostic
        result = permutation_importance(
            clf, X_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1
        )
        records = [
            {
                "feature": name,
                "importance": float(mean),
                "importance_std": float(std),
                "direction": None,
            }
            for name, mean, std in zip(feature_names, result.importances_mean, result.importances_std)
        ]

    else:
        raise ValueError(f"Unknown model: {model_name}")

    records.sort(key=lambda r: abs(r["importance"]), reverse=True)
    print(f"  ✓ Feature importance complete")
    return records


def aggregate_importance(records):
    """Mirror of aggregate_attributions in train.py — returns
    [(name, {"mean": ..., "count": ...}), ...]."""
    return [
        (r["feature"], {"mean": r["importance"], "count": 1,
                        **{k: v for k, v in r.items() if k not in ("feature", "importance")}})
        for r in records
    ]


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train sycophancy flip classifier (small models)")
    parser.add_argument("--model", required=True,
                        choices=["logreg", "xgboost", "svm-rbf"],
                        help="Classifier choice")
    parser.add_argument("--input", required=True, help="Path to merged CSV")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--skip-ig", action="store_true", help="Skip feature-importance step")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Reproducibility: seed all sources of randomness
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Device: cpu (small models — no GPU needed)")

    # ── Step 1: Load and split ──
    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"{'='*50}")

    df_train, df_val, df_test = load_and_split(args.input)

    # ── Step 2: Extract features ──
    print(f"\nExtracting features...")
    train_feats, feature_names = extract_features(df_train)
    val_feats, _ = extract_features(df_val)
    test_feats, _ = extract_features(df_test)
    print(f"  Feature count: {len(feature_names)}")

    # Align columns (in case some batches are missing in a split)
    val_feats = val_feats.reindex(columns=feature_names, fill_value=0)
    test_feats = test_feats.reindex(columns=feature_names, fill_value=0)

    # Standardize for linear / kernel models
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feats.values)
    X_val = scaler.transform(val_feats.values)
    X_test = scaler.transform(test_feats.values)

    y_train = df_train["label_encoded"].to_numpy()
    y_val = df_val["label_encoded"].to_numpy()
    y_test = df_test["label_encoded"].to_numpy()

    # ── Step 3: Compute class weights ──
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    print(f"\nClass weights: HOLD={class_weights[0]:.3f}, FLIP={class_weights[1]:.3f}")

    print(f"\nTotal features: {len(feature_names)}")

    # ── Step 5: Train ──
    print(f"\nTraining...")
    clf, training_log = train_model(
        model_name=args.model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        class_weights=class_weights,
        output_dir=os.path.join(args.output, "checkpoints"),
    )

    # ── Step 6: Evaluate on test set ──
    print(f"\nEvaluating on test set...")
    test_results = evaluate_test(clf, X_test, y_test)

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

    # ── Step 7: Feature importance ──
    importance_records = None
    top_tokens = None
    if not args.skip_ig:
        importance_records = run_feature_importance(
            args.model, clf, X_test, y_test, feature_names,
        )
        top_tokens = aggregate_importance(importance_records)

        print(f"\nTop 20 features driving predictions:")
        for i, (name, stats) in enumerate(top_tokens[:20]):
            print(f"  {i+1:>2}. {name:<25} importance: {stats['mean']:.4f}")

    # ── Step 8: Save everything ──
    results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "device": "cpu",
        "hyperparameters": {
            "logreg":  {"C": 1.0, "class_weight": "balanced", "max_iter": 2000},
            "xgboost": {"n_estimators": 500, "learning_rate": 0.1, "max_depth": 5,
                        "subsample": 0.9, "colsample_bytree": 0.9,
                        "early_stopping_rounds": 20},
            "svm-rbf": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
        }[args.model],
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

    # Save feature-importance records separately (mirrors the IG output file pattern)
    if importance_records:
        ig_path = os.path.join(args.output, "ig_attributions.json")
        with open(ig_path, "w") as f:
            json.dump(importance_records, f, indent=2)
        print(f"✓ Feature-importance records saved to {ig_path}")

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
