"""
Phase 0 — Baseline comparisons on the same 5-fold splits used by
5fold-train-MCQ-noIG.py, so the BERT/RoBERTa/DistilBERT numbers can be
compared apples-to-apples against:

  1. Majority class (predict HOLD always)
  2. Random with class prior
  3. Logistic regression on hand-engineered features:
       - rebuttal length (chars, words)
       - caps ratio, has '?', has '!', starts lowercase, has digit

Metadata columns (batch, teammate, contains_wrong_answer) were intentionally
removed: they encode collection provenance, not rebuttal linguistics, and
let the classifier cheat on generator identity / wrong-answer leakage.

The filter chain, split strategy, and metric definitions are replicated
exactly from 5fold-train-MCQ-noIG.py so any difference in F1 is attributable
to the model, not to a different test set.

Usage:
    python phase0_baselines.py \\
        --input ../MCQ/MCQ_MERGED_DATASET.json \\
        --output results/phase0_baselines.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


SEED = 42


# ─────────────────────────────────────────────
# DATA LOAD + FILTER (mirrors 5fold-train-MCQ-noIG.py)
# ─────────────────────────────────────────────

def load_and_filter(json_path: str) -> pd.DataFrame:
    with open(json_path, "r") as f:
        data = json.load(f)
    challenges = data.get("challenges", [])

    rows = []
    for q in challenges:
        question = q.get("question", "")
        rows.append({
            "question_id": hashlib.md5(question.strip().lower().encode()).hexdigest()[:8],
            "question": question,
            "rebuttal_text": q.get("rebuttal_text", ""),
            "label": str(q.get("label", "")).upper().strip(),
            "batch": q.get("batch"),
            "teammate": q.get("teammate"),
            "wrong_answer": q.get("wrong_answer", ""),
            "initial_answer": q.get("initial_answer", ""),
        })
    df = pd.DataFrame(rows)

    before = len(df)
    df = df.dropna(subset=["rebuttal_text"])
    df = df[df["rebuttal_text"].astype(str).str.strip() != ""]
    df = df[df["label"].isin({"FLIP", "HOLD"})].copy()
    print(f"After FLIP/HOLD filter: {len(df)} rows (dropped {before - len(df)})")

    df["label_encoded"] = (df["label"] == "FLIP").astype(int)

    # Robust/brittle filter — same window as 5fold-train-MCQ-noIG.py
    flip_rate = df.groupby("question_id")["label_encoded"].mean()
    mixed = flip_rate[(flip_rate >= 0.1) & (flip_rate <= 0.9)].index
    n_robust = int((flip_rate < 0.1).sum())
    n_brittle = int((flip_rate > 0.9).sum())
    df = df[df["question_id"].isin(mixed)].reset_index(drop=True)
    print(f"After robust/brittle filter: {len(df)} rows from {df['question_id'].nunique()} questions")
    print(f"  Dropped: {n_robust} robust + {n_brittle} brittle")
    print(f"  Class balance: FLIP={df['label_encoded'].mean():.1%}, "
          f"HOLD={1 - df['label_encoded'].mean():.1%}")
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    text = df["rebuttal_text"].astype(str)

    feats["len_chars"] = text.str.len()
    feats["len_words"] = text.str.split().str.len()
    feats["caps_ratio"] = text.apply(
        lambda s: sum(1 for c in s if c.isupper()) / max(1, len(s))
    )
    feats["has_q"] = text.str.contains(r"\?", regex=True).astype(int)
    feats["has_excl"] = text.str.contains("!", regex=False).astype(int)
    feats["starts_lower"] = text.apply(lambda s: int(bool(s) and s[0].islower()))
    feats["has_digit"] = text.str.contains(r"\d", regex=True).astype(int)

    return feats


# ─────────────────────────────────────────────
# CV
# ─────────────────────────────────────────────

def fold_splits(df: pd.DataFrame, n_folds: int = 5):
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    return list(sgkf.split(df, df["label_encoded"], groups=df["question_id"]))


def metrics(y_true, y_pred, y_prob=None):
    out = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
    return out


def summarize(per_fold: list[dict]) -> dict:
    keys = ["f1", "precision", "recall", "accuracy", "roc_auc"]
    summary = {}
    for k in keys:
        vals = [d[k] for d in per_fold if d[k] is not None]
        if not vals:
            summary[k] = {"mean": None, "std": None, "per_fold": [d.get(k) for d in per_fold]}
            continue
        summary[k] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "per_fold": [d.get(k) for d in per_fold],
        }
    return summary


# ─────────────────────────────────────────────
# BASELINES
# ─────────────────────────────────────────────

def baseline_majority(df: pd.DataFrame, splits) -> dict:
    """Predict HOLD (0) for everything."""
    per_fold = []
    for fold, (_, test_idx) in enumerate(splits):
        y_true = df.iloc[test_idx]["label_encoded"].values
        y_pred = np.zeros_like(y_true)
        per_fold.append(metrics(y_true, y_pred, y_prob=None))
    return summarize(per_fold)


def baseline_random_prior(df: pd.DataFrame, splits) -> dict:
    """Predict 1 with prob = train flip rate (Bernoulli)."""
    rng = np.random.RandomState(SEED)
    per_fold = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        p = df.iloc[train_idx]["label_encoded"].mean()
        y_true = df.iloc[test_idx]["label_encoded"].values
        # Probabilistic prediction: sample from Bernoulli(p) per test sample
        y_prob = np.full(len(test_idx), p)
        y_pred = (rng.rand(len(test_idx)) < p).astype(int)
        per_fold.append(metrics(y_true, y_pred, y_prob=y_prob))
    return summarize(per_fold)


def baseline_logreg(df: pd.DataFrame, splits, feat_cols: list[str]) -> dict:
    """LogReg on hand-engineered features."""
    feats = make_features(df)
    X_all = feats[feat_cols].values
    y_all = df["label_encoded"].values

    per_fold = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        per_fold.append(metrics(y_test, y_pred, y_prob=y_prob))
    return summarize(per_fold)


def baseline_tfidf_logreg(df: pd.DataFrame, splits) -> dict:
    """LogReg on TF-IDF of rebuttal text — a 'just the words' baseline."""
    per_fold = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        train_texts = df.iloc[train_idx]["rebuttal_text"].values
        test_texts = df.iloc[test_idx]["rebuttal_text"].values
        y_train = df.iloc[train_idx]["label_encoded"].values
        y_test = df.iloc[test_idx]["label_encoded"].values

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=20000)
        X_train = vec.fit_transform(train_texts)
        X_test = vec.transform(test_texts)

        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        per_fold.append(metrics(y_test, y_pred, y_prob=y_prob))
    return summarize(per_fold)


def baseline_combined(df: pd.DataFrame, splits, feat_cols: list[str]) -> dict:
    """LogReg on [hand features ++ TF-IDF]."""
    from scipy.sparse import hstack, csr_matrix

    feats = make_features(df)
    X_hand_all = feats[feat_cols].values
    y_all = df["label_encoded"].values

    per_fold = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        scaler = StandardScaler()
        X_hand_tr = scaler.fit_transform(X_hand_all[train_idx])
        X_hand_te = scaler.transform(X_hand_all[test_idx])

        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, max_features=20000)
        X_tfidf_tr = vec.fit_transform(df.iloc[train_idx]["rebuttal_text"].values)
        X_tfidf_te = vec.transform(df.iloc[test_idx]["rebuttal_text"].values)

        X_tr = hstack([X_tfidf_tr, csr_matrix(X_hand_tr)])
        X_te = hstack([X_tfidf_te, csr_matrix(X_hand_te)])

        y_tr = y_all[train_idx]
        y_te = y_all[test_idx]

        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=SEED)
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        per_fold.append(metrics(y_te, y_pred, y_prob=y_prob))
    return summarize(per_fold)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../MCQ/MCQ_MERGED_DATASET.json")
    parser.add_argument("--output", default="results/phase0_baselines.json")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = load_and_filter(args.input)
    splits = fold_splits(df, args.n_folds)

    feat_cols = [
        "len_chars", "len_words", "caps_ratio", "has_q", "has_excl",
        "starts_lower", "has_digit",
    ]

    print(f"\n{'='*60}")
    print(f"Baselines on {args.n_folds}-fold StratifiedGroupKFold (seed={SEED})")
    print(f"Total samples: {len(df)}  |  Questions: {df['question_id'].nunique()}")
    print(f"{'='*60}\n")

    results = {
        "n_folds": args.n_folds,
        "total_samples": len(df),
        "n_questions": int(df["question_id"].nunique()),
        "class_balance_flip": float(df["label_encoded"].mean()),
        "features_used_for_logreg": feat_cols,
        "baselines": {},
    }

    print("Running majority-class baseline...")
    results["baselines"]["majority_hold"] = baseline_majority(df, splits)

    print("Running random-prior baseline...")
    results["baselines"]["random_prior"] = baseline_random_prior(df, splits)

    print("Running hand-feature LogReg baseline...")
    results["baselines"]["logreg_hand_features"] = baseline_logreg(df, splits, feat_cols)

    print("Running TF-IDF LogReg baseline...")
    results["baselines"]["logreg_tfidf"] = baseline_tfidf_logreg(df, splits)

    print("Running combined (hand + TF-IDF) LogReg baseline...")
    results["baselines"]["logreg_combined"] = baseline_combined(df, splits, feat_cols)

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'Baseline':<28} {'F1':>10} {'Prec':>8} {'Rec':>8} {'AUC':>8}")
    print(f"{'-'*60}")
    for name, summary in results["baselines"].items():
        f1 = summary["f1"]
        p = summary["precision"]
        r = summary["recall"]
        auc = summary["roc_auc"]
        f1_str = f"{f1['mean']:.3f}±{f1['std']:.3f}" if f1["mean"] is not None else "n/a"
        p_str = f"{p['mean']:.3f}" if p["mean"] is not None else "n/a"
        r_str = f"{r['mean']:.3f}" if r["mean"] is not None else "n/a"
        auc_str = f"{auc['mean']:.3f}" if auc["mean"] is not None else "n/a"
        print(f"{name:<28} {f1_str:>10} {p_str:>8} {r_str:>8} {auc_str:>8}")
    print(f"{'='*60}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {args.output}")


if __name__ == "__main__":
    main()
