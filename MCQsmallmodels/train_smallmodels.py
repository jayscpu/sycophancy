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
    pip install pandas numpy scikit-learn xgboost vaderSentiment textstat matplotlib \
                spacy sentence-transformers
    python -m spacy download en_core_web_sm

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
import spacy
from sentence_transformers import SentenceTransformer
import optuna
import shap
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

# Modal verbs partitioned by epistemic certainty (Palmer 2001, Biber et al. 1999)
MODALS_CERTAIN = {"must", "will", "shall", "would"}
MODALS_UNCERTAIN = {"may", "might", "could", "can", "should"}
MODALS_ALL = MODALS_CERTAIN | MODALS_UNCERTAIN

NOMINALISATION_SUFFIXES = ("tion", "ment", "ness", "ity", "ance", "ence", "sion", "ship")


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
_NLP = None
_SBERT = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm", disable=["ner"])
    return _NLP


def _get_sbert():
    global _SBERT
    if _SBERT is None:
        _SBERT = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT


def _cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity for two (n, d) matrices."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return np.sum(a_norm * b_norm, axis=1)


def _morph_syntax_features(docs: list) -> pd.DataFrame:
    """Per-doc morphological + syntactic feature dict, then stacked."""
    rows = []
    for doc in docs:
        n_tokens = max(1, sum(1 for t in doc if not t.is_space))
        n_words = max(1, sum(1 for t in doc if t.is_alpha))
        sents = list(doc.sents) or [doc[:]]
        n_sents = max(1, len(sents))
        sent_lens = [sum(1 for t in s if not t.is_space) for s in sents]

        modals_all = modals_certain = modals_uncertain = 0
        past = present = 0
        nominalisations = 0
        n_nouns = 0
        negations = 0
        passive_aux = passive_subj = 0
        subord = 0
        conj = 0
        n_verbs = 0  # finite verbs / clause heads
        for t in doc:
            lemma = t.lemma_.lower()
            if t.pos_ == "AUX" or t.tag_ in {"MD"}:
                if lemma in MODALS_ALL:
                    modals_all += 1
                    if lemma in MODALS_CERTAIN:
                        modals_certain += 1
                    elif lemma in MODALS_UNCERTAIN:
                        modals_uncertain += 1
            if t.pos_ in {"VERB", "AUX"}:
                tense = t.morph.get("Tense")
                if "Past" in tense:
                    past += 1
                elif "Pres" in tense:
                    present += 1
                if t.tag_ in {"VBD", "VBP", "VBZ", "MD"}:
                    n_verbs += 1
            if t.pos_ == "NOUN":
                n_nouns += 1
                if lemma.endswith(NOMINALISATION_SUFFIXES):
                    nominalisations += 1
            if t.dep_ == "neg":
                negations += 1
            if t.dep_ == "auxpass":
                passive_aux += 1
            if t.dep_ in {"nsubjpass", "csubjpass"}:
                passive_subj += 1
            if t.dep_ in {"ccomp", "xcomp", "advcl", "acl", "acl:relcl"}:
                subord += 1
            if t.dep_ in {"cc", "conj"}:
                conj += 1

        depths = []
        for t in doc:
            d = 0
            cur = t
            while cur.head.i != cur.i and d < 50:
                d += 1
                cur = cur.head
            depths.append(d)
        mean_depth = float(np.mean(depths)) if depths else 0.0

        rows.append({
            # morphological
            "n_modals_all": modals_all,
            "n_modals_certain": modals_certain,
            "n_modals_uncertain": modals_uncertain,
            "modal_uncertainty_ratio": modals_uncertain / max(1, modals_all),
            "n_past_tense": past,
            "n_present_tense": present,
            "past_pres_ratio": past / max(1, past + present),
            "n_nominalisations": nominalisations,
            "nominalisation_rate": nominalisations / max(1, n_nouns),
            "n_negations": negations,
            # syntactic
            "n_clauses": n_verbs,
            "clauses_per_sentence": n_verbs / n_sents,
            "mean_parse_depth": mean_depth,
            "n_passive_aux": passive_aux,
            "n_passive_subj": passive_subj,
            "is_passive": int(passive_aux > 0 or passive_subj > 0),
            "n_subord_clauses": subord,
            "subord_rate": subord / n_sents,
            "n_conjunctions": conj,
            "mean_sentence_len": float(np.mean(sent_lens)),
            "sentence_len_var": float(np.var(sent_lens)) if len(sent_lens) > 1 else 0.0,
            # normalisations for cross-length comparability
            "modal_density": modals_all / n_words,
        })
    return pd.DataFrame(rows)


def precompute_nlp_and_embeddings(df: pd.DataFrame):
    """Run spaCy + sbert once over the full df. Returns (docs, sim_correct, sim_wrong)
    aligned to df.index. Pass these into extract_features to avoid re-running per split."""
    text_list = df["rebuttal_text"].astype(str).tolist()
    nlp = _get_nlp()
    docs = list(nlp.pipe(text_list, batch_size=64))

    sim_correct = sim_wrong = None
    if "correct_answer" in df.columns and "wrong_answer" in df.columns:
        sbert = _get_sbert()
        reb_emb = sbert.encode(text_list, batch_size=64, show_progress_bar=False,
                               convert_to_numpy=True)
        cor_emb = sbert.encode(df["correct_answer"].fillna("").astype(str).tolist(),
                               batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        wro_emb = sbert.encode(df["wrong_answer"].fillna("").astype(str).tolist(),
                               batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        sim_correct = _cos_sim(reb_emb, cor_emb)
        sim_wrong = _cos_sim(reb_emb, wro_emb)
    return docs, sim_correct, sim_wrong


def extract_features(
    df: pd.DataFrame,
    docs: list | None = None,
    sim_correct: np.ndarray | None = None,
    sim_wrong: np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a numeric feature matrix from a DataFrame of rebuttals.
    Returns (features_df, feature_names).

    Six linguistic levels per the proposal:
      orthographic — length, punctuation, casing
      lexical      — readability, hedging/certainty/politeness/aggression/source/anecdote/
                     authority/command lexicon counts
      morphological — modal verbs (epistemic certainty split), tense, nominalisation,
                      negation                            (spaCy)
      syntactic    — clause count, parse depth, passive voice, subordination,
                     sentence-length variance             (spaCy)
      semantic     — VADER sentiment + cosine similarity of rebuttal to the
                     correct_answer and wrong_answer      (sentence-transformers)
      pragmatic    — covered by lexical pragmatic markers above

    Metadata columns (batch, teammate) are intentionally excluded: they encode
    collection provenance, not properties of the rebuttal.

    Pre-compute `docs` / `sim_*` once on the full df with precompute_nlp_and_embeddings()
    and pass them in to avoid running spaCy + sbert separately for each CV split.
    """
    if docs is None or sim_correct is None or sim_wrong is None:
        docs, sim_correct, sim_wrong = precompute_nlp_and_embeddings(df)

    feats = pd.DataFrame(index=df.index)
    text = df["rebuttal_text"].astype(str)
    text_lower = text.str.lower()

    # — Orthographic / style / length
    feats["len_chars"] = text.str.len()
    feats["len_words"] = text.str.split().str.len()
    # Sentence count from spaCy (shared with syntactic features for consistency)
    feats["n_sentences"] = [max(1, len(list(d.sents))) for d in docs]
    # Average word length: characters per word excluding whitespace
    feats["avg_word_len"] = (
        text.str.replace(r"\s+", "", regex=True).str.len()
        / feats["len_words"].clip(lower=1)
    )
    feats["n_question_marks"] = text.str.count(r"\?")
    feats["n_exclamations"] = text.str.count("!")
    feats["n_commas"] = text.str.count(",")
    feats["caps_ratio"] = text.apply(lambda s: sum(1 for c in s if c.isupper()) / max(1, len(s)))
    feats["starts_lower"] = text.apply(lambda s: int(bool(s) and s[0].islower()))
    feats["has_digit"] = text.str.contains(r"\d", regex=True).astype(int)
    feats["ends_with_q"] = text.str.rstrip().str.endswith("?").astype(int)

    # — Lexical: readability
    feats["flesch_reading_ease"] = text.apply(lambda s: textstat.flesch_reading_ease(s) if s.strip() else 0.0)
    feats["flesch_grade"] = text.apply(lambda s: textstat.flesch_kincaid_grade(s) if s.strip() else 0.0)

    # — Lexical / pragmatic: lexicon counts
    feats["n_hedging"] = text_lower.apply(lambda s: _count_phrases(s, HEDGING))
    feats["n_certainty"] = text_lower.apply(lambda s: _count_phrases(s, CERTAINTY))
    feats["n_politeness"] = text_lower.apply(lambda s: _count_phrases(s, POLITENESS))
    feats["n_aggression"] = text_lower.apply(lambda s: _count_phrases(s, AGGRESSION))
    feats["n_sources"] = text_lower.apply(lambda s: _count_phrases(s, SOURCES))
    feats["n_anecdote"] = text_lower.apply(lambda s: _count_phrases(s, ANECDOTE))
    feats["n_authority"] = text_lower.apply(lambda s: _count_phrases(s, AUTHORITY))
    feats["n_command"] = text_lower.apply(lambda s: _count_phrases(s, COMMAND))

    # — Semantic: VADER sentiment
    vader_scores = text.apply(lambda s: _VADER.polarity_scores(s))
    feats["vader_pos"] = vader_scores.apply(lambda d: d["pos"])
    feats["vader_neg"] = vader_scores.apply(lambda d: d["neg"])
    feats["vader_neu"] = vader_scores.apply(lambda d: d["neu"])
    feats["vader_compound"] = vader_scores.apply(lambda d: d["compound"])

    # — Morphological + syntactic via spaCy (uses pre-computed docs)
    morph_syn = _morph_syntax_features(docs)
    morph_syn.index = df.index
    feats = pd.concat([feats, morph_syn], axis=1)

    # — Semantic: cosine similarity (pre-computed)
    if sim_correct is not None and sim_wrong is not None:
        feats["sim_to_correct"] = sim_correct
        feats["sim_to_wrong"] = sim_wrong
        feats["sim_margin_wrong_minus_correct"] = sim_wrong - sim_correct

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
# NESTED CV + OPTUNA + SHAP
# ─────────────────────────────────────────────

def _build_classifier(model_name: str, params: dict, y_train: np.ndarray):
    """Instantiate a fresh classifier with the given (tuned) hyperparameters
    plus fixed defaults. `params` must contain only the tuned keys —
    fixed defaults like solver/class_weight/scale_pos_weight are added here."""
    if model_name == "logreg":
        # liblinear supports both l1 and l2; ignores n_jobs (single-threaded).
        return LogisticRegression(
            **params, solver="liblinear", max_iter=2000,
            class_weight="balanced", random_state=42,
        )
    if model_name == "xgboost":
        spw = (len(y_train) - y_train.sum()) / max(1, y_train.sum())
        return xgb.XGBClassifier(
            **params, scale_pos_weight=float(spw),
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    if model_name == "svm-rbf":
        return SVC(
            **params, kernel="rbf", class_weight="balanced",
            probability=True, random_state=42,
        )
    raise ValueError(f"Unknown model: {model_name}")


def _suggest_params(trial: "optuna.Trial", model_name: str) -> dict:
    """Optuna hyperparameter search space per model — TUNED params only.
    Fixed params (solver, kernel, etc.) live in _build_classifier."""
    if model_name == "logreg":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
    if model_name == "svm-rbf":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
        }
    raise ValueError(f"Unknown model: {model_name}")


def _optuna_objective(trial, model_name, X_inner, y_inner, groups_inner, n_inner_folds=3):
    """Inner-CV mean F1 for one trial's hyperparameters.

    Scaling is fit inside each inner fold to avoid leakage.
    """
    params = _suggest_params(trial, model_name)
    inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
    f1s = []
    for tr, va in inner_cv.split(X_inner, y_inner, groups_inner):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_inner[tr])
        X_va = scaler.transform(X_inner[va])
        clf = _build_classifier(model_name, params, y_inner[tr])
        clf.fit(X_tr, y_inner[tr])
        f1s.append(f1_score(y_inner[va], clf.predict(X_va), zero_division=0))
    return float(np.mean(f1s))


def _compute_shap(model_name, clf, X_train, X_test, bg_size=100, max_test=200, seed=42):
    """SHAP attributions for class FLIP. Returns (n_explained, n_features) array.

      logreg  → LinearExplainer (exact, fast)
      xgboost → TreeExplainer (exact, fast)
      svm-rbf → KernelExplainer with subsampled background AND test set (slow)
    """
    rng = np.random.RandomState(seed)
    if model_name == "logreg":
        bg = shap.sample(X_train, min(bg_size, len(X_train)), random_state=seed)
        explainer = shap.LinearExplainer(clf, bg)
        sv = explainer.shap_values(X_test)
        X_explained = X_test
    elif model_name == "xgboost":
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_test)
        X_explained = X_test
    elif model_name == "svm-rbf":
        bg = shap.sample(X_train, min(bg_size, len(X_train)), random_state=seed)
        if len(X_test) > max_test:
            idx = rng.choice(len(X_test), max_test, replace=False)
            X_explained = X_test[idx]
        else:
            X_explained = X_test
        explainer = shap.KernelExplainer(clf.predict_proba, bg)
        sv = explainer.shap_values(X_explained, silent=True, nsamples="auto")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Normalise output shape to (n_explained, n_features) for the FLIP class
    sv = np.asarray(sv) if not isinstance(sv, list) else sv
    if isinstance(sv, list):
        # Older API: list of [class0_array, class1_array]
        sv = np.asarray(sv[1])
    elif sv.ndim == 3:
        # Newer API: (n_samples, n_features, n_classes)
        sv = sv[..., 1]
    return sv, X_explained


def run_nested_cv(
    model_name, X, y, groups, feature_names,
    n_outer_folds=5, n_inner_folds=3, n_trials=30, seed=42,
):
    """Outer StratifiedGroupKFold × Optuna-tuned inner CV + per-fold SHAP.

    For each outer fold: tune on inner CV, refit on full outer-train, evaluate
    on outer-test, compute SHAP. Aggregate metrics and mean-|SHAP| across folds.
    """
    outer_cv = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=seed)
    fold_results = []
    oof_probs = np.zeros(len(y))
    oof_preds = np.zeros(len(y), dtype=int)
    oof_assigned = np.zeros(len(y), dtype=bool)
    shap_means_abs = []  # one (n_features,) array per fold

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    for fold_i, (outer_tr, outer_te) in enumerate(
        outer_cv.split(X, y, groups), start=1
    ):
        print(f"\n── Outer fold {fold_i}/{n_outer_folds} "
              f"(train={len(outer_tr)}, test={len(outer_te)}) ──")
        X_inner = X[outer_tr]
        y_inner = y[outer_tr]
        g_inner = groups[outer_tr]
        X_test = X[outer_te]
        y_test = y[outer_te]

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
            study_name=f"{model_name}_fold{fold_i}",
        )
        study.optimize(
            lambda t: _optuna_objective(t, model_name, X_inner, y_inner, g_inner, n_inner_folds),
            n_trials=n_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        best_inner_f1 = float(study.best_value)
        print(f"  best inner F1: {best_inner_f1:.4f}  params: {best_params}")

        # Refit on full outer-train with best params
        scaler = StandardScaler()
        X_inner_s = scaler.fit_transform(X_inner)
        X_test_s = scaler.transform(X_test)
        clf = _build_classifier(model_name, best_params, y_inner)
        clf.fit(X_inner_s, y_inner)

        probs = clf.predict_proba(X_test_s)[:, 1]
        preds = (probs >= 0.5).astype(int)
        oof_probs[outer_te] = probs
        oof_preds[outer_te] = preds
        oof_assigned[outer_te] = True

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fold_metric = {
            "fold": fold_i,
            "best_params": best_params,
            "best_inner_f1": best_inner_f1,
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else None,
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "confusion_matrix": cm.tolist(),
        }
        print(f"  fold {fold_i}  F1={fold_metric['f1']:.4f}  "
              f"P={fold_metric['precision']:.3f}  R={fold_metric['recall']:.3f}  "
              f"AUC={fold_metric['roc_auc']:.3f}")

        print(f"  computing SHAP ({model_name})...")
        sv, _ = _compute_shap(model_name, clf, X_inner_s, X_test_s)
        shap_means_abs.append(np.abs(sv).mean(axis=0))

        fold_results.append(fold_metric)

    # ── Aggregate across folds
    def _agg(key):
        vals = [r[key] for r in fold_results if r[key] is not None]
        return {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "per_fold": [r[key] for r in fold_results],
        }

    summary = {k: _agg(k) for k in
               ("f1", "precision", "recall", "accuracy", "roc_auc", "fpr", "best_inner_f1")}

    # OOF aggregate view (one row per sample, predicted while held out)
    cm_oof = confusion_matrix(y[oof_assigned], oof_preds[oof_assigned], labels=[0, 1])
    fpr_oof, tpr_oof, _ = roc_curve(y[oof_assigned], oof_probs[oof_assigned])
    oof_auc = float(roc_auc_score(y[oof_assigned], oof_probs[oof_assigned])) \
        if len(np.unique(y[oof_assigned])) > 1 else None

    # SHAP aggregation: mean of mean-|shap| per feature across folds
    shap_per_fold = np.stack(shap_means_abs)  # (n_folds, n_features)
    shap_mean = shap_per_fold.mean(axis=0)
    shap_std = shap_per_fold.std(axis=0, ddof=1) if len(shap_per_fold) > 1 else np.zeros_like(shap_mean)
    shap_records = [
        {
            "feature": fn,
            "mean_abs_shap": float(m),
            "std_abs_shap": float(s),
            "per_fold_mean_abs_shap": shap_per_fold[:, i].tolist(),
        }
        for i, (fn, m, s) in enumerate(zip(feature_names, shap_mean, shap_std))
    ]
    shap_records.sort(key=lambda r: r["mean_abs_shap"], reverse=True)

    return {
        "fold_results": fold_results,
        "summary": summary,
        "oof_predictions": oof_preds.tolist(),
        "oof_probabilities": oof_probs.tolist(),
        "oof_labels": y.tolist(),
        "oof_confusion_matrix": cm_oof.tolist(),
        "oof_roc_curve": {"fpr": fpr_oof.tolist(), "tpr": tpr_oof.tolist()},
        "oof_roc_auc": oof_auc,
        "shap_records": shap_records,
    }


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
    parser.add_argument("--n-outer-folds", type=int, default=5)
    parser.add_argument("--n-inner-folds", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna trials per outer fold")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"\n{'='*50}")
    print(f"Model: {args.model}")
    print(f"{'='*50}")
    print("Device: cpu (small models — no GPU needed)")

    # ── Step 1: Load CSV, drop AMBIGUOUS/empty rows ──
    df = pd.read_csv(args.input)
    if "rebuttal" in df.columns and "rebuttal_text" not in df.columns:
        df = df.rename(columns={"rebuttal": "rebuttal_text"})
    df = df.dropna(subset=["rebuttal_text"]).copy()
    df = df[df["rebuttal_text"].astype(str).str.strip() != ""].copy()
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["label"].isin({"FLIP", "HOLD"})].reset_index(drop=True)
    df["label_encoded"] = (df["label"] == "FLIP").astype(int)
    print(f"Loaded {len(df)} samples, {df['question_id'].nunique()} questions, "
          f"FLIP={df['label_encoded'].mean()*100:.1f}%")

    # ── Step 2: Pre-compute spaCy + sbert ONCE on full df ──
    print(f"\nPre-computing spaCy + sentence-transformer embeddings (once)...")
    docs, sim_c, sim_w = precompute_nlp_and_embeddings(df)

    # ── Step 3: Extract features ──
    print(f"Extracting features...")
    feats, feature_names = extract_features(df, docs=docs, sim_correct=sim_c, sim_wrong=sim_w)
    print(f"  Feature count: {len(feature_names)}")

    X = feats.values.astype(np.float64)
    y = df["label_encoded"].values
    groups = df["question_id"].values

    # ── Step 4: Nested CV with Optuna + per-fold SHAP ──
    print(f"\nNested {args.n_outer_folds}-fold outer × {args.n_inner_folds}-fold inner CV, "
          f"{args.n_trials} Optuna trials per fold")
    cv = run_nested_cv(
        args.model, X, y, groups, feature_names,
        n_outer_folds=args.n_outer_folds,
        n_inner_folds=args.n_inner_folds,
        n_trials=args.n_trials,
        seed=args.seed,
    )

    # ── Step 5: Print summary ──
    print(f"\n{'='*60}")
    print(f"Nested-CV summary — {args.model}")
    print(f"{'='*60}")
    s = cv["summary"]
    for k in ["f1", "precision", "recall", "accuracy", "roc_auc", "fpr", "best_inner_f1"]:
        m, sd = s[k]["mean"], s[k]["std"]
        if m is None:
            continue
        print(f"  {k:>14}: {m:.4f} ± {sd:.4f}")
    print(f"  {'oof_roc_auc':>14}: {cv['oof_roc_auc']:.4f}" if cv["oof_roc_auc"] else "")

    print(f"\nTop 20 features by mean |SHAP|:")
    for i, rec in enumerate(cv["shap_records"][:20], 1):
        print(f"  {i:>2}. {rec['feature']:<35} "
              f"mean|SHAP|={rec['mean_abs_shap']:.4f}  "
              f"(±{rec['std_abs_shap']:.4f})")

    # ── Step 6: Save OOF ROC plot ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.plot(cv["oof_roc_curve"]["fpr"], cv["oof_roc_curve"]["tpr"], lw=2,
             label=f"{args.model} (OOF AUC = {cv['oof_roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Out-of-fold ROC — {args.model}")
    plt.legend(loc="lower right"); plt.tight_layout()
    roc_path = os.path.join(args.output, "roc_curve.png")
    plt.savefig(roc_path, dpi=150); plt.close()
    print(f"\n✓ ROC curve saved to {roc_path}")

    # ── Step 7: Save everything ──
    results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "device": "cpu",
        "config": {
            "n_outer_folds": args.n_outer_folds,
            "n_inner_folds": args.n_inner_folds,
            "n_trials": args.n_trials,
            "seed": args.seed,
        },
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_samples": len(df),
        "n_questions": int(df["question_id"].nunique()),
        "class_balance_flip": float(df["label_encoded"].mean()),
        "cv_summary": cv["summary"],
        "fold_results": cv["fold_results"],
        "oof_confusion_matrix": cv["oof_confusion_matrix"],
        "oof_roc_auc": cv["oof_roc_auc"],
    }
    results_path = os.path.join(args.output, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ results.json saved to {results_path}")

    shap_path = os.path.join(args.output, "shap_values.json")
    with open(shap_path, "w") as f:
        json.dump(cv["shap_records"], f, indent=2)
    print(f"✓ shap_values.json saved to {shap_path}")

    pred_df = df.copy()
    pred_df["predicted"] = cv["oof_predictions"]
    pred_df["probability"] = cv["oof_probabilities"]
    pred_path = os.path.join(args.output, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"✓ predictions.csv (out-of-fold) saved to {pred_path}")

    print(f"\n{'='*50}")
    print(f"DONE — {args.model}")
    print(f"  CV F1:  {s['f1']['mean']:.4f} ± {s['f1']['std']:.4f}")
    print(f"  CV AUC: {s['roc_auc']['mean']:.4f} ± {s['roc_auc']['std']:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
