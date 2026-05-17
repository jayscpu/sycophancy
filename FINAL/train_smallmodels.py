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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    "not sure", "i'm not sure", "appears", "kind of", "sort of",
    "possibly", "probably", "i suppose",
    "i'm not certain", "if i'm not mistaken", "i could be wrong", "correct me if",
}
CERTAINTY = {
    "definitely", "absolutely", "100%", "certainly", "clearly", "obviously",
    "without doubt", "no doubt", "for sure", "undoubtedly", "indeed", "of course",
    "i'm sure", "i know", "trust me", "guaranteed", "there's no way",
    "that's a fact", "pretty sure",
}
POLITENESS = {
    "please", "could you", "would you mind", "thank you", "thanks", "sorry",
    "appreciate", "kindly", "if you don't mind",
}
AGGRESSION = {
    "incorrect", "you're wrong", "that's wrong", "it's wrong", "thats wrong",
    "ur wrong", "stop", "duh", "seriously", "come on", "ridiculous", "nonsense",
    "lol", "lmao", "bruh", "are you kidding", "you can't be serious", "smh",
}
SOURCES = {
    "study", "studies", "research", "scientist", "scientists", "expert", "experts",
    "research paper", "a paper", "journal", "article", "source", "i read", "i heard",
    "wikipedia", "google", "internet", "online", "documentary", "read a book about",
    "youtube", "video", "podcast", "news", "report", "according to",
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
    "update your", "update it", "fix it", "correct yourself", "re-check", "try again", "do your research",
}

# Modal verbs partitioned by epistemic certainty (Palmer 2001, Biber et al. 1999)
MODALS_CERTAIN = {"must", "will", "shall", "would"}
MODALS_UNCERTAIN = {"may", "might", "could", "can", "should"}
MODALS_ALL = MODALS_CERTAIN | MODALS_UNCERTAIN

NOMINALISATION_SUFFIXES = ("tion", "ment", "ness", "ity", "ance", "ence", "sion", "ship")

CONTRAST_MARKERS = {
    "actually", "but", "however", "in fact", "on the contrary",
    "rather", "instead", "yet", "nevertheless", "nonetheless",
}
# "absolutely" also appears in CERTAINTY — intentional: it is both an intensifier and a
# certainty marker. SHAP will split its contribution between n_intensifiers and n_certainty.
INTENSIFIERS = {
    "very", "really", "extremely", "super", "totally", "absolutely",
}

# Concede-then-redirect rhetorical move: "yes but...", "you're right but..."
# Captures rebuttals that acknowledge the AI's position before redirecting.
CONCEDE_MARKERS = {
    "yes but", "true but", "you're right but", "youre right but",
    "i agree but", "i agree, but", "agreed but", "agree but",
    "fair point but", "fair enough but", "fair point",
    "i see your point but", "i see your point",
    "you have a point but", "you have a point",
    "that's true but", "thats true but",
    "valid but", "valid point but",
}

# Rhetorical question markers: phrases that frame the rebuttal as a question
# even when no answer is expected ("have you considered...", "isn't it...").
RHETORICAL_QUESTION_MARKERS = {
    "have you considered", "have you thought",
    "don't you think", "dont you think",
    "isn't it", "isnt it",
    "wouldn't it", "wouldnt it",
    "shouldn't it", "shouldnt it",
    "aren't you", "arent you",
    "wasn't it", "wasnt it",
    "what if", "what about",
}

_FIRST_PERSON_RE = re.compile(r"\b(i|my|me|i'm|i've|i'll|i'd|myself)\b")
_SECOND_PERSON_RE = re.compile(r"\b(you|your|you're|you've|you'll|yourself)\b")
_THIRD_PERSON_RE = re.compile(
    r"\b(he|she|it|they|him|her|them|his|hers|its|their|theirs|"
    r"himself|herself|itself|themself|themselves|"
    r"he's|she's|it's|they're|they've|they'd|they'll)\b"
)

# Typography / emphasis patterns
_REPEATED_PUNCT_RE = re.compile(r"[!?.]{2,}")
_ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF"     # most emoji blocks
    "\U00002600-\U000027BF"      # misc symbols + dingbats
    "\U0001F000-\U0001F02F"      # mahjong / cards
    "]"
)
_QUOTED_RE = re.compile(r"[\"“”][^\"“”]+[\"“”]")

# Initial-reframe pattern: rebuttal starts with a contradiction/correction word
# (leading whitespace and an optional opening quote are skipped).
_INITIAL_REFRAME_RE = re.compile(
    r"^[\s\"'“‘]*\s*(actually|but|however|wait|no|nope|nah|um|umm)\b",
    re.IGNORECASE,
)

# Numerical expressions: digits possibly with decimals/separators and an
# optional unit/ordinal/currency suffix. Captures specificity / evidence density.
_NUMERICAL_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?(?:%|st|nd|rd|th|°[cf]?|s)?\b",
    re.IGNORECASE,
)

# Hesitation/musing punctuation — distinct from emphatic !!/??/... lumped under
# n_repeated_punctuation. Captures unicode horizontal ellipsis (…) and 3+ dots.
_ELLIPSIS_RE = re.compile(r"…|\.{3,}")

# Markdown asterisk emphasis: *word* (single token) or *phrase here* (multi).
_ASTERISK_EMPH_RE = re.compile(r"\*[^\s*][^*]*\*")


def _count_phrases(text_lower: str, lexicon: set[str]) -> int:
    """Count how many lexicon phrases appear in the text (substring match)."""
    return sum(1 for phrase in lexicon if phrase in text_lower)


def _count_hedging(text_lower: str) -> int:
    """Count hedging markers; corrects for 'i don't think' which is assertive
    but contains the substring 'i think'."""
    count = _count_phrases(text_lower, HEDGING)
    count -= text_lower.count("i don't think") + text_lower.count("i dont think")
    return max(0, count)




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
        # NER enabled so we can count doc.ents for n_named_entities
        _NLP = spacy.load("en_core_web_sm")
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
            if t.dep_ in {"auxpass", "aux:pass"}:
                passive_aux += 1
            if t.dep_ in {"nsubjpass", "csubjpass", "nsubj:pass", "csubj:pass"}:
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
            "n_modals_certain": modals_certain,
            "n_modals_uncertain": modals_uncertain,
            "modal_uncertainty_ratio": modals_uncertain / max(1, modals_all),
            "modal_density": modals_all / n_words,
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
            "n_named_entities": len(doc.ents),
        })
    return pd.DataFrame(rows)


N_SBERT_PCA = 60  # how many PCA components of the rebuttal embedding to use


def precompute_nlp_and_embeddings(df: pd.DataFrame, n_pca: int = N_SBERT_PCA,
                                   skip_sbert: bool = False):
    """Run spaCy + sbert once over the full df.

    Returns (docs, init_docs, sim_correct, sim_wrong, sbert_pcs) aligned to df.index.
      docs       — spaCy parse of rebuttal_text
      init_docs  — spaCy parse of initial_answer (or None if column absent);
                   used internally by extract_features to compute the
                   rebuttal-vs-initial difference features
      sbert_pcs  — top-`n_pca` PCA components of rebuttal sentence embeddings

    Note: PCA is fit on the full dataset (unsupervised, label-free), which is
    a mild methodological compromise vs per-fold refitting. Standard practice
    for embedding dimensionality reduction.
    """
    text_list = df["rebuttal_text"].astype(str).tolist()
    nlp = _get_nlp()
    docs = list(nlp.pipe(text_list, batch_size=64))

    # Parse Haiku's initial answer too — needed for diff features (skipped in rebuttal_only mode)
    init_docs = None
    if not skip_sbert and "initial_answer" in df.columns:
        init_text_list = df["initial_answer"].fillna("").astype(str).tolist()
        init_docs = list(nlp.pipe(init_text_list, batch_size=64))

    sim_correct = sim_wrong = None
    sbert_pcs = None
    if not skip_sbert and "correct_answer" in df.columns and "wrong_answer" in df.columns:
        sbert = _get_sbert()
        reb_emb = sbert.encode(text_list, batch_size=64, show_progress_bar=False,
                               convert_to_numpy=True)
        cor_emb = sbert.encode(df["correct_answer"].fillna("").astype(str).tolist(),
                               batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        wro_emb = sbert.encode(df["wrong_answer"].fillna("").astype(str).tolist(),
                               batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        sim_correct = _cos_sim(reb_emb, cor_emb)
        sim_wrong = _cos_sim(reb_emb, wro_emb)

        # PCA on rebuttal embeddings — gives classifier semantic content
        if n_pca > 0:
            pca = PCA(n_components=min(n_pca, reb_emb.shape[1], len(reb_emb)),
                      random_state=42)
            sbert_pcs = pca.fit_transform(reb_emb)
            explained = pca.explained_variance_ratio_.sum()
            print(f"  sbert PCA: {sbert_pcs.shape[1]} components capture "
                  f"{explained * 100:.1f}% of embedding variance")

    return docs, init_docs, sim_correct, sim_wrong, sbert_pcs


def _init_response_quick(init_text_series: pd.Series, init_docs: list) -> dict:
    """Compute the minimal init-side quantities used by the diff features.
    Returned as a dict of arrays — values are not exposed as features themselves,
    only consumed by the diff computation below."""
    hedging, certainty, modal_unc_ratio, vader_comp, n_words = [], [], [], [], []
    for s, doc in zip(init_text_series, init_docs):
        s = str(s); s_lower = s.lower()
        nw = max(1, len(s.split()))
        n_words.append(nw)
        hedging.append(_count_hedging(s_lower))
        certainty.append(_count_phrases(s_lower, CERTAINTY))
        modals_all = modals_uncertain = 0
        for t in doc:
            lemma = t.lemma_.lower()
            if (t.pos_ == "AUX" or t.tag_ == "MD") and lemma in MODALS_ALL:
                modals_all += 1
                if lemma in MODALS_UNCERTAIN:
                    modals_uncertain += 1
        modal_unc_ratio.append(modals_uncertain / max(1, modals_all))
        vader_comp.append(_VADER.polarity_scores(s)["compound"])
    return {
        "init_n_hedging": np.array(hedging, dtype=float),
        "init_n_certainty": np.array(certainty, dtype=float),
        "init_modal_uncertainty_ratio": np.array(modal_unc_ratio, dtype=float),
        "init_vader_compound": np.array(vader_comp, dtype=float),
        "init_len_words": np.array(n_words, dtype=float),
    }


def extract_features(
    df: pd.DataFrame,
    docs: list | None = None,
    init_docs: list | None = None,
    sim_correct: np.ndarray | None = None,
    sim_wrong: np.ndarray | None = None,
    sbert_pcs: np.ndarray | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a numeric feature matrix from a DataFrame of rebuttals.
    Returns (features_df, feature_names).

    Six linguistic levels per the proposal:
      orthographic  — length, punctuation, casing
      lexical       — readability (Flesch), lexical diversity (type-token ratio)
      pragmatic     — epistemic stance (hedging/certainty), politeness, aggression,
                      evidential strategies (sources/anecdote/authority), directive
                      speech acts (command), contrast markers, intensifiers,
                      person ratios, question-only detection
      morphological — modal verbs (epistemic certainty split), tense, nominalisation,
                      negation                             (spaCy)
      syntactic     — clause count, parse depth, passive voice, subordination,
                      sentence-length variance              (spaCy)
      semantic      — VADER sentiment + cosine similarity of rebuttal to the
                      wrong_answer                          (sentence-transformers)

    Metadata columns (batch, teammate) are intentionally excluded: they encode
    collection provenance, not properties of the rebuttal.

    Pre-compute `docs` / `sim_*` once on the full df with precompute_nlp_and_embeddings()
    and pass them in to avoid running spaCy + sbert separately for each CV split.
    """
    if docs is None:
        docs, init_docs, sim_correct, sim_wrong, sbert_pcs = precompute_nlp_and_embeddings(df)

    feats = pd.DataFrame(index=df.index)
    text = df["rebuttal_text"].astype(str)
    text_lower = text.str.lower()

    # — Orthographic / style / length  (len_chars dropped: r~0.98 with len_words)
    feats["len_words"] = text.str.split().str.len()
    feats["n_sentences"] = [max(1, len(list(d.sents))) for d in docs]
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
    # ends_with_q dropped: near-identical signal to n_question_marks > 0

    # — Typography / emphasis
    feats["n_all_caps_words"] = text.apply(lambda s: len(_ALL_CAPS_WORD_RE.findall(s)))
    feats["n_repeated_punctuation"] = text.apply(lambda s: len(_REPEATED_PUNCT_RE.findall(s)))
    feats["has_emoji"] = text.apply(lambda s: int(bool(_EMOJI_RE.search(s))))
    feats["n_quoted_strings"] = text.apply(lambda s: len(_QUOTED_RE.findall(s)))

    # — Lexical: readability + diversity  (flesch_grade dropped: r~0.95 with flesch_reading_ease)
    feats["flesch_reading_ease"] = text.apply(lambda s: textstat.flesch_reading_ease(s) if s.strip() else 0.0)
    feats["type_token_ratio"] = text_lower.apply(
        lambda s: len(set(s.split())) / max(1, len(s.split()))
    )

    # — Pragmatic: epistemic stance, speech acts, interpersonal stance
    feats["n_hedging"] = text_lower.apply(_count_hedging)
    feats["n_certainty"] = text_lower.apply(lambda s: _count_phrases(s, CERTAINTY))
    feats["n_politeness"] = text_lower.apply(lambda s: _count_phrases(s, POLITENESS))
    feats["n_aggression"] = text_lower.apply(lambda s: _count_phrases(s, AGGRESSION))
    feats["n_sources"] = text_lower.apply(lambda s: _count_phrases(s, SOURCES))
    feats["n_anecdote"] = text_lower.apply(lambda s: _count_phrases(s, ANECDOTE))
    feats["n_authority"] = text_lower.apply(lambda s: _count_phrases(s, AUTHORITY))
    feats["n_command"] = text_lower.apply(lambda s: _count_phrases(s, COMMAND))
    feats["discourse_contrast_markers"] = text_lower.apply(lambda s: _count_phrases(s, CONTRAST_MARKERS))
    feats["n_intensifiers"] = text_lower.apply(lambda s: _count_phrases(s, INTENSIFIERS))
    feats["first_person_ratio"] = text_lower.apply(
        lambda s: len(_FIRST_PERSON_RE.findall(s)) / max(1, len(s.split()))
    )
    feats["second_person_ratio"] = text_lower.apply(
        lambda s: len(_SECOND_PERSON_RE.findall(s)) / max(1, len(s.split()))
    )
    feats["third_person_ratio"] = text_lower.apply(
        lambda s: len(_THIRD_PERSON_RE.findall(s)) / max(1, len(s.split()))
    )
    feats["is_question_only"] = text.apply(
        lambda s: int(bool(s.strip()) and s.strip().endswith("?") and "." not in s and "!" not in s)
    )

    # — Discourse / rhetorical move structure
    feats["n_concede_markers"] = text_lower.apply(
        lambda s: _count_phrases(s, CONCEDE_MARKERS)
    )
    feats["has_initial_reframe"] = text.apply(
        lambda s: int(bool(_INITIAL_REFRAME_RE.match(s)))
    )
    feats["n_rhetorical_question_markers"] = text_lower.apply(
        lambda s: _count_phrases(s, RHETORICAL_QUESTION_MARKERS)
    )

    # — Specificity / evidence density
    feats["n_numerical_expressions"] = text.apply(
        lambda s: len(_NUMERICAL_RE.findall(s))
    )

    # — Data-grounded patterns kept after empirical screening (3 others
    # tested and dropped after near-zero SHAP attribution):
    # Hesitant/musing punctuation (distinct from emphatic !! ?? lumped in
    # n_repeated_punctuation): "wait… really?" patterns.
    feats["n_ellipsis"] = text.apply(lambda s: len(_ELLIPSIS_RE.findall(s)))
    # Markdown-style emphasis: *word* — common in LLM-generated rhetorical text
    feats["n_asterisk_emphasis"] = text.apply(lambda s: len(_ASTERISK_EMPH_RE.findall(s)))

    # — Semantic: VADER  (vader_neu dropped: exact linear dependence neu = 1 - pos - neg)
    vader_scores = text.apply(lambda s: _VADER.polarity_scores(s))
    feats["vader_pos"] = vader_scores.apply(lambda d: d["pos"])
    feats["vader_neg"] = vader_scores.apply(lambda d: d["neg"])
    feats["vader_compound"] = vader_scores.apply(lambda d: d["compound"])

    # — Morphological + syntactic via spaCy (uses pre-computed docs)
    morph_syn = _morph_syntax_features(docs)
    morph_syn.index = df.index
    feats = pd.concat([feats, morph_syn], axis=1)

    # — Semantic: cosine similarity  (sim_to_correct dropped: derivable as
    #   sim_to_wrong - sim_margin, keeping it splits SHAP across correlated features)
    if sim_correct is not None and sim_wrong is not None:
        feats["sim_to_wrong"] = sim_wrong
        feats["sim_margin_wrong_minus_correct"] = sim_wrong - sim_correct

    # — Semantic: sentence-transformer PCA components of the rebuttal embedding.
    # Gives the classifier direct access to semantic content of the rebuttal
    # (beyond similarity-to-answer). PCA components are not individually
    # interpretable; SHAP on them tells us *that* meaning matters, not *what* meaning.
    if sbert_pcs is not None:
        pca_cols = pd.DataFrame(
            sbert_pcs,
            index=df.index,
            columns=[f"sbert_pc_{i:02d}" for i in range(sbert_pcs.shape[1])],
        )
        feats = pd.concat([feats, pca_cols], axis=1)

    # — Difference features: how the rebuttal contrasts with Haiku's initial
    # answer. The init-side values themselves are not exposed; only the
    # rebuttal-minus-initial deltas are surfaced as features.
    if init_docs is not None and "initial_answer" in df.columns:
        init = _init_response_quick(df["initial_answer"], init_docs)
        feats["diff_n_hedging"] = feats["n_hedging"].values - init["init_n_hedging"]
        feats["diff_n_certainty"] = feats["n_certainty"].values - init["init_n_certainty"]
        feats["diff_modal_uncertainty"] = (
            feats["modal_uncertainty_ratio"].values - init["init_modal_uncertainty_ratio"]
        )
        feats["diff_vader_compound"] = feats["vader_compound"].values - init["init_vader_compound"]
        feats["diff_len_words"] = feats["len_words"].values - init["init_len_words"]

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
    if model_name == "mlp":
        # MLPClassifier doesn't accept class_weight/sample_weight; per-fold
        # threshold tuning compensates for the mild 45/55 imbalance.
        # hidden_layer_sizes comes in as a string from Optuna; parse to tuple.
        mlp_params = dict(params)
        if isinstance(mlp_params.get("hidden_layer_sizes"), str):
            import ast
            mlp_params["hidden_layer_sizes"] = ast.literal_eval(mlp_params["hidden_layer_sizes"])
        return MLPClassifier(
            **mlp_params, max_iter=500, random_state=42,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
        )
    if model_name == "elasticnet":
        # Logistic regression with combined L1+L2 penalty. saga solver is
        # required for elasticnet. l1_ratio=0 -> pure L2, =1 -> pure L1.
        return LogisticRegression(
            **params, penalty="elasticnet", solver="saga",
            max_iter=5000, class_weight="balanced", random_state=42, n_jobs=-1,
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
    if model_name == "mlp":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes", ["(32,)", "(64,)", "(128,)", "(64, 32)", "(128, 64)"]
            ),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    if model_name == "elasticnet":
        return {
            "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
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
    if model_name in ("logreg", "elasticnet"):
        bg = shap.sample(X_train, min(bg_size, len(X_train)), random_state=seed)
        explainer = shap.LinearExplainer(clf, bg)
        sv = explainer.shap_values(X_test)
        X_explained = X_test
    elif model_name == "xgboost":
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_test)
        X_explained = X_test
    elif model_name in ("svm-rbf", "mlp"):
        # Model-agnostic KernelExplainer — sklearn MLPClassifier doesn't expose
        # gradients via shap.DeepExplainer (that needs keras/torch), so kernel
        # SHAP is the cleanest option.
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

        # Threshold tuning: get OOF probs on outer-train via inner CV with the
        # tuned hyperparameters, then sweep thresholds for max F1. Tuned
        # threshold is applied to outer-test predictions. Threshold tuning is
        # leakage-safe because the held-out probs come from inner CV (no
        # outer-train datum is predicted by a model it was trained on).
        tune_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=seed)
        oof_train_probs = np.zeros(len(y_inner))
        for tr, va in tune_cv.split(X_inner, y_inner, g_inner):
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_inner[tr])
            X_va_s = sc.transform(X_inner[va])
            clf_in = _build_classifier(model_name, best_params, y_inner[tr])
            clf_in.fit(X_tr_s, y_inner[tr])
            oof_train_probs[va] = clf_in.predict_proba(X_va_s)[:, 1]

        thresholds = np.arange(0.05, 0.96, 0.01)
        f1_per_t = np.array([
            f1_score(y_inner, (oof_train_probs >= t).astype(int), zero_division=0)
            for t in thresholds
        ])
        best_t = float(thresholds[int(np.argmax(f1_per_t))])
        best_inner_tuned_f1 = float(f1_per_t.max())

        probs = clf.predict_proba(X_test_s)[:, 1]
        preds_default = (probs >= 0.5).astype(int)
        preds = (probs >= best_t).astype(int)  # tuned-threshold preds for reporting
        oof_probs[outer_te] = probs
        oof_preds[outer_te] = preds
        oof_assigned[outer_te] = True

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fold_metric = {
            "fold": fold_i,
            "best_params": best_params,
            "best_inner_f1": best_inner_f1,
            "tuned_threshold": best_t,
            "tuned_inner_f1": best_inner_tuned_f1,
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "f1_at_0.5": float(f1_score(y_test, preds_default, zero_division=0)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else None,
            "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
            "confusion_matrix": cm.tolist(),
        }
        print(f"  fold {fold_i}  F1={fold_metric['f1']:.4f} (vs {fold_metric['f1_at_0.5']:.4f} @0.5, t={best_t:.2f})  "
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
                        choices=["logreg", "xgboost", "svm-rbf", "mlp", "elasticnet"],
                        help="Classifier choice")
    parser.add_argument("--input", required=True, help="Path to merged CSV")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--n-outer-folds", type=int, default=5)
    parser.add_argument("--n-inner-folds", type=int, default=3)
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Optuna trials per outer fold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--features", default="full",
                        choices=["full", "embedding", "linguistic", "rebuttal_only"],
                        help="full = all 121 features; embedding = 60 sbert PCA components only; "
                             "linguistic = hand-crafted features only, no embeddings; "
                             "rebuttal_only = 60 rebuttal-text-only features (no sim/diff/sbert)")
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
    skip_sbert = args.features == "rebuttal_only"
    print(f"\nPre-computing spaCy + sentence-transformer embeddings (once)...")
    docs, init_docs, sim_c, sim_w, sbert_pcs = precompute_nlp_and_embeddings(
        df, skip_sbert=skip_sbert
    )

    # ── Step 3: Extract features ──
    print(f"Extracting features...")
    feats, feature_names = extract_features(df, docs=docs, init_docs=init_docs,
                                             sim_correct=sim_c, sim_wrong=sim_w,
                                             sbert_pcs=sbert_pcs)
    print(f"  Feature count (full): {len(feature_names)}")

    # ── Step 3b: Filter to requested feature subset ──
    _REBUTTAL_ONLY_EXCLUDE = {
        "sim_to_wrong", "sim_margin_wrong_minus_correct",
        "diff_n_hedging", "diff_n_certainty", "diff_modal_uncertainty",
        "diff_vader_compound", "diff_len_words",
    }
    if args.features == "embedding":
        keep = [c for c in feature_names if c.startswith("sbert_pc_")]
        feats = feats[keep]
        feature_names = keep
        print(f"  --features embedding: kept {len(feature_names)} sbert PCA columns only")
    elif args.features == "linguistic":
        keep = [c for c in feature_names if not c.startswith("sbert_pc_")]
        feats = feats[keep]
        feature_names = keep
        print(f"  --features linguistic: kept {len(feature_names)} hand-crafted columns only")
    elif args.features == "rebuttal_only":
        keep = [c for c in feature_names
                if c not in _REBUTTAL_ONLY_EXCLUDE and not c.startswith("sbert_pc_")]
        feats = feats[keep]
        feature_names = keep
        print(f"  --features rebuttal_only: kept {len(feature_names)} rebuttal-text-only features")
    # else "full" — no filter

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
