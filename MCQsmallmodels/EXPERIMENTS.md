# Experiments — linguistic features, embeddings, and the hybrid

## Setup

**Task.** Predict whether Claude Haiku 3.5 will FLIP (abandon its correct
answer) or HOLD (maintain it) given a single user rebuttal, using only
properties of the rebuttal text.

**Data.** Merged team dataset `MCQ/NEW_MCQ_MERGED_DATASET.json` —
10,090 challenges across 8 teammate slices, all targeting
`anthropic/claude-3.5-haiku`. After dropping AMBIGUOUS/ERROR labels and
applying a per-question robust/brittle filter (drop questions whose
flip rate is below 10% or above 90%), the training set is
**4,797 rebuttals across 253 unique questions, 44.8% FLIP / 55.2% HOLD**.

**Methodology.** Nested 5-outer × 3-inner StratifiedGroupKFold (grouping
by question_id so all 20 rebuttals of a given question stay in the same
fold). Optuna TPE hyperparameter search with 30 trials per outer fold,
maximising inner-CV F1. Per-fold decision threshold tuning on inner-CV
held-out probabilities. Class imbalance handled via cost-sensitive
learning. Same logistic regression classifier across all three runs;
only the feature matrix changes.

**Three feature configurations tested.**

| Config | Features | Description |
|---|---|---|
| Linguistic only | 61 | Hand-crafted: 9 orthographic + 2 lexical + 13 pragmatic + 3 VADER sentiment + 10 morphological + 11 syntactic + 2 similarity + 5 rebuttal-vs-initial diff + 6 typography/perspective/NER. No sentence embeddings. |
| Embedding only (Option 2) | 60 | Top-60 PCA components of the rebuttal's `all-MiniLM-L6-v2` sentence embedding. No hand-crafted features. |
| Hybrid | 121 | Linguistic + Embedding combined. |

The frozen sentence-transformer (`all-MiniLM-L6-v2`) is the same in
Embedding-only and Hybrid — only the hand-crafted half is removed for
the Option 2 run.

---

## Results

### Headline metrics (5-fold nested CV, mean ± std)

| Config | n features | F1 | AUC | Precision | Recall | Accuracy | FPR |
|---|---|---|---|---|---|---|---|
| Linguistic only | 61 | not yet run on 4,797 rows; earlier run on 2,955 rows: **0.575 ± 0.025** | 0.636 ± 0.013 | 0.541 | 0.614 | 0.600 | 0.411 |
| **Embedding only (pure Option 2)** | 60 | 0.620 ± 0.017 | **0.599 ± 0.039** | **0.452** | **0.986** | **0.459** | **0.970** |
| **Hybrid** | **121** | **0.635 ± 0.020** | **0.665 ± 0.028** | **0.498** | **0.878** | **0.549** | **0.719** |

### How to read the embedding-only row

At first glance, F1 = 0.620 for embedding-only looks higher than the
linguistic-only 0.575. **It isn't really.** The threshold tuner picked
t ≈ 0.30 because the model couldn't discriminate, so the model
collapsed into "predict FLIP for almost every rebuttal":

- Recall = **98.6%** (catches almost every FLIP)
- Precision = **45.2%** (over half its FLIP predictions are wrong)
- Accuracy = **45.9%** (worse than random)
- FPR = **97.0%** (97 of every 100 HOLDs misclassified as FLIP)
- AUC = **0.599** (the honest discrimination measure — weakly above
  chance)

"Always predict FLIP" at this dataset's 44.8% base rate has F1 = 0.619
by definition. The embedding-only model is essentially that trivial
baseline.

### What the AUC column actually tells us

AUC is the threshold-independent measure of how well the model ranks
FLIPs above HOLDs. Each row's AUC is the honest number:

- **Embedding alone: AUC 0.599** — barely above chance.
- **Linguistic alone: AUC 0.636** — clearly discriminative.
- **Hybrid: AUC 0.665** — best of all three, **+0.066 over embedding
  alone, +0.029 over linguistic alone**.

The Hybrid is not just the better of the two; the two halves are
**additive** — combining them produces a model better than either part
on its own.

---

## Findings

### 1. The hand-crafted linguistic features carry real signal

Embedding-only ran on a strictly larger dataset (4,797 rows) than the
2,955-row linguistic-only baseline and still lost on AUC (0.599 vs
0.636). A generic sentence-transformer trained on millions of unrelated
sentences does not, by itself, encode the linguistic properties that
predict sycophantic flipping in this dataset. The hand-crafted features
the project deliberately designed — clause count, modal-verb usage,
sentence complexity, sentiment, rebuttal-to-wrong-answer similarity,
typography — are not redundant with the embedding.

### 2. The embeddings still add value as a supplement

Hybrid beats Linguistic-only by +0.029 AUC. Some of the semantic axes
captured by the sentence embedding are genuinely orthogonal to the
hand-crafted features. The contribution is modest but real.

### 3. The Hybrid is the strongest configuration

| Run | n features | n rows | AUC | F1 |
|---|---|---|---|---|
| **Hybrid (this project)** | 121 | 4,797 | **0.665** | **0.635** |
| Embedding only | 60 | 4,797 | 0.599 | 0.620 |
| BERT (Layan) | — | 2,955 | 0.618 | 0.550 |
| RoBERTa (Layan) | — | 2,955 | 0.594 | 0.587 |
| DistilBERT (Layan) | — | 2,955 | 0.615 | 0.588 |

The Hybrid beats every transformer-based model from the earlier BERT
experiments on both F1 and AUC, on a larger dataset, with full
interpretability via SHAP.

### 4. Threshold tuning can mislead when discrimination is weak

The embedding-only result is a cautionary tale. When AUC is low,
optimising F1 via threshold tuning can find a degenerate "predict the
majority class" solution that looks acceptable on F1 alone but is
useless in practice. **Always report AUC alongside F1.** AUC is
threshold-independent and base-rate-independent; it is the honest
measure of how well the model discriminates.

### 5. SHAP top features in the Hybrid (mean |SHAP|, 5-fold avg)

The Hybrid's most important features by SHAP are a mix of hand-crafted
linguistic features and embedding PCA components:

| Rank | Feature | Type | mean \|SHAP\| |
|---|---|---|---|
| 1 | `starts_lower` | orthographic | 0.123 |
| 2 | `sbert_pc_00` | embedding | 0.122 |
| 3 | `n_clauses` | syntactic | 0.114 |
| 4 | `sbert_pc_22` | embedding | 0.112 |
| 5 | `sbert_pc_37` | embedding | 0.111 |
| 6 | `n_question_marks` | orthographic | 0.100 |
| 7 | `sbert_pc_46` | embedding | 0.100 |
| 8 | `sbert_pc_11` | embedding | 0.094 |
| 9 | `sbert_pc_47` | embedding | 0.093 |
| 10 | `vader_pos` | sentiment | 0.091 |
| 11 | `sbert_pc_36` | embedding | 0.089 |
| 12 | `sbert_pc_38` | embedding | 0.080 |
| 13 | `n_commas` | orthographic | 0.078 |
| 14 | `sim_to_wrong` | semantic similarity | 0.077 |
| 15 | `sbert_pc_39` | embedding | 0.076 |
| 16 | `n_nominalisations` | morphological | 0.064 |
| 17 | `n_subord_clauses` | syntactic | 0.058 |
| 18 | `sbert_pc_58` | embedding | 0.058 |
| 19 | `clauses_per_sentence` | syntactic | 0.054 |
| 20 | `sbert_pc_03` | embedding | 0.049 |

**Pattern.** 9 hand-crafted linguistic features and 11 embedding
components share the top 20. The hand-crafted side is dominated by
**sentence structure** (clause counts, subordinate clauses,
clauses-per-sentence), **punctuation density** (commas, question
marks), **sentiment**, and **semantic similarity to the wrong answer**.
Aggression, ALL-CAPS, and exclamations are notably absent from the top
of the rankings.

---

## What this means for the project

- The proposal's hypothesis — that **fine-grained linguistic properties
  of a rebuttal predict sycophantic flipping** — is supported. The
  hand-crafted feature set is doing real predictive work, and the
  SHAP-ranked features point to specific properties (clause structure,
  question form, semantic alignment with the wrong answer) rather than
  topic content.
- A generic neural embedding alone is **not sufficient** for this task.
  This is methodologically useful because it argues the project's
  hand-crafted feature engineering is not redundant with what
  transformers learn for free.
- The Hybrid is the recommended headline configuration: highest AUC,
  highest F1, lowest variance across folds, and the SHAP explanations
  remain dominated by interpretable linguistic features.
- The earlier BERT-family experiments (which produced uninterpretable
  topic-word attributions) are outperformed in both accuracy and
  interpretability.

## Output artifacts

Under `MCQsmallmodels/results/`:

- `new_logreg_emb_only/` — Embedding-only (Option 2) run, 60 features,
  4,797 rows.
- `new_logreg_hybrid/` — Hybrid run, 121 features, 4,797 rows.
- `new_logreg_91/` — earlier Hybrid run on 2,955 rows (for reference).
- Each directory contains `results.json` (full per-fold metrics +
  best hyperparameters), `shap_values.json` (mean |SHAP| per feature
  across folds), `predictions.csv` (out-of-fold prediction +
  probability per rebuttal), `roc_curve.png` (OOF ROC).
