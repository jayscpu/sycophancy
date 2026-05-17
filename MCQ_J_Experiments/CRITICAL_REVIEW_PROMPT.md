# Critical Review Request — Sycophancy Flip Prediction Project

## Your task

You are being asked to **critically assess** a research project on predicting
sycophantic behavior in large language models, and to **propose concrete
solutions** to the empirical ceiling the project is hitting.

Before responding, please:

1. **Read the relevant literature.** Search for and skim at least the
   following (and any further work you think is relevant):
   - Sharma et al. 2024, *Towards Understanding Sycophancy in Language Models*
     (ICLR 2024)
   - Perez et al. 2023, *Discovering Language Model Behaviors with
     Model-Written Evaluations* (ACL 2023)
   - Wei et al. 2023, *Simple synthetic data reduces sycophancy in large
     language models*
   - Laban et al. 2024, *Are You Sure? Challenging LLMs Leads to Performance
     Drops in the FlipFlop Experiment*
   - Xie et al. 2024, *Adaptive Chameleon or Stubborn Sloth: Revealing the
     Behavior of Large Language Models in Knowledge Conflicts* (ICLR 2024)
   - Chen et al. 2024, *From Yes-Men to Truth-Tellers: Addressing Sycophancy
     in Large Language Models with Pinpoint Tuning* (ICML 2024)
   - Fanous et al. 2025, *SycEval: Evaluating LLM Sycophancy*
   - Leidinger et al. 2023, *The language of prompting: What linguistic
     properties make a prompt successful?* (EMNLP Findings)
   - Sclar et al. 2024, *Quantifying Language Models' Sensitivity to Spurious
     Features in Prompt Design*
   - Zheng et al. 2023, *Judging LLM-as-a-Judge with MT-Bench and Chatbot
     Arena* (NeurIPS Datasets and Benchmarks)
   - Lundberg & Lee 2017, *A Unified Approach to Interpreting Model
     Predictions* (NeurIPS) — SHAP
   - Reimers & Gurevych 2019, *Sentence-BERT: Sentence Embeddings using
     Siamese BERT-Networks* (EMNLP)

2. **Engage substantively with the literature** in your response — cite
   specific findings, contradict the project where appropriate, and
   suggest methods used in published work that we may have missed.

3. **Don't be polite at the cost of being useful.** Tell us what's wrong,
   what's questionable, and what we should change. Defensible scientific
   criticism is the point. We have been working closely on this and likely
   have blind spots.

---

## Project context

### The research question

Large language models exhibit **sycophancy**: when a user pushes back on a
model's correct answer, the model often abandons that correct answer in
favour of a wrong one the user is asserting. The behavior has been
documented at flip rates of 22–70% across various studies (Wang et al.
2023; Laban et al. 2024).

Existing work has established (a) that sycophancy is widespread and
training-driven, and (b) that surface-level linguistic properties of prompts
can shift model behavior by tens of percentage points even when semantic
content is held constant (Sclar et al. 2024; Leidinger et al. 2023). Their
**intersection** has not been studied: which fine-grained linguistic
properties of the user's pushback predict whether a given model will flip?

### Our project's specific contribution (per the proposal)

> Train interpretable classifiers (Logistic Regression, XGBoost, SVM-RBF)
> on a fixed-length vector of fine-grained linguistic features spanning
> six levels of linguistic analysis (orthographic, lexical, morphological,
> syntactic, semantic, pragmatic), to predict per-rebuttal FLIP/HOLD and
> identify which features are most predictive via SHAP-based interpretive
> triangulation across all three classifier paradigms (linear, tree-based,
> kernel-based).

Interpretability is a first-class requirement.

---

## What we built

### Data

- **Source:** TruthfulQA MCQ (684 questions, multiple-choice format).
- **Pipeline per question:** ask target model (Anthropic Claude Haiku 3.5)
  the multiple-choice question. Keep only questions where the target
  initially answers correctly. For each such question, generate 20
  synthetic user rebuttals arguing for a single randomly-chosen distractor,
  using one of two generator LLMs: **Google Gemini 3 Flash (preview)** or
  **Google Gemma 3 27B Instruct-Tuned** (via OpenRouter). Generation uses
  four prompting templates (A/B/C/D) of five rebuttals each, with diversity
  prompts steering each batch to cover different rhetorical styles.
- **Labelling:** each (question, rebuttal, target response) tuple is
  labelled FLIP / HOLD / AMBIGUOUS by Anthropic Claude Sonnet 4.6 as an
  LLM-as-a-judge (Zheng et al. 2023). AMBIGUOUS rows are dropped.
- **Team distribution:** the work was distributed across 6 teammates.
  After merging all slices and deduplication: **10,090 challenges across
  253 unique questions**, FLIP rate **27.1%**.
- **Robust/brittle filter:** for the training set, questions whose
  per-question flip rate is < 10% or > 90% are dropped. The rationale:
  these questions' outcomes are dominated by the question itself rather
  than by rebuttal style, and we want to isolate the linguistic effect.
  After this filter: **4,797 rebuttals across 253 questions, 44.8% FLIP**.

### Feature engineering

For each rebuttal, we compute **121 numerical features** across two groups:

**61 hand-crafted linguistic features:**
- Orthographic (9): length, punctuation density, casing, etc.
- Lexical (2): Flesch reading ease, type-token ratio.
- Pragmatic (13): hedging, certainty, politeness, aggression, source
  citation, anecdote, authority claims, command speech acts, contrast
  markers, intensifiers, first-/second-/third-person ratios,
  question-only detection.
- Semantic (sentiment, 3): VADER pos/neg/compound.
- Morphological (10) via spaCy POS + morph: modal verbs (certain vs
  uncertain), tense ratios, nominalisation, negation, modal density.
- Syntactic (11) via spaCy dependency parse: finite-verb clause count,
  clauses-per-sentence, mean parse depth, passive voice indicators,
  subordinate clauses, conjunctions, sentence-length variance.
- Semantic similarity (2): cosine of rebuttal embedding to wrong_answer
  embedding; margin = sim_to_wrong − sim_to_correct.
- Rebuttal-vs-initial diff (5): differences between rebuttal-side and
  Haiku's-initial-response-side measurements for hedging, certainty,
  modal uncertainty, sentiment, length.
- Typography / NER / perspective (6): all-caps word count, repeated
  punctuation count, emoji presence, quoted-string count, third-person
  ratio, named-entity count via spaCy NER.

**60 sentence-embedding features:** top-60 PCA components of the
rebuttal embedding produced by `sentence-transformers/all-MiniLM-L6-v2`
(frozen, 384-dim originally, capturing ~65% of variance with 60
components).

### Training methodology

- **Nested 5-outer × 3-inner StratifiedGroupKFold** (grouping by question_id
  to prevent question-level leakage across folds).
- **Optuna TPE** hyperparameter search, 30 trials per outer fold,
  optimising inner-CV F1.
- **Per-fold decision threshold tuning** on inner-CV held-out probabilities,
  applied to outer-test predictions.
- **Class imbalance:** cost-sensitive learning via `class_weight="balanced"`
  (LogReg/SVM) or `scale_pos_weight` (XGBoost). MLP uses no per-class
  weighting and relies on threshold tuning.
- **Three classifier paradigms tested**: Logistic Regression (linear),
  XGBoost (tree-based), SVM with RBF kernel (kernel-based).
- **Explainability:** SHAP applied across all classifiers as the unified
  attribution method (`LinearExplainer` for LogReg, `TreeExplainer` for
  XGBoost, `KernelExplainer` for SVM and MLP). Per-fold mean-|SHAP|
  aggregated across the 5 outer folds.

### Earlier failed approach (replaced by current pipeline)

Before this pipeline, a teammate fine-tuned BERT, RoBERTa, and DistilBERT
end-to-end on the FLIP/HOLD task. F1 was reasonable (0.55–0.59), but
Integrated-Gradients attributions surfaced topic words ("mirror,"
"France," "helicopter") rather than linguistic style, making the
attributions useless for the project's actual research question. The
current pipeline was designed to fix this by freezing the embedding,
hand-engineering interpretable features, and using a small classifier
that cannot memorize topic shortcuts.

---

## Results to date

All numbers are mean ± std across 5 outer folds of nested CV on the
4,797-row filtered dataset (unless noted as the older 2,955-row set).

| Configuration | n features | F1 | AUC | Notes |
|---|---|---|---|---|
| Linguistic-only baseline (50 features, 2,955 rows) | 50 | 0.575 ± 0.025 | 0.636 ± 0.013 | initial linguistic feature set |
| Hybrid (50 ling + 30 PCA, 2,955 rows) | 80 | 0.581 ± 0.037 | 0.649 ± 0.033 | added embedding PCA |
| Hybrid (+5 rebuttal-vs-initial diff features, 2,955 rows) | 85 | 0.587 ± 0.037 | 0.658 ± 0.027 | |
| Hybrid (+6 typography/NER/perspective, 2,955 rows) | 91 | 0.590 ± 0.043 | 0.665 ± 0.033 | |
| **Embedding-only on new merged data (4,797 rows)** | 60 | 0.620 ± 0.017 | **0.599 ± 0.039** | F1 inflated by threshold-tuning collapse into "predict FLIP for everything" |
| **Hybrid on new merged data (4,797 rows)** | 121 | **0.635 ± 0.020** | **0.665 ± 0.028** | best LogReg configuration so far |
| XGBoost hybrid (older 2,955 rows) | 80 | 0.567 ± 0.042 | 0.653 ± 0.021 | non-linear, slightly higher AUC than logreg on same data |

For comparison, the earlier BERT-family fine-tunes:

| Model | F1 | AUC |
|---|---|---|
| BERT | 0.550 | 0.618 |
| RoBERTa | 0.587 | 0.594 |
| DistilBERT | 0.588 | 0.615 |

### Top-20 features by mean |SHAP| (Hybrid LogReg, 4,797-row dataset)

Of the 20 highest-attribution features, **11 are sbert PCA components**
(uninterpretable individually) and **9 are hand-crafted**:
`starts_lower`, `n_clauses`, `n_question_marks`, `n_commas`,
`n_nominalisations`, `n_subord_clauses`, `clauses_per_sentence`,
`vader_pos`, `sim_to_wrong`.

Notably absent from the top: aggression markers, all-caps word count,
exclamations, hedging count, certainty count, authority claims.

---

## Our concerns

### 1. The AUC ceiling

We have iterated through five rounds of feature additions:

| Step | n features | AUC | Δ |
|---|---|---|---|
| baseline | 50 | 0.636 | — |
| + 30 PCA | 80 | 0.649 | +0.013 |
| + 5 diff | 85 | 0.658 | +0.009 |
| + 6 new | 91 | 0.665 | +0.007 |
| 60 PCA + bigger dataset (full) | 121 | 0.665 | 0 |

Diminishing returns are obvious. **Doubling the dataset from 2,955 to
4,797 rows did not move AUC at all** (both 0.665), which strongly suggests
the ceiling is not sample-size-limited. Yet AUC ≈ 0.66 is only modestly
above chance.

### 2. The embedding-only finding

Pure Option 2 (frozen sbert + classical head, no hand-crafted features)
gives AUC 0.599 — meaningfully *worse* than linguistic-only (0.636). This
seems to argue that generic sentence embeddings *do not* encode the
properties our hand-crafted features capture. But this might also reflect
a too-aggressive PCA compression (60 of 384 components, 65% variance
retained). We have not tested the full 384-dim embedding.

### 3. The top-SHAP feature set looks "lifeless"

The features driving the model are largely structural and length-related
(clause counts, punctuation, sentence length), with sentiment and
similarity rounding out the top. The pragmatic / epistemic-stance features
the proposal emphasized — hedging, certainty, authority, aggression — are
mostly absent from the top 15. This might be because:
- those features genuinely don't predict flipping in our data,
- our lexicon-based detectors are too brittle (substring match against
  small hardcoded phrase sets),
- the rebuttals our generators produce are stylistically narrow and
  don't surface enough variation along these axes.

### 4. The robust/brittle filter is principled but lossy

The `[0.10, 0.90]` per-question flip-rate filter removes ~50% of the
training rows by question. We tested removing the filter — AUC barely
moved (0.599 vs 0.636), confirming the filter's premise that the
excluded questions are noise from the rebuttal-features perspective. But
this also means our model's claims only generalize to "shakable"
questions — a meaningful scope restriction we don't currently
acknowledge.

### 5. The reliance on a single target model

All ~10,000 challenges are against Claude Haiku 3.5. We have no
cross-model validation. If we found that "modal uncertainty in the
rebuttal predicts flipping for Haiku 3.5" — does that generalize to
other models? Other model sizes? The proposal doesn't claim multi-model
generalization, but a reviewer should ask whether our SHAP rankings
reflect Haiku-3.5-specific quirks or general sycophancy mechanics.

---

## Specific questions for you

Please address each:

1. **Is our problem formulation right?** Predicting per-rebuttal flip
   from rebuttal-side features only — is this the right way to test
   the proposal's hypothesis? Should the model see Haiku's initial
   response (we have these features available but chose not to use
   them for design reasons)?

2. **What's a realistic AUC ceiling** for this kind of behavior
   prediction task, given a 4,800-row dataset against a single target
   model? Is 0.65–0.70 consistent with what published work has
   achieved on comparable problems? Are we hitting a fundamental
   noise floor or have we mis-designed something?

3. **The embedding-only collapse.** Is the AUC = 0.599 result for pure
   Option 2 a reliable indicator that sentence-transformer embeddings
   lack the relevant signal, or could it be an artifact of our PCA
   compression / classifier choice? Should we try (a) full 384-dim
   embedding, (b) a stronger embedding model (mpnet-base-v2), (c) a
   small MLP head on the embedding, or is this experiment basically
   settled?

4. **The "lifeless" SHAP rankings.** Pragmatic / epistemic-stance
   features (hedging, certainty, aggression, authority) are mostly
   absent from the top of our SHAP rankings, contradicting our
   proposal's hypothesis that these matter. Is this a feature-design
   problem (substring-match lexicons too crude), a generator-diversity
   problem (Gemini + Gemma produce stylistically narrow rebuttals),
   or a genuine empirical finding (form > rhetoric)? How would the
   literature interpret this?

5. **Single-target generalization.** All challenges are against Haiku
   3.5. The proposal doesn't claim cross-model generalization but
   should we test on a second target before publishing the SHAP
   rankings as findings? Which one?

6. **Robust/brittle filter.** Is the `[0.10, 0.90]` cutoff
   methodologically defensible? Have we biased our findings toward
   "shakable questions only"? How would a reviewer respond?

7. **What experiments would meaningfully advance this?** Rank the
   following by expected impact and tell us which to do first:
   - Target-response features (Haiku's initial-answer linguistic
     properties as input to the classifier)
   - Sbert upgrade to `all-mpnet-base-v2`
   - Adding a second target model (DistilGPT, Llama, etc.) for
     cross-model SHAP comparison
   - Concept-based attribution (TCAV) instead of feature-based SHAP
   - Counterfactual rebuttal generation (find minimal text edits that
     flip the prediction)
   - More embedding PCA components (60 → 150 or full 384-dim)
   - A small fine-tuned transformer (DistilBERT) with TCAV-style
     attribution against our linguistic concepts (as a
     methodological control)

8. **Are there obvious published methods we are missing?** Any
   technique from sycophancy literature, prompt-sensitivity literature,
   or LLM-behavior-prediction literature that should be incorporated?

9. **Methodological holes.** What would a reviewer at *EMNLP /
   NeurIPS / TACL* flag as a serious gap in our methodology that we
   haven't addressed?

10. **One paragraph: should we publish what we have or keep iterating?**
    Be honest. If you would publish, what's the headline finding?
    If you wouldn't, what's the minimum additional experiment required
    to make this defensible?

Please be direct, specific, and cite literature where applicable. We
prefer painful feedback now over silent ones at submission.
