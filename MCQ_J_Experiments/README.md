# MCQ_J_Experiments

All experiments run on the linguistic-feature sycophancy-flip prediction
pipeline. Self-contained envelope of results, methodology notes, and
the critical review that shaped the iteration.

## Contents

| File / directory | What it is |
|---|---|
| `EXPERIMENTS.md` | Methodology + results writeup. Setup, three feature configurations (linguistic, embedding-only, hybrid), result tables, SHAP top-20, and the interpretation of the findings. |
| `CRITICAL_REVIEW_PROMPT.md` | The prompt used to request an external critical review of the project (with specific questions, expected literature, and decision points). |
| `REVIEWER_RESPONSE.md` | The reviewer's response — independent assessment of the project, including ranked next experiments, methodological holes, and a publish/iterate verdict. |
| `results/` | One subdirectory per training run. Each contains `results.json` (per-fold metrics + best hyperparameters), `shap_values.json` (mean \|SHAP\| per feature across folds), `predictions.csv` (out-of-fold prediction + probability per rebuttal), `roc_curve.png` (out-of-fold ROC). |

## The runs at a glance

All runs use nested 5-outer × 3-inner StratifiedGroupKFold cross-validation,
30 Optuna trials per outer fold, per-fold decision-threshold tuning, and
SHAP attribution. Dataset is the merged team dataset (`MCQ/NEW_MCQ_MERGED_DATASET.json`)
after FLIP/HOLD filtering and the per-question `[0.10, 0.90]` robust/brittle
flip-rate filter.

### Iteration on the LogReg + feature-set axis

| Directory | Classifier | Features | n | Dataset |
|---|---|---|---|---|
| `new_logreg/` | LogReg | linguistic baseline (no embedding, no diff, no typography) | 50 | 2,955 rows / 117 q |
| `new_logreg_pca/` | LogReg | + 30 sbert PCA | 80 | 2,955 |
| `new_logreg_pca_diff/` | LogReg | + 5 rebuttal-vs-initial diff | 85 | 2,955 |
| `new_logreg_91/` | LogReg | + 6 typography / NER / perspective | 91 | 2,955 |
| `new_logreg_emb_only/` | LogReg | embedding only (60 sbert PCA, no hand-crafted) | 60 | 4,797 |
| `new_logreg_hybrid/` | LogReg | full hybrid (linguistic + embedding + diff + typography) | 121 | 4,797 |
| `new_logreg_linguistic_v3/` | LogReg | linguistic-only with 5 data-grounded features | 70 | 4,797 |
| `new_logreg_linguistic_67/` | LogReg | **linguistic-only curated** (drops 3 dead data-grounded features) | **67** | **4,797** |

### Cross-classifier comparisons on the curated linguistic-only set

| Directory | Classifier | Features | n | AUC |
|---|---|---|---|---|
| `new_elasticnet_linguistic/` | Elasticnet (L1+L2) | linguistic baseline | 61 | 0.643 |
| `new_elasticnet_linguistic_v2/` | Elasticnet | + 4 discourse features | 65 | 0.649 |
| `new_elasticnet_linguistic_v3/` | Elasticnet | + 5 data-grounded features | 70 | 0.646 |
| `new_logreg_linguistic_67/` | LogReg | curated 67 | 67 | 0.648 |
| `new_xgboost_linguistic_67/` | XGBoost | curated 67 | 67 | 0.645 |

### Hybrid + non-linear comparison

| Directory | Classifier | Features | n | AUC |
|---|---|---|---|---|
| `new_xgboost/` | XGBoost | full hybrid | 80 | 0.653 |
| `new_mlp_hybrid/` | MLP (Optuna-tuned hidden sizes) | full hybrid | 121 | 0.621 |

## Headline finding

The AUC ceiling clusters at **0.645 ± 0.005 across linear, tree-based,
and kernel-based classifiers** on the 67 curated hand-crafted linguistic
features. Adding 60-component sbert PCA bumps AUC to 0.665 in the hybrid
LogReg setup, but pure embedding-only is much worse (AUC 0.599), confirming
the hand-crafted features are doing meaningful work.

The headline configuration recommended for the proposal-aligned writeup is
`new_logreg_linguistic_67/` (or `new_xgboost_linguistic_67/`): 67
interpretable hand-engineered features, fully interpretable SHAP rankings
spanning all six proposal linguistic levels, ~0.645 AUC.

## Where the code that produced these lives

- Training pipeline: `MCQsmallmodels/train_smallmodels.py` (in the original
  shared location — `--model {logreg,xgboost,svm-rbf,mlp,elasticnet}`,
  `--features {full,embedding,linguistic}`).
- Data merge: `MCQ/merge_mcq.py`.
- CSV preparation: `MCQsmallmodels/mcq_to_csv.py`.
- Source merged dataset: `MCQ/NEW_MCQ_MERGED_DATASET.json` (10,321 challenges
  at time of these runs).
- Filtered training CSV: `MCQsmallmodels/new_mcq_merged_filtered.csv`.
