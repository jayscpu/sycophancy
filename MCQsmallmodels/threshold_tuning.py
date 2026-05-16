"""
Threshold tuning for the three small models.

  - Train each model (same as train_smallmodels.py).
  - Pick the threshold on the validation set that maximizes F1.
  - Apply that threshold to the test set.
  - Report F1@0.5 (current/default) vs F1@val-tuned-threshold.

This is the honest way to threshold-tune: tune on val, evaluate on test.
Sweeping on test alone would overfit.
"""

import os
import json
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# Reuse code from train_smallmodels.py
sys.path.insert(0, os.path.dirname(__file__))
from train_smallmodels import load_and_split, extract_features, train_model


def evaluate_at_threshold(probs, y, t):
    preds = (probs >= t).astype(int)
    return {
        "threshold": float(t),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y, preds)),
    }


def main(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df_train, df_val, df_test = load_and_split(input_csv)

    train_feats, feature_names = extract_features(df_train)
    val_feats, _ = extract_features(df_val)
    test_feats, _ = extract_features(df_test)

    val_feats = val_feats.reindex(columns=feature_names, fill_value=0)
    test_feats = test_feats.reindex(columns=feature_names, fill_value=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feats.values)
    X_val = scaler.transform(val_feats.values)
    X_test = scaler.transform(test_feats.values)

    y_train = df_train["label_encoded"].to_numpy()
    y_val = df_val["label_encoded"].to_numpy()
    y_test = df_test["label_encoded"].to_numpy()

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)

    # Threshold sweep granularity
    thresholds = np.linspace(0.05, 0.95, 91)  # 0.05, 0.06, ..., 0.95

    summary = {}

    for model_name in ["logreg", "xgboost", "svm-rbf"]:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        clf, _ = train_model(
            model_name=model_name,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            class_weights=class_weights,
            output_dir=os.path.join(output_dir, model_name, "checkpoints"),
        )

        val_probs = clf.predict_proba(X_val)[:, 1]
        test_probs = clf.predict_proba(X_test)[:, 1]

        # Sweep on val
        val_sweep = [evaluate_at_threshold(val_probs, y_val, t) for t in thresholds]
        best_val = max(val_sweep, key=lambda r: r["f1"])
        best_thr = best_val["threshold"]

        # Evaluate test at default 0.5 and at val-tuned threshold
        test_default = evaluate_at_threshold(test_probs, y_test, 0.5)
        test_tuned = evaluate_at_threshold(test_probs, y_test, best_thr)

        # AUC (threshold-independent) for reference
        auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else None

        print(f"\n  Best threshold (val): {best_thr:.2f}")
        print(f"  Val F1 at that threshold: {best_val['f1']:.4f}")
        print(f"")
        print(f"  Test F1 @ 0.5 (default):       {test_default['f1']:.4f}  "
              f"(P={test_default['precision']:.3f}, R={test_default['recall']:.3f}, "
              f"acc={test_default['accuracy']:.3f})")
        print(f"  Test F1 @ {best_thr:.2f} (val-tuned):    {test_tuned['f1']:.4f}  "
              f"(P={test_tuned['precision']:.3f}, R={test_tuned['recall']:.3f}, "
              f"acc={test_tuned['accuracy']:.3f})")
        print(f"  Test AUC (threshold-indep): {auc:.4f}")

        summary[model_name] = {
            "best_threshold_on_val": best_thr,
            "val_f1_at_best_threshold": best_val["f1"],
            "test_at_threshold_0.5": test_default,
            "test_at_tuned_threshold": test_tuned,
            "test_roc_auc": auc,
        }

    print(f"\n{'='*60}")
    print("SUMMARY — F1 lift from threshold tuning (test set)")
    print(f"{'='*60}")
    print(f"{'Model':<12}{'F1@0.5':>10}{'F1@tuned':>12}{'Δ':>8}{'thr':>8}{'AUC':>8}")
    for name, r in summary.items():
        d = r["test_at_threshold_0.5"]["f1"]
        t = r["test_at_tuned_threshold"]["f1"]
        thr = r["best_threshold_on_val"]
        auc = r["test_roc_auc"]
        print(f"{name:<12}{d:>10.4f}{t:>12.4f}{t-d:>+8.4f}{thr:>8.2f}{auc:>8.4f}")

    out_path = os.path.join(output_dir, "threshold_tuning_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Summary saved to {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="mcq_merged_filtered.csv")
    parser.add_argument("--output", default="results/threshold_tuning")
    args = parser.parse_args()
    main(args.input, args.output)
