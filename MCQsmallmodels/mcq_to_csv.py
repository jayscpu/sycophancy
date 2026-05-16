"""
Convert MCQ_MERGED_DATASET.json to a CSV compatible with train_smallmodels.py.

  - Filters to FLIP/HOLD only (drops AMBIGUOUS/ERROR).
  - Applies the same robust/brittle filter as 5fold-train-MCQ-noIG.py:
    keeps questions with flip rate in [0.1, 0.9] (drops questions Haiku always
    flips on or never flips on, since they teach the classifier nothing about
    rebuttal style).
  - Generates question_id from MD5 hash of normalized question text.
  - Writes a CSV with: question_id, rebuttal_text, label, batch, wrong_answer,
    correct_answer, initial_answer, teammate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def main(input_json, output_csv):
    with open(input_json, "r") as f:
        data = json.load(f)
    challenges = data.get("challenges", [])
    print(f"Loaded {len(challenges)} challenges from {input_json}")

    rows = []
    for c in challenges:
        q = c.get("question", "")
        rows.append({
            "question_id": hashlib.md5(q.strip().lower().encode()).hexdigest()[:8],
            "rebuttal_text": c.get("rebuttal_text", ""),
            "label": str(c.get("label", "")).upper().strip(),
            "batch": c.get("batch"),
            "wrong_answer": c.get("wrong_answer", ""),
            "correct_answer": c.get("correct_answer", ""),
            "initial_answer": c.get("initial_answer", ""),
            "teammate": c.get("teammate", ""),
            "question": q,
        })
    df = pd.DataFrame(rows)

    before = len(df)
    df = df[df["label"].isin({"FLIP", "HOLD"})].copy()
    print(f"After FLIP/HOLD filter: {len(df)} rows (dropped {before - len(df)} AMBIGUOUS/ERROR)")

    # Robust/brittle filter — match 5fold-train-MCQ-noIG.py
    df["_y"] = (df["label"] == "FLIP").astype(int)
    flip_rate = df.groupby("question_id")["_y"].mean()
    mixed = flip_rate[(flip_rate >= 0.1) & (flip_rate <= 0.9)].index
    n_robust = int((flip_rate < 0.1).sum())
    n_brittle = int((flip_rate > 0.9).sum())
    df = df[df["question_id"].isin(mixed)].drop(columns=["_y"]).reset_index(drop=True)
    print(f"After robust/brittle filter: {len(df)} rows from {df['question_id'].nunique()} questions")
    print(f"  Dropped: {n_robust} robust (flip rate < 10%) + {n_brittle} brittle (flip rate > 90%)")
    flip_n = (df["label"] == "FLIP").sum()
    print(f"  Class balance: FLIP={flip_n} ({flip_n/len(df)*100:.1f}%), HOLD={len(df)-flip_n} ({(len(df)-flip_n)/len(df)*100:.1f}%)")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../MCQ/MCQ_MERGED_DATASET.json")
    parser.add_argument("--output", default="mcq_merged_filtered.csv")
    args = parser.parse_args()
    main(args.input, args.output)
