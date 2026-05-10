"""Merge teammates' MCQ challenge result JSON files into a single training dataset.

Adds a `teammate` field to each challenge for provenance and prefixes every
`rebuttal_id` with the teammate name to avoid collisions (Maha and Munera both
use Q001-style ids that overlap numerically). Verifies all sources target the
same model and warns on cross-teammate question overlap.

To add a new teammate: append a (name, path) tuple to SOURCES below and re-run.
"""

import json
import collections
from datetime import datetime
from pathlib import Path

# (teammate, path-to-challenge-results.json)
SOURCES = [
    ("jay",    "MCQ/jay-mcq/jays_results_haiku3.5.json"),
    ("maha",   "MCQ/mahas-mcq/Haiku/mahas-mcq-challenge-haiku35.json"),
    ("munera", "MCQ/munera's/results_haiku35.json"),
    # ("loly",   "MCQ/<loly-folder>/results_haiku35.json"),  # add when available
]

OUTPUT_PATH = "MCQ/MCQ_MERGED_DATASET.json"


def normalize_model_id(model_id: str) -> str:
    """OpenRouter sometimes uses dashes, sometimes dots for the same Haiku 3.5
    model id (`claude-3-5-haiku` vs `claude-3.5-haiku`). Canonicalize to dots."""
    return model_id.replace("-3-5-", "-3.5-")


def main():
    sources_meta = []
    all_challenges = []
    seen_models = set()
    teammate_questions = {}

    for teammate, path_str in SOURCES:
        path = Path(path_str)
        if not path.exists():
            print(f"⚠ SKIPPING {teammate} — file not found: {path}")
            continue

        with open(path) as f:
            data = json.load(f)
        meta = data.get("metadata", {})
        challenges = data.get("challenges", [])

        canonical_model = normalize_model_id(meta.get("model", ""))
        seen_models.add(canonical_model)

        # Tag each challenge with teammate + namespace its rebuttal_id
        questions_seen = set()
        for c in challenges:
            c["teammate"] = teammate
            original_rid = c.get("rebuttal_id", "")
            c["rebuttal_id"] = f"{teammate}_{original_rid}"
            questions_seen.add(c.get("question", ""))
            all_challenges.append(c)

        teammate_questions[teammate] = questions_seen
        sources_meta.append({
            "teammate": teammate,
            "path": str(path),
            "model": meta.get("model"),
            "total_challenges": len(challenges),
            "flips": meta.get("flips"),
            "holds": meta.get("holds"),
            "ambiguous": meta.get("ambiguous"),
        })

        print(f"Loaded {len(challenges):4d} challenges from {teammate:8s} ({path})")

    # Sanity checks
    if len(seen_models) > 1:
        print(f"\n⚠ WARNING: mixed canonical models: {seen_models}")
    canonical_model = next(iter(seen_models)) if len(seen_models) == 1 else list(seen_models)

    # Pairwise question overlap
    print("\nQuestion overlap (pairwise):")
    names = list(teammate_questions.keys())
    any_overlap = False
    for i, t1 in enumerate(names):
        for t2 in names[i + 1:]:
            overlap = len(teammate_questions[t1] & teammate_questions[t2])
            if overlap:
                any_overlap = True
                print(f"  ⚠ {t1} ∩ {t2}: {overlap} questions in common (duplicates merged into same group_id by trainer)")
            else:
                print(f"  {t1} ∩ {t2}: 0 questions in common")
    if not any_overlap:
        print("  (clean — all teammates picked disjoint TruthfulQA subsets)")

    # Aggregate label distribution
    labels = collections.Counter(c.get("label") for c in all_challenges)

    output = {
        "metadata": {
            "merged_at": datetime.now().isoformat(),
            "model": canonical_model,
            "sources": sources_meta,
            "total_challenges": len(all_challenges),
            "flips": labels.get("FLIP", 0),
            "holds": labels.get("HOLD", 0),
            "ambiguous": labels.get("AMBIGUOUS", 0),
            "errors": labels.get("ERROR", 0),
            "merge_script_version": "1.0",
            "notes": "rebuttal_id prefixed with teammate name to avoid collisions; "
                     "teammate field added per challenge for provenance.",
        },
        "challenges": all_challenges,
    }

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = OUTPUT_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    Path(tmp_path).replace(OUTPUT_PATH)

    flips = labels.get("FLIP", 0)
    holds = labels.get("HOLD", 0)
    eff = flips / (flips + holds) * 100 if (flips + holds) else 0

    print(f"\n{'=' * 50}")
    print(f"Merged {len(all_challenges)} challenges from {len(sources_meta)} teammates")
    print(f"  Model:      {canonical_model}")
    print(f"  FLIP:       {flips}")
    print(f"  HOLD:       {holds}")
    print(f"  AMBIGUOUS:  {labels.get('AMBIGUOUS', 0)}")
    print(f"  ERROR:      {labels.get('ERROR', 0)}")
    print(f"  Effective flip rate: {eff:.1f}%")
    print(f"  Output:     {OUTPUT_PATH}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
