#!/usr/bin/env python3
"""
Build the camera-ready n=500 two-annotator human-annotation sample
(revision item 2.6: n >= 500, two annotators, Cohen's kappa).

Upgrades over the rebuttal-stage n=200 package
(`data/annotations/pilot_n200/`):

  1. TRUE M3 stratification. The n=200 sample predated item 2.0 and used
     `NEEDS_JUDGE -> resolved A/B` as a proxy for ambiguity. This sample
     stratifies directly on 4-variant LLM-judge agreement from
     `results/validation/judge_prompt_variants/stratification.json`:
       - split_2_2    : judge variants split 2-2   (maximally ambiguous)
       - majority_3_1 : judge variants split 3-1   (mildly ambiguous)
       - agree        : all 4 variants agree       (calibration anchors)
  2. TYPE_C strata sampled from `expanded_pairs.json` (Type C was not
     re-judged in exp_2_0). BBH Type C is oversampled because the n=200
     study found the rule-based TYPE_C definition mis-labels genuine
     silent bypass as propagation on BBH (37.8% calibration).
  3. Two annotator files with identical rows, same `annotation_id`, but
     independently shuffled row order (reduces shared order effects).
  4. The 200 already-annotated examples are EXCLUDED (composite key
     (id, strategy, perturbation_point)), so total distinct
     human-annotated examples across both studies = 700.

Deterministic: fixed seed, sorted candidate pools. Re-running reproduces
byte-identical CSVs.

Outputs:
  ../for_annotators/annotator1/data_to_label_annotator1.csv   BLIND
  ../for_annotators/annotator2/data_to_label_annotator2.csv   BLIND
  ./_analysis_metadata.csv       hidden labels — researcher only
  ./sample_manifest.json         stratum counts + coverage stats

Usage:
  python3 data/annotations/study_n500/analysis/build_sample_n500.py
"""

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent          # .../human_annotation_n500/analysis
ANNOT_DIR = HERE.parent / "for_annotators"
REPO = HERE.parent.parent.parent.parent
SEED = 20260829

EXPANDED_PAIRS = REPO / "data" / "processed" / "expanded_pairs.json"
STRATIFICATION = (REPO / "results" / "validation" / "judge_prompt_variants" / "stratification.json")
OLD_META = REPO / "data" / "annotations" / "pilot_n200" / "_analysis_metadata.csv"

# Stratum -> (source, tier, n). Total = 500.
#   60% maximally/mildly ambiguous (split_2_2 + majority_3_1) per M3
#   15% all-4-agree calibration anchors
#   13% Type C (BBH oversampled: rule-bug quantification from n=200)
#   12% random baseline
DESIGN = [
    ("GSM_split22",    "gsm8k", "split_2_2",    60),
    ("MMLU_split22",   "mmlu",  "split_2_2",    60),
    ("BBH_split22",    "bbh",   "split_2_2",    60),
    ("GSM_majority31",  "gsm8k", "majority_3_1", 40),
    ("MMLU_majority31", "mmlu",  "majority_3_1", 40),
    ("BBH_majority31",  "bbh",   "majority_3_1", 40),
    ("GSM_agree",      "gsm8k", "agree",        25),
    ("MMLU_agree",     "mmlu",  "agree",        25),
    ("BBH_agree",      "bbh",   "agree",        25),
    ("GSM_type_c",     "gsm8k", "type_c",       15),
    ("MMLU_type_c",    "mmlu",  "type_c",       15),
    ("BBH_type_c",     "bbh",   "type_c",       35),
    ("RANDOM_baseline", None,   "random",       60),
]

ANNOTATOR_COLS = [
    "annotation_id", "source", "prompt", "original_step", "perturbed_step",
    "continuation", "reference_answer", "correct_letter", "extracted_answer",
    "human_label", "human_confidence", "human_notes",
]

META_COLS = [
    "annotation_id", "source", "example_id", "stratum", "perturbation_point",
    "strategy", "final_label", "labeled_by", "judge_label",
    "variant_agreement", "variant_labels",
]

SUBTYPE_TO_LETTER = {"TYPE_A": "A", "TYPE_B": "B", "TYPE_C": "C",
                     "UNCLEAR": "U"}


def ckey(rec):
    return (rec["id"], rec["perturbation_strategy"], rec["perturbation_point"])


def skey(entry):
    return (entry["id"], entry["perturbation_strategy"],
            entry["perturbation_point"])


def main():
    rng = random.Random(SEED)

    pairs = json.loads(EXPANDED_PAIRS.read_text())
    by_key = {ckey(r): r for r in pairs}
    assert len(by_key) == len(pairs), "composite key not unique"

    strat = json.loads(STRATIFICATION.read_text())
    # variant-agreement tier + per-variant labels for each exp_2_0 record
    variant_info = {}
    for tier in ("agree", "majority_3_1", "split_2_2"):
        for e in strat[tier]:
            variant_info[skey(e)] = (tier, e.get("labels", {}))

    # Exclude the 200 already-annotated examples (n=200 study)
    with open(OLD_META, newline="") as f:
        old_keys = {(r["example_id"], r["strategy"], r["perturbation_point"])
                    for r in csv.DictReader(f)}
    missing_old = [k for k in old_keys if k not in by_key]
    if missing_old:
        print(f"warn: {len(missing_old)} old-n200 keys not in expanded_pairs",
              file=sys.stderr)

    used = set(old_keys)

    def pool_for(source, tier):
        if tier in ("agree", "majority_3_1", "split_2_2"):
            keys = [skey(e) for e in strat[tier]
                    if e["source"] == source]
        elif tier == "type_c":
            keys = [k for k, r in by_key.items()
                    if r["source"] == source and r["subtype"] == "TYPE_C"]
        elif tier == "random":
            keys = [k for k, r in by_key.items()
                    if r["subtype"] != "UNCLEAR"]
        else:
            raise ValueError(tier)
        return sorted(k for k in keys if k in by_key and k not in used)

    sampled = []  # (stratum, key)
    for stratum, source, tier, n in DESIGN:
        pool = pool_for(source, tier)
        if len(pool) < n:
            print(f"warn: stratum {stratum} has only {len(pool)} candidates "
                  f"(< {n}); taking all + topping up from majority_3_1",
                  file=sys.stderr)
            take = list(pool)
            topup_pool = [k for k in pool_for(source, "majority_3_1")
                          if k not in take]
            take += rng.sample(topup_pool, n - len(take))
        else:
            take = rng.sample(pool, n)
        for k in take:
            used.add(k)
            sampled.append((stratum, k))

    assert len(sampled) == 500, f"got {len(sampled)}"
    assert len({k for _, k in sampled}) == 500, "duplicate keys sampled"

    # Canonical order: shuffle once, assign annotation_id 1..500
    rng.shuffle(sampled)

    rows_annot, rows_meta = [], []
    n_empty_answer = 0
    for i, (stratum, key) in enumerate(sampled, start=1):
        r = by_key[key]
        extracted = str(r.get("perturbed_answer") or "").strip()
        if not extracted:
            n_empty_answer += 1
        rows_annot.append({
            "annotation_id": str(i),
            "source": r["source"],
            "prompt": r["prompt"],
            "original_step": r["original_step"],
            "perturbed_step": r["perturbed_step"],
            "continuation": r["continuation"],
            "reference_answer": r["reference_answer"],
            "correct_letter": r.get("correct_letter", ""),
            "extracted_answer": extracted,
            "human_label": "", "human_confidence": "", "human_notes": "",
        })
        tier, vlabels = variant_info.get(key, ("not_covered", {}))
        judge = r.get("judge_label") or ""
        rows_meta.append({
            "annotation_id": str(i),
            "source": r["source"],
            "example_id": r["id"],
            "stratum": stratum,
            "perturbation_point": r["perturbation_point"],
            "strategy": r["perturbation_strategy"],
            "final_label": SUBTYPE_TO_LETTER[r["subtype"]],
            "labeled_by": "judge" if judge else "rule",
            "judge_label": judge,
            "variant_agreement": tier,
            "variant_labels": "|".join(
                f"{k}={v or '-'}" for k, v in sorted(vlabels.items())),
        })

    def write_csv(path, cols, rows):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
            w.writeheader()
            w.writerows(rows)

    # Two annotator files: identical rows + ids, different row order.
    # Each annotator gets their own numbered folder (self-contained,
    # instructions included) — the number is the annotator identity.
    for annot_idx in (1, 2):
        subdir = ANNOT_DIR / f"annotator{annot_idx}"
        subdir.mkdir(parents=True, exist_ok=True)
        order = list(rows_annot)
        random.Random(SEED + annot_idx).shuffle(order)
        write_csv(subdir / f"data_to_label_annotator{annot_idx}.csv",
                  ANNOTATOR_COLS, order)

    write_csv(HERE / "_analysis_metadata.csv", META_COLS, rows_meta)

    manifest = {
        "seed": SEED,
        "n": 500,
        "design": [
            {"stratum": s, "source": src, "tier": t, "n": n}
            for s, src, t, n in DESIGN],
        "by_source": dict(Counter(r["source"] for r in rows_meta)),
        "by_stratum": dict(Counter(r["stratum"] for r in rows_meta)),
        "by_final_label": dict(Counter(r["final_label"] for r in rows_meta)),
        "by_variant_agreement": dict(
            Counter(r["variant_agreement"] for r in rows_meta)),
        "n_empty_extracted_answer": n_empty_answer,
        "n_judge_labeled": sum(1 for r in rows_meta if r["judge_label"]),
        "excluded_previous_n200": len(old_keys),
    }
    (HERE / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Wrote 2 annotator files to {ANNOT_DIR}")
    print(f"Wrote metadata + manifest to {HERE}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
