"""Assemble the Hugging Face dataset release from the artifacts in this repository.

Writes the file layout expected by https://huggingface.co/datasets/ReneeJia/cot-load-bearingness
into a target directory, which is a separate git clone of that dataset repository.

    python experiments/release/build_hf_dataset.py --out ../cot-load-bearingness

Nothing here reruns inference or reclassifies anything. Each config is a
reshaping of files already released under data/ and results/; the source of every
row is recorded in the dataset card.
"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Columns kept from the paper's continuation records, in output order.
CONTINUATION_FIELDS = [
    "id", "model", "source", "subject", "question", "choices", "reference_answer",
    "prompt", "baseline_cot", "baseline_answer", "total_steps", "target_step_idx",
    "perturbation_point", "perturbation_strategy", "perturbation_description",
    "original_step", "perturbed_step", "perturbed_prefix", "continuation",
    "perturbed_answer", "rule_label", "judge_label", "final_label",
]
RENAMED = {"baseline_cot": "faithful_cot", "baseline_answer": "faithful_answer",
           "perturbed_prefix": "prefix", "rule_label": "label_3class"}
LABEL_FROM_JUDGE = {"A": "silent_bypass", "B": "self_correction", "C": "error_propagation"}
# The pilot metadata writes TYPE_A/B/C where the n = 500 study writes A/B/C.
SHORT_LABEL = {"TYPE_A": "A", "TYPE_B": "B", "TYPE_C": "C", "UNCLEAR": "U"}


def normalize(value):
    """Empty CSV cells become null; TYPE_A-style labels become A."""
    value = (value or "").strip()
    return SHORT_LABEL.get(value, value) or None


def parse_variants(value):
    """'original=A|strict_bypass=B' -> {'original': 'A', 'strict_bypass': 'B'}."""
    if not value:
        return None
    return dict(pair.split("=", 1) for pair in value.split("|") if "=" in pair)


SAMPLES = [("gemma-2-9b-it", "data/samples/gemma_sample_50.json"),
           ("llama-3.1-8b-instruct", "data/samples/llama_sample_50.json")]

# Aggregate summaries copied verbatim beside the row-level configs.
AGGREGATES = [
    ("results/controls/variance_decomposition/results.json", "variance_decomposition.json"),
    ("data/annotations/study_n500/analysis/results_n500.json", "human_agreement_n500.json"),
    ("data/annotations/pilot_n200/results_n200.json", "human_agreement_pilot_n200.json"),
    ("results/probes/human_agreement/results.json", "probe_vs_human_agreement.json"),
    ("figures/table4_behavioral.csv", "behavioral_taxonomy.csv"),
    ("figures/table6_probe_judged.csv", "probe_layer_sweep.csv"),
    ("figures/table7_intervention.csv", "steering_dose_response.csv"),
]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {path.name}: {len(rows)} rows")
    return len(rows)


def build_continuations(out):
    """One row per perturbed continuation, with the paper's rule and judge labels."""
    rows = []
    for model, source in SAMPLES:
        for record in json.loads((ROOT / source).read_text()):
            row = {}
            for field in CONTINUATION_FIELDS:
                if field == "model":
                    row[field] = model
                elif field == "final_label":
                    row[field] = LABEL_FROM_JUDGE.get(record.get("judge_label")) or record.get("label_3class")
                else:
                    row[field] = record.get(RENAMED.get(field, field))
            rows.append(row)
    return write_jsonl(out / "continuations" / "sample_100.jsonl", rows)


def build_judge_prompt_variants(out):
    """One row per non-Type-C continuation, labeled under each of four judge prompts."""
    source = json.loads((ROOT / "results/validation/judge_prompt_variants/stratification.json").read_text())
    rows = []
    for tier, records in source.items():
        if tier == "incomplete":
            continue
        for record in records:
            rows.append({
                "id": record["id"],
                "source": record.get("source"),
                "perturbation_strategy": record.get("perturbation_strategy"),
                "perturbation_point": record.get("perturbation_point"),
                "variant_labels": record.get("labels"),
                "agreement_tier": tier,
                "consensus": record.get("consensus") or record.get("majority"),
            })
    return write_jsonl(out / "judge_prompt_variants" / "variants.jsonl", rows)


def build_judge_validation(out):
    """Hand-labeled cases used to validate the production judge."""
    rows = json.loads((ROOT / "data/processed/judge_validation.json").read_text())
    return write_jsonl(out / "judge_validation" / "cases.jsonl", rows)


def build_human_annotation(out):
    """Blinded annotation sheets joined to the hidden per-row metadata."""
    studies = [
        ("pilot_n200", "data/annotations/pilot_n200/data_to_label.csv",
         "data/annotations/pilot_n200/_analysis_metadata.csv"),
        ("study_n500", "data/annotations/study_n500/for_annotators/annotator1/data_to_label_annotator1.csv",
         "data/annotations/study_n500/analysis/_analysis_metadata.csv"),
    ]
    rows = []
    for study, sheet, metadata in studies:
        hidden = {r["annotation_id"]: r for r in csv.DictReader((ROOT / metadata).open())}
        for record in csv.DictReader((ROOT / sheet).open()):
            meta = hidden.get(record["annotation_id"], {})
            rows.append({
                "study": study,
                "annotation_id": record["annotation_id"],
                "source": record["source"],
                "prompt": record["prompt"],
                "original_step": record["original_step"],
                "perturbed_step": record["perturbed_step"],
                "continuation": record["continuation"],
                "reference_answer": record["reference_answer"],
                "extracted_answer": record["extracted_answer"],
                "stratum": meta.get("stratum"),
                "perturbation_point": meta.get("perturbation_point"),
                "strategy": meta.get("strategy"),
                "paper_label": normalize(meta.get("final_label") or meta.get("auto_label")),
                "labeled_by": meta.get("labeled_by") or ("judge" if normalize(meta.get("judge_label")) else "rule"),
                "judge_label": normalize(meta.get("judge_label")),
                "variant_labels": parse_variants(meta.get("variant_labels")),
            })
    return write_jsonl(out / "human_annotation" / "rows.jsonl", rows)


def copy_aggregates(out):
    destination = out / "aggregates"
    destination.mkdir(parents=True, exist_ok=True)
    for source, name in AGGREGATES:
        (destination / name).write_bytes((ROOT / source).read_bytes())
    print(f"  aggregates/: {len(AGGREGATES)} files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path,
                        help="target directory (a clone of the Hugging Face dataset repository)")
    arguments = parser.parse_args()
    out = arguments.out.expanduser().resolve() / "data"
    print(f"Building into {out}")
    counts = {
        "continuations": build_continuations(out),
        "judge_prompt_variants": build_judge_prompt_variants(out),
        "judge_validation": build_judge_validation(out),
        "human_annotation": build_human_annotation(out),
    }
    copy_aggregates(out)
    print("Row counts:", json.dumps(counts))


if __name__ == "__main__":
    main()
