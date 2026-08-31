"""
Script 0: Download GSM8K + MMLU datasets.

Downloads from HuggingFace Hub:
- GSM8K: Full test set (~1.3K), filtered for multi-step problems (3+ calculation steps)
- MMLU: Full test set (~14K), all subjects included

Only filters on quality (minimum reasoning steps for GSM8K), no subsampling.
We'll decide later how many to use for hidden state extraction based on compute budget.
"""

import json
import random
from pathlib import Path

from datasets import load_dataset

# Reproducibility
random.seed(42)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_gsm8k() -> list[dict]:
    """Download full GSM8K test set, filter for multi-step problems."""
    print("Downloading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    print(f"  Full test set: {len(ds)} examples")

    # Filter for multi-step problems (3+ calculation steps)
    results = []
    for example in ds:
        answer_text = example["answer"]
        # Count reasoning steps by looking for lines with calculations (<<...>> markers)
        steps = [line for line in answer_text.split("\n") if line.strip() and "<<" in line]
        if len(steps) >= 3:
            # Extract the final numeric answer
            final_answer = answer_text.split("####")[-1].strip()
            results.append({
                "id": f"gsm8k_{len(results)}",
                "source": "gsm8k",
                "question": example["question"],
                "reference_solution": answer_text,
                "reference_answer": final_answer,
                "n_steps": len(steps),
            })

    print(f"  After filtering (3+ steps): {len(results)} examples")
    step_dist = {}
    for r in results:
        n = r["n_steps"]
        step_dist[n] = step_dist.get(n, 0) + 1
    print(f"  Step distribution: {dict(sorted(step_dist.items()))}")
    return results


def download_mmlu() -> list[dict]:
    """Download full MMLU test set, all subjects."""
    print("Downloading MMLU...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    print(f"  Full test set: {len(ds)} examples")

    results = []
    subject_counts: dict[str, int] = {}
    for example in ds:
        subject = example["subject"]
        choices = example["choices"]
        correct_idx = example["answer"]
        correct_letter = ["A", "B", "C", "D"][correct_idx]

        results.append({
            "id": f"mmlu_{len(results)}",
            "source": "mmlu",
            "subject": subject,
            "question": example["question"],
            "choices": choices,
            "correct_index": correct_idx,
            "correct_letter": correct_letter,
            "reference_answer": choices[correct_idx],
        })

        subject_counts[subject] = subject_counts.get(subject, 0) + 1

    print(f"  Total: {len(results)} examples across {len(subject_counts)} subjects")
    # Show top 10 subjects by count
    top_subjects = sorted(subject_counts.items(), key=lambda x: -x[1])[:10]
    print(f"  Top subjects: {', '.join(f'{s} ({c})' for s, c in top_subjects)}")
    return results


def main():
    gsm8k_data = download_gsm8k()
    mmlu_data = download_mmlu()

    # Save
    gsm8k_path = DATA_DIR / "gsm8k_full.json"
    mmlu_path = DATA_DIR / "mmlu_full.json"

    with open(gsm8k_path, "w") as f:
        json.dump(gsm8k_data, f, indent=2)
    print(f"\nSaved {len(gsm8k_data)} GSM8K questions to {gsm8k_path}")

    with open(mmlu_path, "w") as f:
        json.dump(mmlu_data, f, indent=2)
    print(f"Saved {len(mmlu_data)} MMLU questions to {mmlu_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"GSM8K: {len(gsm8k_data)} questions (multi-step, 3+ reasoning steps)")
    if gsm8k_data:
        avg_steps = sum(e["n_steps"] for e in gsm8k_data) / len(gsm8k_data)
        print(f"  Average reasoning steps: {avg_steps:.1f}")
    print(f"MMLU:  {len(mmlu_data)} questions (all subjects)")
    print(f"Total: {len(gsm8k_data) + len(mmlu_data)} questions ready for CoT generation")
    print(f"\nNote: For hidden state extraction (Exp 2-3), you can subsample")
    print(f"based on compute budget. The full datasets are saved for flexibility.")


if __name__ == "__main__":
    main()
