"""
Exp 3b: Perturbation uptake analysis (local, no GPU).
Checks whether continuations reference/reuse the perturbed value or claim.

This is a text-matching analysis on existing continuations — no new generation needed.

Outputs: results/robustness/exp3b_perturbation_uptake.json
"""

import json
import re
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "robustness"
RESULTS.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"exp3b_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)


def extract_numbers(text):
    """Extract all numbers from text."""
    return set(re.findall(r'\b\d+(?:\.\d+)?\b', text))


def extract_key_phrases(text, max_words=5):
    """Extract distinctive phrases (lowercased, stripped)."""
    words = text.lower().split()
    phrases = set()
    for i in range(len(words)):
        for j in range(i+1, min(i+max_words+1, len(words)+1)):
            phrase = " ".join(words[i:j])
            if len(phrase) > 5:  # skip very short
                phrases.add(phrase)
    return phrases


def compute_uptake(d):
    """Measure how much the continuation references the perturbed step vs original."""
    perturbed_step = d.get("perturbed_step", "")
    original_step = d.get("original_step", "")
    continuation = d.get("continuation", "")

    if not continuation or not perturbed_step:
        return None

    cont_lower = continuation.lower()

    # Numerical uptake: do numbers from perturbed step appear in continuation?
    perturbed_nums = extract_numbers(perturbed_step)
    original_nums = extract_numbers(original_step)
    cont_nums = extract_numbers(continuation)

    # Numbers unique to perturbation (not in original)
    perturb_only_nums = perturbed_nums - original_nums
    original_only_nums = original_nums - perturbed_nums

    perturb_nums_in_cont = perturb_only_nums & cont_nums
    original_nums_in_cont = original_only_nums & cont_nums

    # Textual uptake: key words from perturbed step in continuation
    # Focus on content words that differ between original and perturbed
    perturb_words = set(perturbed_step.lower().split()) - set(original_step.lower().split())
    original_words = set(original_step.lower().split()) - set(perturbed_step.lower().split())

    # Remove common stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would", "could",
                 "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                 "on", "with", "at", "by", "from", "as", "into", "through", "that",
                 "this", "it", "and", "or", "but", "not", "no", "if", "then", "so"}
    perturb_words -= stopwords
    original_words -= stopwords

    cont_words = set(cont_lower.split())
    perturb_words_in_cont = perturb_words & cont_words
    original_words_in_cont = original_words & cont_words

    return {
        "perturb_nums_found": len(perturb_nums_in_cont),
        "perturb_nums_total": len(perturb_only_nums),
        "original_nums_found": len(original_nums_in_cont),
        "original_nums_total": len(original_only_nums),
        "perturb_words_found": len(perturb_words_in_cont),
        "perturb_words_total": len(perturb_words),
        "original_words_found": len(original_words_in_cont),
        "original_words_total": len(original_words),
    }


def get_final_label(d):
    jl = d.get("judge_label")
    if jl == "A": return "A"
    elif jl == "B": return "B"
    elif jl == "C": return "C"
    lc = d.get("label_3class", "")
    if lc == "error_propagation": return "C"
    elif lc == "silent_bypass": return "A"
    elif lc == "self_correction": return "B"
    return "unclear"


def main():
    logging.info("Loading expanded_pairs.json...")
    with open(DATA / "expanded_pairs.json") as f:
        data = json.load(f)
    logging.info(f"Loaded {len(data)} examples")

    for d in data:
        d["final_label"] = get_final_label(d)

    labeled = [d for d in data if d["final_label"] != "unclear"]
    logging.info(f"Labeled: {len(labeled)}")

    # Compute uptake for each example
    results_by_source = defaultdict(lambda: {"uptake_scores": [], "labels": []})
    results_by_label = defaultdict(lambda: {"uptake_scores": []})

    for d in labeled:
        uptake = compute_uptake(d)
        if uptake is None:
            continue

        # Compute a simple uptake score: fraction of perturb-unique tokens found in continuation
        num_denom = uptake["perturb_nums_total"] + uptake["perturb_words_total"]
        num_found = uptake["perturb_nums_found"] + uptake["perturb_words_found"]
        score = num_found / max(num_denom, 1)

        results_by_source[d["source"]]["uptake_scores"].append(score)
        results_by_source[d["source"]]["labels"].append(d["final_label"])
        results_by_label[d["final_label"]]["uptake_scores"].append(score)

    # Report
    logging.info("\n=== Perturbation uptake by dataset ===")
    ds_summary = {}
    for source in ["gsm8k", "mmlu", "bbh"]:
        scores = results_by_source[source]["uptake_scores"]
        labels = results_by_source[source]["labels"]
        if not scores:
            continue
        mean_uptake = np.mean(scores)
        c_scores = [s for s, l in zip(scores, labels) if l == "C"]
        a_scores = [s for s, l in zip(scores, labels) if l == "A"]
        ds_summary[source] = {
            "n": len(scores),
            "mean_uptake": round(float(mean_uptake), 3),
            "type_c_mean_uptake": round(float(np.mean(c_scores)), 3) if c_scores else None,
            "type_a_mean_uptake": round(float(np.mean(a_scores)), 3) if a_scores else None,
        }
        logging.info(f"  {source}: mean uptake={mean_uptake:.3f} (n={len(scores)})")
        logging.info(f"    Type C: {np.mean(c_scores):.3f} (n={len(c_scores)})" if c_scores else "    Type C: N/A")
        logging.info(f"    Type A: {np.mean(a_scores):.3f} (n={len(a_scores)})" if a_scores else "    Type A: N/A")

    logging.info("\n=== Perturbation uptake by behavioral type ===")
    label_summary = {}
    for label in ["A", "B", "C"]:
        scores = results_by_label[label]["uptake_scores"]
        if not scores:
            continue
        label_summary[label] = {
            "n": len(scores),
            "mean": round(float(np.mean(scores)), 3),
            "median": round(float(np.median(scores)), 3),
            "std": round(float(np.std(scores)), 3),
        }
        logging.info(f"  {label}: mean={np.mean(scores):.3f}, median={np.median(scores):.3f} (n={len(scores)})")

    # Within-MMLU subject analysis
    logging.info("\n=== MMLU subject uptake (Type C only) ===")
    mmlu_c = [d for d in labeled if d["source"] == "mmlu" and d["final_label"] == "C"]
    subj_uptake = defaultdict(list)
    for d in mmlu_c:
        uptake = compute_uptake(d)
        if uptake is None: continue
        num_denom = uptake["perturb_nums_total"] + uptake["perturb_words_total"]
        num_found = uptake["perturb_nums_found"] + uptake["perturb_words_found"]
        score = num_found / max(num_denom, 1)
        subj_uptake[d.get("subject", "unknown")].append(score)

    subj_summary = {}
    for subj, scores in sorted(subj_uptake.items(), key=lambda x: np.mean(x[1])):
        if len(scores) >= 10:
            subj_summary[subj] = {"n": len(scores), "mean": round(float(np.mean(scores)), 3)}
            logging.info(f"  {subj}: uptake={np.mean(scores):.3f} (n={len(scores)})")

    # Save
    output = {
        "timestamp": datetime.now().isoformat(),
        "dataset_summary": ds_summary,
        "label_summary": label_summary,
        "mmlu_subject_uptake": subj_summary,
    }
    outpath = RESULTS / "exp3b_perturbation_uptake.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)
    logging.info(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
