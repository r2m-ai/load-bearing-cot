"""
V9 Fix: Correct BBH misclassification caused by answer format mismatch.

Bug: check_correct() in 2_create_perturbations.py compared extracted answer ("A")
against correct_letter field, which is empty for BBH. BBH's reference_answer uses
"(A)" format. Result: 3,229 correct BBH answers misclassified as Type C.

This script:
1. Identifies misclassified BBH examples (answered correctly but labeled error_propagation)
2. Re-classifies them using rule-based heuristics (same as 2b_subclassify.py)
3. Sends NEEDS_JUDGE cases to Claude Haiku for A/B classification
4. Updates expanded_pairs.json with corrected labels
5. Recomputes all statistics

Usage:
  python scripts/v9/fix_bbh_labels.py [--dry-run] [--concurrency N]
    --dry-run: analyze without modifying data or calling API
    --concurrency: parallel API calls (default: 40)

Requires: ANTHROPIC_API_KEY env var (unless --dry-run)
"""

import json
import re
import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "v9"
RESULTS.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"v9_fix_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)


def normalize_bbh_answer(ans: str) -> str:
    """Normalize BBH answer format: '(A)' -> 'A', 'True' -> 'TRUE', etc."""
    ans = str(ans).strip()
    m = re.match(r'^\(([A-E])\)$', ans)
    if m:
        return m.group(1).upper()
    if len(ans) == 1 and ans.upper() in "ABCDE":
        return ans.upper()
    return ans.upper()


# ---- Rule-based classification (mirrors 2b_subclassify.py) ----

CORRECTION_KEYWORDS = [
    "wait", "actually", "however", "but", "correction", "mistake",
    "error", "wrong", "incorrect", "let me reconsider", "recalculate",
    "should be", "not ", "instead", "oops", "hold on", "re-read",
    "looking back", "upon reflection", "i made", "that's not right",
    "double-check", "rechecking",
]


def classify_bbh_rule_based(pair: dict) -> tuple[str, str]:
    """Classify a BBH example that answered correctly into Type A / B / NEEDS_JUDGE."""
    continuation = pair.get("continuation", "")
    cont_lower = continuation.lower()

    has_correction_keywords = any(kw in cont_lower for kw in CORRECTION_KEYWORDS)

    # Check if the model references the perturbed step content
    perturbed_step = pair.get("perturbed_step", "")
    original_step = pair.get("original_step", "")

    # For BBH: check if continuation contains correction-like language
    if has_correction_keywords:
        return "TYPE_B", "correction_keywords"

    # If no correction keywords, likely silent bypass
    return "NEEDS_JUDGE", "bbh_needs_judge"


# ---- LLM Judge ----

JUDGE_PROMPT = """Classify how a language model handled a perturbed reasoning step.
The model saw a WRONG step but still gave the CORRECT final answer. Did it:
A) SILENT_BYPASS — ignored the wrong step, continued as if it was correct
B) SELF_CORRECTION — noticed the error and corrected it

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "A" or "B"."""


async def judge_batch(pairs, api_key, concurrency=40):
    """Send pairs to Claude Haiku for A/B classification."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    results = {}
    total_tokens = 0

    async def judge_one(pair):
        nonlocal total_tokens
        async with semaphore:
            prompt = JUDGE_PROMPT.format(
                original_step=pair.get("original_step", "N/A")[:500],
                perturbed_step=pair.get("perturbed_step", "N/A")[:500],
                continuation=pair.get("continuation", "N/A")[:800],
            )
            for attempt in range(3):
                try:
                    response = await client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=5,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.content[0].text.strip().upper()
                    total_tokens += response.usage.input_tokens + response.usage.output_tokens
                    label = "A" if text.startswith("A") else ("B" if text.startswith("B") else None)
                    results[pair["id"] + "|" + pair.get("perturbation_strategy", "") + "|" + pair.get("perturbation_point", "")] = label
                    return
                except Exception as e:
                    if "rate" in str(e).lower() or "429" in str(e):
                        await asyncio.sleep(min(2.0, 0.1 * (2 ** attempt)))
                    elif attempt == 2:
                        results[pair["id"] + "|" + pair.get("perturbation_strategy", "") + "|" + pair.get("perturbation_point", "")] = None
                    else:
                        await asyncio.sleep(0.5)

    t0 = time.time()
    chunk_size = 500
    for i in range(0, len(pairs), chunk_size):
        chunk = [judge_one(pairs[j]) for j in range(i, min(i + chunk_size, len(pairs)))]
        await asyncio.gather(*chunk)
        logging.info(f"  Judged {min(i + chunk_size, len(pairs))}/{len(pairs)}")

    elapsed = time.time() - t0
    logging.info(f"  Judge done in {elapsed:.0f}s, {total_tokens} tokens")
    return results, total_tokens


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, no API calls or data changes")
    parser.add_argument("--concurrency", type=int, default=40)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and not args.dry_run:
        logging.error("Set ANTHROPIC_API_KEY environment variable (or use --dry-run)")
        return

    # Load data
    logging.info("Loading expanded_pairs.json...")
    with open(DATA / "expanded_pairs.json") as f:
        data = json.load(f)

    bbh = [d for d in data if d["source"] == "bbh"]
    logging.info(f"Total examples: {len(data)}, BBH: {len(bbh)}")

    # ---- Step 1: Identify misclassified BBH examples ----
    logging.info("\n=== Step 1: Identify misclassified BBH examples ===")

    misclassified = []
    for d in bbh:
        if d.get("label_3class") != "error_propagation":
            continue
        ref = d.get("reference_answer", "")
        pert = d.get("perturbed_answer")
        if pert is None:
            continue
        if normalize_bbh_answer(ref) == normalize_bbh_answer(str(pert)):
            misclassified.append(d)

    logging.info(f"Misclassified BBH examples (correct answer, labeled as Type C): {len(misclassified)}")

    # Breakdown
    ref_formats = Counter()
    for d in misclassified:
        ref = d.get("reference_answer", "")
        if re.match(r'^\([A-E]\)$', ref):
            ref_formats["(X) format"] += 1
        elif ref.lower() in ("true", "false", "yes", "no"):
            ref_formats["boolean"] += 1
        else:
            ref_formats["other: " + ref[:20]] += 1
    logging.info(f"Reference answer formats: {dict(ref_formats)}")

    # ---- Step 2: Rule-based classification ----
    logging.info("\n=== Step 2: Rule-based classification ===")

    type_b_rule = []
    needs_judge = []
    for d in misclassified:
        subtype, reason = classify_bbh_rule_based(d)
        d["_v9_rule_subtype"] = subtype
        d["_v9_rule_reason"] = reason
        if subtype == "TYPE_B":
            type_b_rule.append(d)
        else:
            needs_judge.append(d)

    logging.info(f"Rule-based Type B: {len(type_b_rule)}")
    logging.info(f"Needs judge: {len(needs_judge)}")

    if args.dry_run:
        logging.info("\n=== DRY RUN: Estimating corrected statistics ===")
        # Assume all needs_judge → Type A (conservative for error prop rate)
        corrected_a = len(misclassified)  # Upper bound
        original_c = sum(1 for d in bbh if d.get("label_3class") == "error_propagation")
        original_a = sum(1 for d in bbh if d.get("label_3class") == "silent_bypass")
        corrected_c = original_c - len(misclassified)

        logging.info(f"\nOriginal BBH: A={original_a}, B=0, C={original_c}")
        logging.info(f"Corrected BBH (worst case): A={original_a + corrected_a}, B=?, C={corrected_c}")
        logging.info(f"Original C rate: {original_c / (original_a + original_c) * 100:.1f}%")
        logging.info(f"Corrected C rate (upper bound): {corrected_c / (original_a + original_c) * 100:.1f}%")

        # Save analysis
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": True,
            "misclassified_count": len(misclassified),
            "rule_based_type_b": len(type_b_rule),
            "needs_judge": len(needs_judge),
            "original_counts": {"A": original_a, "B": 0, "C": original_c},
            "corrected_c_count": corrected_c,
            "corrected_c_rate": round(corrected_c / (original_a + original_c) * 100, 1),
        }
        with open(RESULTS / "v9_bbh_fix_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        logging.info(f"Analysis saved to {RESULTS / 'v9_bbh_fix_analysis.json'}")
        return

    # ---- Step 3: LLM Judge for ambiguous cases ----
    logging.info(f"\n=== Step 3: Judging {len(needs_judge)} examples with Claude Haiku ===")
    judge_results, tokens = await judge_batch(needs_judge, api_key, args.concurrency)
    logging.info(f"Judge results: {Counter(judge_results.values())}")

    # ---- Step 4: Apply corrections to data ----
    logging.info("\n=== Step 4: Applying corrections ===")

    # Build lookup for misclassified examples
    misclassified_keys = set()
    for d in misclassified:
        key = d["id"] + "|" + d.get("perturbation_strategy", "") + "|" + d.get("perturbation_point", "")
        misclassified_keys.add(key)

    corrections = {"A": 0, "B": 0, "unchanged": 0}

    for d in data:
        if d["source"] != "bbh":
            continue
        key = d["id"] + "|" + d.get("perturbation_strategy", "") + "|" + d.get("perturbation_point", "")
        if key not in misclassified_keys:
            continue

        # Check rule-based first
        rule_subtype = d.get("_v9_rule_subtype")
        if rule_subtype == "TYPE_B":
            d["label_3class"] = "self_correction"
            d["subtype"] = "TYPE_B"
            d["subtype_reason"] = "v9_rule_correction_keywords"
            d["judge_label"] = "B"
            d["behavior"] = "self_corrects"
            corrections["B"] += 1
        else:
            # Use judge result
            judge_label = judge_results.get(key)
            if judge_label == "B":
                d["label_3class"] = "self_correction"
                d["subtype"] = "TYPE_B"
                d["subtype_reason"] = "v9_judge"
                d["judge_label"] = "B"
                d["behavior"] = "self_corrects"
                corrections["B"] += 1
            elif judge_label == "A":
                d["label_3class"] = "silent_bypass"
                d["subtype"] = "TYPE_A"
                d["subtype_reason"] = "v9_judge"
                d["judge_label"] = "A"
                d["behavior"] = "self_corrects"
                corrections["A"] += 1
            else:
                # Judge failed, default to Type A (conservative)
                d["label_3class"] = "silent_bypass"
                d["subtype"] = "TYPE_A"
                d["subtype_reason"] = "v9_judge_fallback"
                d["judge_label"] = "A"
                d["behavior"] = "self_corrects"
                corrections["A"] += 1

        # Clean up temp fields
        d.pop("_v9_rule_subtype", None)
        d.pop("_v9_rule_reason", None)

    logging.info(f"Corrections applied: {dict(corrections)}")

    # ---- Step 5: Save corrected data ----
    logging.info("\n=== Step 5: Saving corrected data ===")

    # Backup original
    backup_path = DATA / "expanded_pairs_pre_v9.json"
    if not backup_path.exists():
        import shutil
        shutil.copy2(DATA / "expanded_pairs.json", backup_path)
        logging.info(f"Backed up original to {backup_path}")

    with open(DATA / "expanded_pairs.json", "w") as f:
        json.dump(data, f, indent=2)
    logging.info(f"Saved corrected expanded_pairs.json ({len(data)} examples)")

    # ---- Step 6: Recompute statistics ----
    logging.info("\n=== Step 6: Recomputed statistics ===")

    bbh_corrected = [d for d in data if d["source"] == "bbh"]
    labels = Counter(d.get("label_3class", "") for d in bbh_corrected)
    logging.info(f"BBH label distribution: {dict(labels)}")

    total_classified = labels.get("silent_bypass", 0) + labels.get("self_correction", 0) + labels.get("error_propagation", 0)
    a_rate = labels.get("silent_bypass", 0) / total_classified * 100
    b_rate = labels.get("self_correction", 0) / total_classified * 100
    c_rate = labels.get("error_propagation", 0) / total_classified * 100
    logging.info(f"BBH: A={a_rate:.1f}%, B={b_rate:.1f}%, C={c_rate:.1f}%")

    # All datasets
    for source in ["gsm8k", "mmlu", "bbh"]:
        subset = [d for d in data if d["source"] == source]
        sl = Counter(d.get("label_3class", "") for d in subset)
        st = sl.get("silent_bypass", 0) + sl.get("self_correction", 0) + sl.get("error_propagation", 0)
        if st == 0:
            continue
        logging.info(f"{source}: A={sl.get('silent_bypass',0)/st*100:.1f}% B={sl.get('self_correction',0)/st*100:.1f}% C={sl.get('error_propagation',0)/st*100:.1f}% (n={st})")

    # BBH subtask breakdown
    logging.info("\nBBH subtask breakdown:")
    subtask_labels = defaultdict(Counter)
    for d in bbh_corrected:
        # BBH subtask is derived from question or id
        subtask = d.get("subtype_reason", "unknown")
        subtask_labels[d.get("label_3class", "")]["total"] += 1

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "bug_description": "BBH reference_answer format '(A)' vs perturbed_answer 'A' caused 3229 misclassifications",
        "misclassified_count": len(misclassified),
        "corrections": dict(corrections),
        "judge_tokens": tokens,
        "estimated_cost_usd": round(tokens * 0.00000025, 2),
        "corrected_bbh_labels": dict(labels),
        "corrected_rates": {
            "bbh": {"A": round(a_rate, 1), "B": round(b_rate, 1), "C": round(c_rate, 1)},
        },
        "all_dataset_rates": {},
    }
    for source in ["gsm8k", "mmlu", "bbh"]:
        subset = [d for d in data if d["source"] == source]
        sl = Counter(d.get("label_3class", "") for d in subset)
        st = sl.get("silent_bypass", 0) + sl.get("self_correction", 0) + sl.get("error_propagation", 0)
        if st == 0:
            continue
        results["all_dataset_rates"][source] = {
            "A": round(sl.get("silent_bypass", 0) / st * 100, 1),
            "B": round(sl.get("self_correction", 0) / st * 100, 1),
            "C": round(sl.get("error_propagation", 0) / st * 100, 1),
            "n": st,
        }

    with open(RESULTS / "v9_bbh_fix_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nResults saved to {RESULTS / 'v9_bbh_fix_results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
