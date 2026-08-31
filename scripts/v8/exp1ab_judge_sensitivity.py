"""
Exp 1a-1b: Judge sensitivity analysis.
Tests 5 prompt variants on existing non-Type-C continuations.
Reports A/B/C stability, A+B vs C stability, gradient slope stability.

Requires: ANTHROPIC_API_KEY env var
Outputs: results/v8/exp1ab_judge_sensitivity.json

Usage: python scripts/v8/exp1ab_judge_sensitivity.py [--sample N] [--concurrency N]
  --sample N: judge only N examples per variant (default: all non-C examples)
  --concurrency N: parallel API calls (default: 40)
"""

import json
import argparse
import asyncio
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

ROOT = Path(__file__).parent.parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "v8"
RESULTS.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"exp1ab_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

# ========== 5 JUDGE PROMPT VARIANTS ==========

PROMPTS = {
    "strict_bypass": """Classify how a language model handled a perturbed reasoning step.
The model saw a WRONG step but still gave the CORRECT final answer.

TYPE_A (SILENT BYPASS): The model shows NO evidence of noticing the perturbation. It continues with correct reasoning as if the wrong step never happened. Even if it uses correct values in computation, classify as A ONLY if there is no reference to or engagement with the perturbation.

TYPE_B (SELF-CORRECTION): The model shows ANY evidence of engaging with the perturbation — using correct values where the perturbation gave wrong ones counts as correction, even without explicit language like "wait" or "actually."

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "A" or "B".""",

    "strict_correction": """Classify how a language model handled a perturbed reasoning step.
The model saw a WRONG step but still gave the CORRECT final answer.

TYPE_A (SILENT BYPASS): The model continues without explicitly acknowledging or correcting the error. This includes cases where the model happens to use correct values through computation — unless it explicitly flags the error, it's bypass.

TYPE_B (SELF-CORRECTION): The model EXPLICITLY acknowledges the error using language like "wait", "actually", "but that's wrong", "however the correct value is", etc. Simply computing with correct values without verbal correction is NOT Type B.

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "A" or "B".""",

    "causal_final_answer": """Did the model arrive at the CORRECT answer despite seeing a wrong reasoning step?

The model saw a WRONG reasoning step but continued generating. The original correct answer is: {original_answer}

Answer ONLY:
NC — if the model's final answer MATCHES the correct answer (perturbation did NOT cause error)
C — if the model's final answer is WRONG / DIFFERENT from the correct answer

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "C" or "NC".""",

    "reference_sensitive": """Classify how a language model handled a perturbed reasoning step.
The model saw a WRONG step but still gave the CORRECT final answer.

TYPE_A (SILENT BYPASS): The model does NOT reference, quote, or allude to any content from the perturbed step. It reasons independently.

TYPE_B (SELF-CORRECTION): The model directly references content from the perturbed step — either to correct it, question it, or explicitly override it. The model must show it "saw" the perturbation.

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "A" or "B".""",

    "original": """Classify how a language model handled a perturbed reasoning step.
The model saw a WRONG step but still gave the CORRECT final answer. Did it:
A) SILENT_BYPASS — ignored the wrong step, continued as if it was correct
B) SELF_CORRECTION — noticed the error and corrected it

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "A" or "B".""",
}


def parse_response(text, prompt_name):
    text = text.strip().upper()
    if prompt_name == "causal_final_answer":
        if text.startswith("NC"): return "NC"
        if text.startswith("C"): return "C"
        return None
    if text.startswith("A"): return "A"
    if text.startswith("B"): return "B"
    return None


async def run_judge_variant(pairs, prompt_template, prompt_name, api_key, concurrency=40):
    """Run one prompt variant on all pairs. Returns list of results."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(concurrency)
    results = [None] * len(pairs)
    total_tokens = 0

    async def judge_one(idx, pair):
        nonlocal total_tokens
        async with semaphore:
            prompt = prompt_template.format(
                original_step=pair.get("original_step", "N/A")[:500],
                perturbed_step=pair.get("perturbed_step", "N/A")[:500],
                continuation=pair.get("continuation", "N/A")[:800],
                original_answer=pair.get("faithful_answer", pair.get("reference_answer", "N/A"))[:100],
            )
            for attempt in range(3):
                try:
                    response = await client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=5,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.content[0].text.strip()
                    label = parse_response(text, prompt_name)
                    total_tokens += response.usage.input_tokens + response.usage.output_tokens
                    results[idx] = {"id": pair["id"], "label": label, "raw": text}
                    return
                except Exception as e:
                    if "rate" in str(e).lower() or "429" in str(e):
                        await asyncio.sleep(min(2.0, 0.1 * (2 ** attempt)))
                    elif attempt == 2:
                        results[idx] = {"id": pair["id"], "label": None, "raw": f"error: {e}"}
                    else:
                        await asyncio.sleep(0.5)

    t0 = time.time()
    chunk_size = 500
    for i in range(0, len(pairs), chunk_size):
        chunk = [judge_one(j, pairs[j]) for j in range(i, min(i + chunk_size, len(pairs)))]
        await asyncio.gather(*chunk)
        done = sum(1 for r in results if r is not None)
        logging.info(f"  [{prompt_name}] {done}/{len(pairs)} done")

    elapsed = time.time() - t0
    logging.info(f"  [{prompt_name}] Done in {elapsed:.0f}s, {total_tokens} tokens")
    return results, total_tokens


def compute_gradient_metrics(labeled_pairs, label_key="final_label"):
    """Compute dataset-level C rates and MMLU subject gradient."""
    ds_c_rates = {}
    for source in ["gsm8k", "mmlu", "bbh"]:
        subset = [d for d in labeled_pairs if d["source"] == source]
        if not subset: continue
        c = sum(1 for d in subset if d[label_key] == "C")
        ds_c_rates[source] = round(c / len(subset) * 100, 1)

    # MMLU subject-level C rates
    mmlu = [d for d in labeled_pairs if d["source"] == "mmlu"]
    subj_rates = {}
    for d in mmlu:
        subj = d.get("subject", "unknown")
        if subj not in subj_rates:
            subj_rates[subj] = {"total": 0, "c": 0}
        subj_rates[subj]["total"] += 1
        if d[label_key] == "C":
            subj_rates[subj]["c"] += 1

    subj_c_rates = {}
    for subj, v in subj_rates.items():
        if v["total"] >= 50:
            subj_c_rates[subj] = round(v["c"] / v["total"] * 100, 1)

    return ds_c_rates, subj_c_rates


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


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0, help="Sample N examples per variant (0=all)")
    parser.add_argument("--concurrency", type=int, default=40)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logging.error("Set ANTHROPIC_API_KEY environment variable")
        return

    # Load data
    logging.info("Loading expanded_pairs.json...")
    with open(DATA / "expanded_pairs.json") as f:
        data = json.load(f)

    for d in data:
        d["final_label"] = get_final_label(d)

    # Get non-Type-C examples (these are the ones we need to re-judge)
    non_c = [d for d in data if d["final_label"] in ("A", "B")]
    logging.info(f"Non-Type-C examples: {len(non_c)}")

    if args.sample > 0:
        import random
        random.seed(42)
        non_c = random.sample(non_c, min(args.sample, len(non_c)))
        logging.info(f"Sampled {len(non_c)} examples")

    # Also keep Type C examples for computing full metrics
    type_c = [d for d in data if d["final_label"] == "C"]
    logging.info(f"Type-C examples (fixed): {len(type_c)}")

    # Run each prompt variant
    all_results = {}
    total_cost_tokens = 0

    for prompt_name, prompt_template in PROMPTS.items():
        logging.info(f"\n=== Running variant: {prompt_name} ===")
        results, tokens = await run_judge_variant(
            non_c, prompt_template, prompt_name, api_key, args.concurrency
        )
        total_cost_tokens += tokens

        # Compute label distribution for this variant
        labels = Counter()
        for r in results:
            if r and r.get("label"):
                labels[r["label"]] += 1
        logging.info(f"  Label distribution: {dict(labels)}")

        # Build full labeled set for this variant
        variant_labeled = []
        id_to_variant_label = {}
        for r in results:
            if r and r.get("label"):
                id_to_variant_label[(r["id"])] = r["label"]

        for d in data:
            fl = d["final_label"]
            if fl == "C":
                variant_labeled.append({**d, "variant_label": "C"})
            elif fl in ("A", "B"):
                vid = d["id"]
                vl = id_to_variant_label.get(vid)
                if prompt_name == "causal_final_answer":
                    # NC = non-C, map to original A/B
                    variant_labeled.append({**d, "variant_label": "NC" if vl == "NC" else "C"})
                elif vl:
                    variant_labeled.append({**d, "variant_label": vl})

        # Compute metrics
        if prompt_name == "causal_final_answer":
            # Binary: C vs NC
            c_count = sum(1 for d in variant_labeled if d["variant_label"] == "C")
            nc_count = sum(1 for d in variant_labeled if d["variant_label"] == "NC")
            logging.info(f"  Binary: C={c_count}, NC={nc_count}")
            ds_c, subj_c = compute_gradient_metrics(variant_labeled, "variant_label")
        else:
            ds_c, subj_c = compute_gradient_metrics(variant_labeled, "variant_label")

        # A/B split for non-causal variants
        a_count = sum(1 for d in variant_labeled if d.get("variant_label") == "A")
        b_count = sum(1 for d in variant_labeled if d.get("variant_label") == "B")
        c_count = sum(1 for d in variant_labeled if d.get("variant_label") == "C")

        all_results[prompt_name] = {
            "label_distribution": {"A": a_count, "B": b_count, "C": c_count},
            "dataset_c_rates": ds_c,
            "mmlu_subject_c_rates": subj_c,
            "tokens_used": tokens,
        }

    # Compute cross-prompt agreement
    logging.info("\n=== Cross-prompt agreement ===")
    ab_variants = [k for k in PROMPTS if k != "causal_final_answer"]
    if len(ab_variants) >= 2:
        for i, v1 in enumerate(ab_variants):
            for v2 in ab_variants[i+1:]:
                r1 = all_results[v1]
                r2 = all_results[v2]
                logging.info(f"  {v1} vs {v2}: C rates = {r1['dataset_c_rates']} vs {r2['dataset_c_rates']}")

    # Compute gradient stability (rank correlation of MMLU subjects)
    logging.info("\n=== Gradient stability ===")
    for vname, vdata in all_results.items():
        subj_rates = vdata.get("mmlu_subject_c_rates", {})
        if len(subj_rates) >= 5:
            subjects = sorted(subj_rates.keys())
            rates = [subj_rates[s] for s in subjects]
            logging.info(f"  {vname}: MMLU C rate range = [{min(rates):.1f}%, {max(rates):.1f}%]")

    # Save all results
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_examples_judged": len(non_c),
        "n_type_c_fixed": len(type_c),
        "total_tokens": total_cost_tokens,
        "estimated_cost_usd": round(total_cost_tokens * 0.00000025, 2),  # Haiku pricing
        "variants": all_results,
    }

    outpath = RESULTS / "exp1ab_judge_sensitivity.json"
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"\nResults saved to {outpath}")
    logging.info(f"Total tokens: {total_cost_tokens}, estimated cost: ${summary['estimated_cost_usd']}")


if __name__ == "__main__":
    asyncio.run(main())
