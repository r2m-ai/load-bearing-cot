"""
Script 9a (v5): Validate LLM judge on known TYPE_A and TYPE_B examples.

Runs the judge prompt on examples already classified by the rule-based system
to measure agreement. If agreement > 90%, the judge prompt is good enough
for the full run.

Input: data/processed/subclassified_pairs.json
Output: logs/judge_validation.log, data/processed/judge_validation.json
"""

import json
import argparse
import logging
import time
import random
from datetime import datetime
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"9a_validate_judge_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

JUDGE_PROMPT = """You are classifying how a language model handled a perturbed reasoning step in a math or multiple-choice problem.

Context:
- The model was solving a problem step by step
- One reasoning step was secretly perturbed (a number changed, operation swapped, or misleading claim inserted)
- The model continued generating from the perturbed step
- The model ultimately gave the CORRECT final answer despite the perturbation

Your task: determine HOW the model handled the perturbation.

ORIGINAL STEP (correct):
{original_step}

PERTURBED STEP (what the model saw):
{perturbed_step}

MODEL'S CONTINUATION (what it generated after the perturbed step):
{continuation}

Classify as one of:
A) SILENT_BYPASS — The model ignored the perturbation entirely. It continued as if the original (correct) step was there. It did not acknowledge, question, or react to the wrong value/operation. The correct answer emerged without any sign the model noticed the error.

B) SELF_CORRECTION — The model noticed or reacted to the perturbation in some way. This includes: recalculating with different numbers than the perturbed step implies, questioning a value, noting an inconsistency, switching back to the original value, or any sign that the model processed the error and recovered from it.

Respond with EXACTLY this format:
LABEL: A or B
REASON: one sentence explaining your classification"""


def call_judge(prompt: str, api_key: str, model: str = "claude-sonnet-4-20250514") -> dict:
    """Call Claude API to judge a single example."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    response = client.messages.create(
        model=model,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text.strip()

    # Parse response
    label = None
    reason = ""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("LABEL:"):
            label_text = line.replace("LABEL:", "").strip()
            if "A" in label_text and "B" not in label_text:
                label = "A"
            elif "B" in label_text and "A" not in label_text:
                label = "B"
        elif line.startswith("REASON:"):
            reason = line.replace("REASON:", "").strip()

    return {
        "label": label,
        "reason": reason,
        "raw_response": text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


def format_judge_prompt(pair: dict) -> str:
    """Format the judge prompt for a single example."""
    return JUDGE_PROMPT.format(
        original_step=pair.get("original_step", "N/A"),
        perturbed_step=pair.get("perturbed_step", "N/A"),
        continuation=pair.get("continuation", "N/A")[:1000],  # Truncate long continuations
    )


def main():
    parser = argparse.ArgumentParser(description="Validate LLM judge")
    parser.add_argument("--api-key", type=str, required=True, help="Anthropic API key")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of samples per class (default: 50)")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    # Load subclassified pairs
    with open(PROCESSED_DIR / "subclassified_pairs.json") as f:
        pairs = json.load(f)

    # Get known TYPE_A and TYPE_B examples
    type_a = [p for p in pairs if p.get("subtype") == "TYPE_A"]
    type_b = [p for p in pairs if p.get("subtype") == "TYPE_B"]

    logging.info(f"Available: TYPE_A={len(type_a)}, TYPE_B={len(type_b)}")

    # Sample
    n = min(args.n_samples, len(type_a), len(type_b))
    sample_a = random.sample(type_a, n)
    sample_b = random.sample(type_b, n)

    validation_set = [(p, "A") for p in sample_a] + [(p, "B") for p in sample_b]
    random.shuffle(validation_set)

    logging.info(f"Validating on {len(validation_set)} examples ({n} per class)")

    # Run judge
    results = []
    correct = 0
    total_tokens = 0

    for i, (pair, ground_truth) in enumerate(validation_set):
        prompt = format_judge_prompt(pair)
        try:
            result = call_judge(prompt, args.api_key, args.model)
            total_tokens += result["input_tokens"] + result["output_tokens"]

            is_correct = result["label"] == ground_truth
            if is_correct:
                correct += 1

            results.append({
                "id": pair["id"],
                "source": pair["source"],
                "ground_truth": ground_truth,
                "judge_label": result["label"],
                "judge_reason": result["reason"],
                "correct": is_correct,
                "subtype_reason": pair.get("subtype_reason", ""),
            })

            if (i + 1) % 10 == 0:
                logging.info(f"Progress: {i+1}/{len(validation_set)}, "
                             f"accuracy={correct}/{i+1} ({correct/(i+1)*100:.0f}%)")

            # Rate limiting
            time.sleep(0.5)

        except Exception as e:
            logging.error(f"Error on {pair['id']}: {e}")
            results.append({
                "id": pair["id"],
                "ground_truth": ground_truth,
                "judge_label": None,
                "error": str(e),
                "correct": False,
            })

    # Save results
    output_path = PROCESSED_DIR / "judge_validation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    total = len(results)
    judged = sum(1 for r in results if r.get("judge_label") is not None)
    correct = sum(1 for r in results if r.get("correct"))
    a_correct = sum(1 for r in results if r.get("ground_truth") == "A" and r.get("correct"))
    a_total = sum(1 for r in results if r.get("ground_truth") == "A" and r.get("judge_label") is not None)
    b_correct = sum(1 for r in results if r.get("ground_truth") == "B" and r.get("correct"))
    b_total = sum(1 for r in results if r.get("ground_truth") == "B" and r.get("judge_label") is not None)

    print(f"\n{'='*50}")
    print(f"JUDGE VALIDATION RESULTS")
    print(f"{'='*50}")
    print(f"Total: {total}, Judged: {judged}")
    print(f"Overall accuracy: {correct}/{judged} ({correct/judged*100:.1f}%)" if judged > 0 else "No results")
    print(f"  TYPE_A accuracy: {a_correct}/{a_total} ({a_correct/a_total*100:.1f}%)" if a_total > 0 else "")
    print(f"  TYPE_B accuracy: {b_correct}/{b_total} ({b_correct/b_total*100:.1f}%)" if b_total > 0 else "")
    print(f"Total tokens: {total_tokens}")

    if judged > 0 and correct / judged >= 0.9:
        print(f"\n✓ Agreement ≥ 90% — judge prompt is validated!")
        print(f"  Proceed with: python3 experiments/pipeline/judge_behaviors.py --api-key <key>")
    elif judged > 0 and correct / judged >= 0.8:
        print(f"\n~ Agreement 80-90% — judge is decent but could be improved")
    else:
        print(f"\n✗ Agreement < 80% — judge prompt needs revision")

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
