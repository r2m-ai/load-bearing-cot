"""
Script 11: BBH (BIG-Bench Hard) dataset download + CoT generation.

Downloads BBH from HuggingFace, formats for our pipeline, and generates CoT.
BBH tasks are text-based, so we reuse MMLU perturbation strategies.

Output: data/raw/bbh_full.json, data/processed/faithful_cot.json (appended)
"""

import json
import re
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"11_bbh_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "google/gemma-2-9b-it"


def download_bbh(max_examples: int | None = None) -> list[dict]:
    """Download BBH dataset from HuggingFace."""
    from datasets import load_dataset

    logging.info("Downloading BBH...")
    results = []

    # BBH has 23 tasks
    tasks = [
        "boolean_expressions", "causal_judgement", "date_understanding",
        "disambiguation_qa", "dyck_languages", "formal_fallacies",
        "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
        "logical_deduction_seven_objects", "logical_deduction_three_objects",
        "movie_recommendation", "multistep_arithmetic_two",
        "navigate", "object_counting", "penguins_in_a_table",
        "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection",
        "snarks", "sports_understanding", "temporal_sequences",
        "tracking_shuffled_objects_three_objects",
    ]

    for task_name in tasks:
        try:
            ds = load_dataset("lukaemon/bbh", task_name, split="test")
            for example in ds:
                # BBH format: input (question) + target (answer)
                question = example.get("input", "")
                target = str(example.get("target", ""))

                results.append({
                    "id": f"bbh_{len(results)}",
                    "source": "bbh",
                    "subject": task_name,
                    "question": question,
                    "reference_answer": target,
                    "task": task_name,
                })
        except Exception as e:
            logging.warning(f"Failed to load BBH task {task_name}: {e}")
            continue

    logging.info(f"BBH total: {len(results)} examples across {len(tasks)} tasks")

    if max_examples and len(results) > max_examples:
        import random
        random.seed(42)
        results = random.sample(results, max_examples)
        logging.info(f"Sampled {len(results)} examples")

    return results


def format_bbh_prompt(question: str) -> str:
    """Format a BBH question as a CoT prompt."""
    return (
        "Answer the following question step by step. "
        "Show your reasoning clearly, then give the final answer "
        "on its own line as: Answer: <your answer>\n\n"
        f"Question: {question}\n\n"
        "Solution:"
    )


def extract_bbh_answer(response: str) -> str | None:
    """Extract answer from BBH CoT response."""
    match = re.search(r"Answer:\s*(.+?)(?:\n|$)", response)
    if match:
        return match.group(1).strip()
    # Fallback: last line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return None


def check_bbh_correct(extracted: str | None, reference: str) -> bool:
    """Check if extracted answer matches reference."""
    if extracted is None:
        return False
    # Normalize both
    ext = extracted.lower().strip().strip("()").strip()
    ref = reference.lower().strip().strip("()").strip()
    return ext == ref or ext.startswith(ref) or ref.startswith(ext)


def main():
    parser = argparse.ArgumentParser(description="BBH pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["download", "cot", "all"])
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    bbh_path = RAW_DIR / "bbh_full.json"

    # Step 1: Download
    if args.step in ("download", "all"):
        if not bbh_path.exists():
            bbh_data = download_bbh(args.max_examples)
            with open(bbh_path, "w") as f:
                json.dump(bbh_data, f, indent=2)
            logging.info(f"Saved {len(bbh_data)} BBH examples to {bbh_path}")
        else:
            logging.info(f"BBH already downloaded: {bbh_path}")

    # Step 2: Generate CoT
    if args.step in ("cot", "all"):
        import torch

        with open(bbh_path) as f:
            questions = json.load(f)

        if args.max_examples:
            questions = questions[:args.max_examples]

        logging.info(f"Generating CoT for {len(questions)} BBH questions...")

        # Load model
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16,
                device_map=args.device, attn_implementation="flash_attention_2",
            )
        except (ImportError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16,
                device_map=args.device, attn_implementation="sdpa",
            )
        model.eval()

        # Sort by length for efficient batching
        sorted_indices = sorted(range(len(questions)), key=lambda i: len(questions[i]["question"]))

        results = [None] * len(questions)
        correct_count = 0

        from tqdm import tqdm
        for batch_start in tqdm(range(0, len(sorted_indices), args.batch_size), desc="BBH"):
            batch_indices = sorted_indices[batch_start:batch_start + args.batch_size]
            batch = [questions[i] for i in batch_indices]

            prompts = [format_bbh_prompt(q["question"]) for q in batch]

            # Apply chat template
            formatted = []
            for p in prompts:
                messages = [{"role": "user", "content": p}]
                formatted.append(tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                ))

            inputs = tokenizer(
                formatted, return_tensors="pt", padding=True,
                truncation=True, max_length=2048,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=512,
                    do_sample=False, use_cache=True,
                )

            input_len = inputs["input_ids"].shape[1]
            for i, (orig_idx, q) in enumerate(zip(batch_indices, batch)):
                response = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True).strip()
                extracted = extract_bbh_answer(response)
                is_correct = check_bbh_correct(extracted, q["reference_answer"])

                if is_correct:
                    correct_count += 1

                results[orig_idx] = {
                    "id": q["id"],
                    "source": "bbh",
                    "subject": q.get("subject", q.get("task", "")),
                    "question": q["question"],
                    "reference_answer": q["reference_answer"],
                    "prompt": prompts[batch_indices.index(orig_idx) if orig_idx in batch_indices else 0],
                    "cot_response": response,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                }

        results = [r for r in results if r is not None]

        if len(results) > 0:
            logging.info(f"BBH accuracy: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
        else:
            logging.warning("No BBH results generated")

        # Save BBH results
        bbh_cot_path = PROCESSED_DIR / "bbh_cot_responses.json"
        with open(bbh_cot_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save faithful (correct) examples
        faithful = [r for r in results if r["is_correct"]]
        bbh_faithful_path = PROCESSED_DIR / "bbh_faithful_cot.json"
        with open(bbh_faithful_path, "w") as f:
            json.dump(faithful, f, indent=2)

        logging.info(f"Saved {len(faithful)} faithful BBH examples to {bbh_faithful_path}")

        # Also append to main faithful_cot.json
        main_faithful_path = PROCESSED_DIR / "faithful_cot.json"
        if main_faithful_path.exists():
            with open(main_faithful_path) as f:
                existing = json.load(f)
            # Remove any old BBH entries
            existing = [e for e in existing if e.get("source") != "bbh"]
            existing.extend(faithful)
            with open(main_faithful_path, "w") as f:
                json.dump(existing, f, indent=2)
            logging.info(f"Updated faithful_cot.json: {len(existing)} total")

        print(f"\nBBH Summary: {correct_count}/{len(results)} correct ({correct_count/len(results)*100:.1f}%)")
        print(f"Faithful examples: {len(faithful)}")


if __name__ == "__main__":
    main()
