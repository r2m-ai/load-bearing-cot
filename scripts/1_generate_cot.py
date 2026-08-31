"""
Script 1: Generate Chain-of-Thought responses using Gemma-2-9B-IT.

Uses vLLM for high-throughput inference:
- Continuous batching (no padding waste)
- PagedAttention (efficient KV cache)
- C++ token sampling (minimal Python overhead)
- Tensor parallelism ready

Falls back to HuggingFace if vLLM is not available.

Output: data/processed/faithful_cot.json
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
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"1_generate_cot_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "google/gemma-2-9b-it"


# ---- Prompt formatting ----

def format_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Show your reasoning clearly, then give the final numeric answer "
        "on its own line as: Answer: <number>\n\n"
        f"Question: {question}\n\n"
        "Solution:"
    )


def format_mmlu_prompt(question: str, choices: list[str]) -> str:
    choice_str = "\n".join(
        f"  {letter}) {text}"
        for letter, text in zip(["A", "B", "C", "D"], choices)
    )
    return (
        "Answer the following multiple choice question step by step. "
        "Show your reasoning clearly, then give the final answer "
        "on its own line as: Answer: <letter>\n\n"
        f"Question: {question}\n{choice_str}\n\n"
        "Solution:"
    )


def apply_chat_template(tokenizer, prompt: str) -> str:
    """Wrap prompt in chat template."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# ---- Answer extraction ----

def extract_gsm8k_answer(response: str) -> str | None:
    match = re.search(r"Answer:\s*\$?([\d,]+\.?\d*)", response)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"[\d,]+\.?\d*", response)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def extract_mmlu_answer(response: str) -> str | None:
    match = re.search(r"Answer:\s*([A-Da-d])", response)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-Da-d])\)?\.?\s*$", response)
    if match:
        return match.group(1).upper()
    return None


def check_gsm8k_correct(extracted: str | None, reference: str) -> bool:
    if extracted is None:
        return False
    try:
        return float(extracted) == float(reference.replace(",", ""))
    except ValueError:
        return False


def check_mmlu_correct(extracted: str | None, correct_letter: str) -> bool:
    if extracted is None:
        return False
    return extracted.upper() == correct_letter.upper()


# ---- vLLM inference ----

def generate_with_vllm(model, tokenizer, prompts: list[str], max_tokens: int = 512) -> list[str]:
    """Generate responses using vLLM's offline batched inference."""
    from vllm import SamplingParams

    # Apply chat template to all prompts
    formatted = [apply_chat_template(tokenizer, p) for p in prompts]

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,  # Greedy
    )

    # vLLM handles batching, scheduling, and PagedAttention internally
    outputs = model.generate(formatted, sampling_params)

    # Sort by request_id to maintain input order
    outputs = sorted(outputs, key=lambda x: x.request_id)

    responses = []
    for output in outputs:
        text = output.outputs[0].text.strip()
        responses.append(text)

    return responses


# ---- HuggingFace fallback ----

def generate_with_hf(model, tokenizer, prompts: list[str], max_tokens: int = 512) -> list[str]:
    """Fallback: batched generation with HuggingFace. Sorts by length to reduce padding."""
    import torch

    # Sort by prompt length to minimize padding waste within each batch
    indexed = sorted(enumerate(prompts), key=lambda x: len(x[1]))
    sorted_prompts = [p for _, p in indexed]

    formatted = [apply_chat_template(tokenizer, p) for p in sorted_prompts]
    inputs = tokenizer(
        formatted, return_tensors="pt", padding=True,
        truncation=True, max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens,
            do_sample=False, use_cache=True,
        )

    # Unsort back to original order
    sorted_responses = []
    input_len = inputs["input_ids"].shape[1]
    for output in outputs:
        text = tokenizer.decode(output[input_len:], skip_special_tokens=True)
        sorted_responses.append(text.strip())

    responses = [""] * len(prompts)
    for (orig_idx, _), resp in zip(indexed, sorted_responses):
        responses[orig_idx] = resp

    return responses


# ---- Main processing ----

def process_dataset(
    model, tokenizer, questions: list[dict], source: str,
    batch_size: int, use_vllm: bool,
):
    """Process a dataset with checkpointing."""
    checkpoint_path = PROCESSED_DIR / f"checkpoint_{source}.json"
    results = []
    start_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            results = json.load(f)
        start_idx = len(results)
        logging.info(f"Resuming {source} from checkpoint: {start_idx}/{len(questions)}")

    remaining = questions[start_idx:]
    correct_count = sum(1 for r in results if r["is_correct"])

    if use_vllm:
        # vLLM: submit ALL prompts at once — it handles batching internally
        logging.info(f"Processing {len(remaining)} {source} questions with vLLM (all at once)...")
        if source == "gsm8k":
            prompts = [format_gsm8k_prompt(q["question"]) for q in remaining]
        else:
            prompts = [format_mmlu_prompt(q["question"], q["choices"]) for q in remaining]

        t0 = time.time()
        responses = generate_with_vllm(model, tokenizer, prompts)
        elapsed = time.time() - t0
        avg_time = elapsed / len(remaining) if remaining else 0

        for q, prompt, response in zip(remaining, prompts, responses):
            if source == "gsm8k":
                extracted = extract_gsm8k_answer(response)
                is_correct = check_gsm8k_correct(extracted, q["reference_answer"])
            else:
                extracted = extract_mmlu_answer(response)
                is_correct = check_mmlu_correct(extracted, q["correct_letter"])

            if is_correct:
                correct_count += 1

            result = {
                "id": q["id"],
                "source": source,
                "question": q["question"],
                "reference_answer": q.get("reference_answer", ""),
                "prompt": prompt,
                "cot_response": response,
                "extracted_answer": extracted,
                "is_correct": is_correct,
                "inference_time_s": round(avg_time, 2),
            }
            if source == "gsm8k":
                result["n_steps"] = q["n_steps"]
            else:
                result["subject"] = q["subject"]
                result["choices"] = q["choices"]
                result["correct_letter"] = q["correct_letter"]
            results.append(result)

        throughput = len(remaining) / elapsed if elapsed > 0 else 0
        logging.info(
            f"{source} done: {len(remaining)} examples in {elapsed:.0f}s "
            f"({throughput:.1f} ex/s), acc={correct_count}/{len(results)} "
            f"({correct_count/len(results)*100:.1f}%)"
        )

    else:
        # HuggingFace: process in batches with global length sorting
        from tqdm import tqdm
        logging.info(f"Processing {len(remaining)} {source} questions with HF (batch={batch_size})...")

        # Global sort by question length to minimize padding waste across batches
        sorted_indices = sorted(range(len(remaining)), key=lambda i: len(remaining[i]["question"]))
        sorted_remaining = [remaining[i] for i in sorted_indices]

        pbar = tqdm(total=len(questions), initial=start_idx, desc=source.upper())

        for batch_start in range(0, len(sorted_remaining), batch_size):
            batch = sorted_remaining[batch_start:batch_start + batch_size]

            if source == "gsm8k":
                prompts = [format_gsm8k_prompt(q["question"]) for q in batch]
            else:
                prompts = [format_mmlu_prompt(q["question"], q["choices"]) for q in batch]

            t0 = time.time()
            responses = generate_with_hf(model, tokenizer, prompts)
            batch_time = time.time() - t0

            for i, (q, response) in enumerate(zip(batch, responses)):
                if source == "gsm8k":
                    extracted = extract_gsm8k_answer(response)
                    is_correct = check_gsm8k_correct(extracted, q["reference_answer"])
                else:
                    extracted = extract_mmlu_answer(response)
                    is_correct = check_mmlu_correct(extracted, q["correct_letter"])

                if is_correct:
                    correct_count += 1

                result = {
                    "id": q["id"],
                    "source": source,
                    "question": q["question"],
                    "reference_answer": q.get("reference_answer", ""),
                    "prompt": prompts[i],
                    "cot_response": response,
                    "extracted_answer": extracted,
                    "is_correct": is_correct,
                    "inference_time_s": round(batch_time / len(batch), 2),
                }
                if source == "gsm8k":
                    result["n_steps"] = q["n_steps"]
                else:
                    result["subject"] = q["subject"]
                    result["choices"] = q["choices"]
                    result["correct_letter"] = q["correct_letter"]
                results.append(result)

            pbar.update(len(batch))

            total_done = start_idx + batch_start + len(batch)
            if total_done % 100 < batch_size:
                with open(checkpoint_path, "w") as f:
                    json.dump(results, f)
                throughput = len(batch) / batch_time
                logging.info(
                    f"{source} checkpoint: {total_done}/{len(questions)}, "
                    f"acc={correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%), "
                    f"{throughput:.1f} ex/s"
                )

        pbar.close()

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    logging.info(f"{source} accuracy: {correct_count}/{len(questions)} ({correct_count/len(questions)*100:.1f}%)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate CoT responses")
    parser.add_argument("--max-gsm8k", type=int, default=None)
    parser.add_argument("--max-mmlu", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for HF fallback (ignored with vLLM)")
    parser.add_argument("--gsm8k-only", action="store_true")
    parser.add_argument("--mmlu-only", action="store_true")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Force HuggingFace backend instead of vLLM")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90,
                        help="vLLM GPU memory fraction (default: 0.90)")
    args = parser.parse_args()

    # Try vLLM first
    use_vllm = False
    model = None
    tokenizer = None

    if not args.no_vllm:
        try:
            from vllm import LLM
            from transformers import AutoTokenizer

            logging.info(f"Loading {MODEL_ID} with vLLM...")
            model = LLM(
                model=MODEL_ID,
                dtype="bfloat16",
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=2560,  # prompt (~512) + generation (~512) with headroom
                enforce_eager=False,  # Allow CUDA graphs
            )
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            use_vllm = True
            logging.info("vLLM loaded successfully")
        except ImportError:
            logging.info("vLLM not installed, falling back to HuggingFace")
        except Exception as e:
            logging.warning(f"vLLM failed to load: {e}, falling back to HuggingFace")

    if not use_vllm:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        logging.info(f"Loading {MODEL_ID} with HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16,
                device_map="cuda", attn_implementation="flash_attention_2",
            )
        except (ImportError, ValueError):
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16,
                device_map="cuda", attn_implementation="sdpa",
            )
        model.eval()

    all_results = []

    if not args.mmlu_only:
        with open(RAW_DIR / "gsm8k_full.json") as f:
            gsm8k_questions = json.load(f)
        if args.max_gsm8k:
            gsm8k_questions = gsm8k_questions[:args.max_gsm8k]
        gsm8k_results = process_dataset(
            model, tokenizer, gsm8k_questions, "gsm8k",
            batch_size=args.batch_size, use_vllm=use_vllm,
        )
        all_results.extend(gsm8k_results)

    if not args.gsm8k_only:
        with open(RAW_DIR / "mmlu_full.json") as f:
            mmlu_questions = json.load(f)
        if args.max_mmlu:
            mmlu_questions = mmlu_questions[:args.max_mmlu]
        mmlu_results = process_dataset(
            model, tokenizer, mmlu_questions, "mmlu",
            batch_size=args.batch_size, use_vllm=use_vllm,
        )
        all_results.extend(mmlu_results)

    # Save results — MERGE with existing file (don't overwrite other datasets)
    output_path = PROCESSED_DIR / "cot_responses.json"
    existing = []
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        # Remove old entries for the sources we just processed
        sources_processed = set(r["source"] for r in all_results)
        existing = [e for e in existing if e["source"] not in sources_processed]
    merged = existing + all_results
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)
    logging.info(f"Saved {len(all_results)} new + {len(existing)} existing = {len(merged)} total to {output_path}")

    # Also save per-source backup
    for src in set(r["source"] for r in all_results):
        src_results = [r for r in all_results if r["source"] == src]
        src_path = PROCESSED_DIR / f"{src}_cot_responses.json"
        with open(src_path, "w") as f:
            json.dump(src_results, f, indent=2)
        logging.info(f"Backup: {len(src_results)} {src} responses to {src_path}")

    faithful = [r for r in merged if r["is_correct"]]
    faithful_path = PROCESSED_DIR / "faithful_cot.json"
    with open(faithful_path, "w") as f:
        json.dump(faithful, f, indent=2)
    logging.info(f"Saved {len(faithful)} faithful examples to {faithful_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY (backend: {'vLLM' if use_vllm else 'HuggingFace'})")
    print(f"{'='*50}")
    total_time = sum(r["inference_time_s"] for r in all_results)
    for src in ["gsm8k", "mmlu"]:
        total = sum(1 for r in all_results if r["source"] == src)
        correct = sum(1 for r in all_results if r["source"] == src and r["is_correct"])
        if total > 0:
            print(f"{src.upper()}: {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"Total faithful examples: {len(faithful)}")
    print(f"Total time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
