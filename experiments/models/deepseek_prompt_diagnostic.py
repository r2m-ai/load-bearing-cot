"""
exp_3_1b_deepseek_prompts.py — diagnose the 38.5% GSM8K accuracy on
DeepSeek-R1-Distill-Qwen-7B (item E).

The paper's main exp_3_1 result on DeepSeek was 38.5% GSM8K accuracy.
Public benchmarks for the same model claim ~85%. This script tests 3
prompt formats on 100 GSM8K questions to identify the format gap:

  1. **current** : our exp_3_1 prompt ("Solve... Show your reasoning...
                  Answer: <number>\\n\\nQuestion: X\\n\\nSolution:")
  2. **plain**   : just the question (let R1's chat template + system
                  setup do the steering)
  3. **r1_official** : prompt format from DeepSeek's R1 GitHub repo,
                  which explicitly asks for `<think>...</think>` and
                  `\\boxed{}` answer formatting.

For each format we record:
  - accuracy on the 100-question GSM8K sample
  - mean think-block length (tokens in pre-`</think>`)
  - mean output length

If "r1_official" recovers accuracy to ~70-85%, the exp_3_1 numbers
underestimated DeepSeek's capability and we should rerun the full
pipeline with that prompt. If all three are ~38%, the GSM8K result is
genuinely DeepSeek's behavior on math-with-temperature-0.

Usage:
  python experiments/models/deepseek_prompt_diagnostic.py
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "experiments" / "pipeline" / "perturbation_helpers.py").exists():
            return p
    raise RuntimeError("Could not locate repo root from " + str(start))


ROOT = _find_repo_root(Path(__file__).resolve())
PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"exp_3_1b_prompts_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
N_SAMPLE = 100
MAX_NEW_TOKENS = 4096   # generous for `<think>` traces
OUT_REPORT = PROCESSED / "deepseek_prompt_test.json"


# ---- Three prompt formats ----

def prompt_current(q: str) -> str:
    """Our exp_3_1 prompt (the one that gave 38.5% acc)."""
    return (
        "Solve the following math problem step by step. "
        "Show your reasoning clearly, then give the final numeric answer "
        "on its own line as: Answer: <number>\n\n"
        f"Question: {q}\n\n"
        "Solution:"
    )


def prompt_plain(q: str) -> str:
    """Bare question; let R1's chat template + add_generation_prompt drive
    the `<think>...</think>` format."""
    return q


def prompt_r1_official(q: str) -> str:
    """Prompt structure from DeepSeek's R1 repo. Asks for `<think>` block
    + `\\boxed{}` answer (R1's training format)."""
    return (
        f"{q}\n\n"
        "Please reason step by step inside <think>...</think> tags, "
        "then provide the final answer within \\boxed{}."
    )


PROMPTS = {
    "current": prompt_current,
    "plain": prompt_plain,
    "r1_official": prompt_r1_official,
}


def extract_answer(response: str) -> str | None:
    """Try Answer: regex first, then \\boxed{} fallback, then last
    number heuristic. Operates on full response (vLLM may strip
    `<think>` tags); searches for the LAST match preferentially."""
    matches = list(re.finditer(r"\\boxed\{\s*([\-\+]?[\d,]+\.?\d*)\s*\}", response))
    if matches:
        return matches[-1].group(1).replace(",", "")
    matches = list(re.finditer(r"Answer:\s*\$?([\-\+]?[\d,]+\.?\d*)", response))
    if matches:
        return matches[-1].group(1).replace(",", "")
    nums = re.findall(r"[\d,]+\.?\d*", response)
    if nums:
        return nums[-1].replace(",", "")
    return None


def correct(extracted, ref) -> bool:
    if extracted is None:
        return False
    try:
        return float(extracted) == float(str(ref).replace(",", ""))
    except (ValueError, TypeError):
        return False


def load_model_vllm():
    from transformers import AutoTokenizer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens)
    os.environ.setdefault("VLLM_USE_V1", "0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    from vllm import LLM
    logging.info(f"Loading {MODEL_ID} with vLLM V0...")
    llm = LLM(model=MODEL_ID, dtype="bfloat16",
              gpu_memory_utilization=0.90, max_model_len=4096 + 1024,
              enforce_eager=False)
    return llm, tokenizer


def apply_chat(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=N_SAMPLE)
    args = ap.parse_args()

    gsm = json.load(open(RAW / "gsm8k_full.json"))[:args.n_sample]
    logging.info(f"Testing {len(gsm)} GSM8K questions × {len(PROMPTS)} prompt formats")

    llm, tokenizer = load_model_vllm()
    from vllm import SamplingParams
    sp = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0)

    report = {}
    for name, builder in PROMPTS.items():
        rendered = [apply_chat(tokenizer, builder(q["question"])) for q in gsm]
        t0 = time.time()
        outs = llm.generate(rendered, sp)
        outs = sorted(outs, key=lambda o: int(o.request_id))
        elapsed = time.time() - t0

        results = []
        n_correct = 0
        n_think = 0
        total_tokens = 0
        for q, o in zip(gsm, outs):
            text = o.outputs[0].text
            ext = extract_answer(text)
            ok = correct(ext, q["reference_answer"])
            has_close = "</think>" in text
            tokens = len(text) // 4  # crude
            if has_close:
                n_think += 1
            total_tokens += tokens
            if ok:
                n_correct += 1
            results.append({
                "id": q["id"],
                "extracted": ext,
                "ref": q["reference_answer"],
                "correct": ok,
                "has_close_think": has_close,
                "response_len_chars": len(text),
                "response_preview": text[:300],
            })

        acc = n_correct / len(gsm) * 100
        think_rate = n_think / len(gsm) * 100
        avg_tokens = total_tokens / len(gsm)
        report[name] = {
            "n": len(gsm),
            "accuracy_pct": acc,
            "think_rate_pct": think_rate,
            "avg_tokens_per_response": avg_tokens,
            "wall_time_s": elapsed,
            "results": results[:3],   # only first 3 samples for log brevity
        }
        logging.info(f"  {name:14s}  acc={acc:5.1f}%  think_rate={think_rate:5.1f}%  "
                     f"~{avg_tokens:.0f} tokens/resp  {elapsed:.0f}s")

    json.dump(report, open(OUT_REPORT, "w"), indent=2)
    print("\n" + "=" * 70)
    print("DEEPSEEK PROMPT-FORMAT TEST (item E)")
    print("=" * 70)
    print(f"{'format':14s}  {'acc':>6s}  {'think_rate':>10s}  {'avg_tokens':>10s}")
    for name, r in report.items():
        print(f"{name:14s}  {r['accuracy_pct']:5.1f}%  "
              f"{r['think_rate_pct']:9.1f}%  {r['avg_tokens_per_response']:9.0f}")
    print("-" * 70)
    print("Reference: public benchmarks claim ~85% GSM8K for this distill.")
    print("If r1_official recovers acc to >70%, recommend rerunning full")
    print("exp_3_1 with that prompt; otherwise our exp_3_1 numbers stand.")
    print("=" * 70)
    print(f"Saved: {OUT_REPORT}")


if __name__ == "__main__":
    main()
