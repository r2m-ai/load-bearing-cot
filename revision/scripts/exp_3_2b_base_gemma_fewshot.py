"""
exp_3_2b_base_gemma_fewshot.py — base Gemma-2-9B with **few-shot CoT**
prompting (item D: complete the base-vs-IT story that exp_3_2 (zero-shot)
left as a viability-FAIL).

exp_3_2 ran base Gemma under zero-shot and got 0% multi-step CoT
compliance across all three sources — the viability gate exited
cleanly. Per the 3.5p caveat in the plan, that result is honest but
incomplete: base Gemma may produce structured CoT under few-shot
prompting, and if so, a real base-vs-IT comparison becomes possible
(now confounded with the prompt regime, but that confound is what 3.5p
asks us to surface).

This script:
  1. Re-renders prompts with 2-3 hand-crafted CoT exemplars per source
     (the exemplars exhibit the "Step 1: ... Step 2: ... Answer: X"
     structure the perturbation pipeline depends on).
  2. Runs the same viability go/no-go gate.
  3. If any source passes, runs the full step-1..5 pipeline on that
     source (subject to disk + compute budget).

Inherits all the vLLM-V0 + tokenizer-monkey-patch + cache-dir-fix
plumbing from exp_3_2. Output goes to a separate dir to avoid
clobbering the zero-shot run.

Usage:
  python revision/scripts/exp_3_2b_base_gemma_fewshot.py --stage full
  python revision/scripts/exp_3_2b_base_gemma_fewshot.py --stage viability
"""

# Import everything from exp_3_2 and override only the prompt builders.

import argparse
import importlib.util
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "scripts" / "create_perturbations_v8_helpers.py").exists():
            return p
    raise RuntimeError("Could not locate repo root from " + str(start))


ROOT = _find_repo_root(Path(__file__).resolve())
SCRIPTS_REV = ROOT / "revision" / "scripts"
SCRIPTS = ROOT / "scripts"
PROCESSED = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Few-shot variant — separate output dir so it does NOT clobber zero-shot.
MODEL_SLUG = "gemma2_9b_base_fewshot"
MODEL_DIR = PROCESSED / MODEL_SLUG
HIDDEN_DIR = MODEL_DIR / "hidden_states"
for d in [MODEL_DIR, HIDDEN_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# Import the original exp_3_2 module to inherit pipeline functions.
def _load(filename, modname):
    spec = importlib.util.spec_from_file_location(
        modname, SCRIPTS_REV / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_E32 = _load("exp_3_2_base_gemma_crossmodel.py", "exp_3_2")

# Reuse most of exp_3_2's functions verbatim.
_E32.MODEL_DIR = MODEL_DIR
_E32.HIDDEN_DIR = HIDDEN_DIR

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"exp_3_2b_fewshot_{timestamp}.log"),
        logging.StreamHandler(),
    ],
)


# ---- Few-shot exemplars ----

GSM8K_EXEMPLARS = """
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Step-by-step solution:
Step 1: Natalia sold 48 clips in April.
Step 2: In May, she sold half as many clips as in April, so she sold 48 / 2 = 24 clips.
Step 3: In total across April and May, she sold 48 + 24 = 72 clips.
Answer: 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Step-by-step solution:
Step 1: Weng's hourly rate is $12, so per minute she earns 12 / 60 = $0.20.
Step 2: She worked 50 minutes, so she earned 50 * 0.20 = $10.
Answer: 10
""".strip()


MMLU_EXEMPLARS = """
Question: What is the chemical symbol for gold?
  A) Au
  B) Ag
  C) Gd
  D) Go
Step-by-step solution:
Step 1: Gold's chemical symbol comes from the Latin name "aurum".
Step 2: The symbol derived from "aurum" is Au.
Step 3: Therefore, the chemical symbol for gold is A) Au.
Answer: A

Question: Which planet in our solar system has the most moons?
  A) Earth
  B) Mars
  C) Saturn
  D) Mercury
Step-by-step solution:
Step 1: Earth has 1 moon and Mars has 2 moons.
Step 2: Mercury has no moons.
Step 3: Saturn has the most moons among the listed planets, with over 80 confirmed.
Answer: C
""".strip()


BBH_EXEMPLARS = """
Question: Evaluate: ( ( ( ( 5 + 7 ) * 3 ) - 4 ) + 2 ) =
Step-by-step solution:
Step 1: Inner: 5 + 7 = 12.
Step 2: 12 * 3 = 36.
Step 3: 36 - 4 = 32.
Step 4: 32 + 2 = 34.
Answer: 34

Question: Is the following sentence plausible? "The boxer threw the punch with his glove on."
  Options:
  - yes
  - no
Step-by-step solution:
Step 1: Boxers wear gloves during fights.
Step 2: Throwing a punch with a glove on is the normal action for a boxer.
Step 3: The sentence is plausible.
Answer: yes
""".strip()


# ---- Override the prompt builders to inject few-shot exemplars ----

def format_gsm8k_prompt_fewshot(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "After the final step, write the numeric answer on its own line as "
        "Answer: <number>.\n\n"
        f"{GSM8K_EXEMPLARS}\n\n"
        f"Question: {question}\n"
        "Step-by-step solution:\n"
        "Step 1:"
    )


def format_mmlu_prompt_fewshot(question: str, choices: list[str]) -> str:
    choice_str = "\n".join(
        f"  {letter}) {text}"
        for letter, text in zip(["A", "B", "C", "D"], choices)
    )
    return (
        "Answer the following multiple choice question step by step. "
        "After the final step, write the final answer on its own line as "
        "Answer: <letter>.\n\n"
        f"{MMLU_EXEMPLARS}\n\n"
        f"Question: {question}\n{choice_str}\n"
        "Step-by-step solution:\n"
        "Step 1:"
    )


def format_bbh_prompt_fewshot(question: str) -> str:
    return (
        "Answer the following question step by step. "
        "After the final step, write the final answer on its own line as "
        "Answer: <your answer>.\n\n"
        f"{BBH_EXEMPLARS}\n\n"
        f"Question: {question}\n"
        "Step-by-step solution:\n"
        "Step 1:"
    )


def build_raw_prompt_fewshot(q: dict) -> str:
    src = q["source"]
    if src == "gsm8k":
        return format_gsm8k_prompt_fewshot(q["question"])
    if src == "mmlu":
        return format_mmlu_prompt_fewshot(q["question"], q["choices"])
    return format_bbh_prompt_fewshot(q["question"])


# Patch exp_3_2's prompt builders so its viability + step1 pipeline use
# the few-shot versions.
_E32.format_gsm8k_prompt_raw = format_gsm8k_prompt_fewshot
_E32.format_mmlu_prompt_raw = format_mmlu_prompt_fewshot
_E32.format_bbh_prompt_raw = format_bbh_prompt_fewshot
_E32.build_raw_prompt = build_raw_prompt_fewshot

# Also fix the continuation-input-builder: it strips `Step 1:` from the
# original prompt. With few-shot the trailing "Step 1:" is still at the
# end (we kept the same structure), so the same strip logic works.


def main():
    ap = argparse.ArgumentParser(
        description="Base Gemma + few-shot CoT (item D — complement to exp_3_2)")
    ap.add_argument("--stage", choices=["viability", "full"], default="full")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--skip-to", type=str, default=None,
                    choices=["step2", "step3", "step4", "step5"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--override-viability", action="store_true")
    args = ap.parse_args()

    # Call exp_3_2's main flow with patched functions. Easiest: replicate
    # the main() logic but parameterize on our few-shot helpers.
    # Since we've already monkey-patched build_raw_prompt etc., we can
    # just call _E32.main() — but it has its own argparse. Recreate flow.

    import torch
    _E32.configure_h100()

    logging.info("=" * 64)
    logging.info(f"BASE GEMMA + FEW-SHOT CoT — {_E32.MODEL_ID}")
    logging.info("=" * 64)
    logging.info(f"Stage: {args.stage}")
    logging.info(f"Output dir: {MODEL_DIR}")
    logging.info("NB: this is item D — few-shot complement to exp_3_2 zero-shot. "
                 "Compare base+fs vs IT+zs results to isolate instruction tuning.")

    if args.stage == "viability":
        _E32.stage_viability(args)
        return

    # Full pipeline (mirrors exp_3_2.main)
    model = None
    tokenizer = None
    is_vllm = False
    need_generation = args.skip_to not in ("step3", "step4", "step5")
    need_hf_for_step4 = args.skip_to not in ("step5",)

    if need_generation:
        model, tokenizer, is_vllm = _E32.load_model_for_generation(
            args.device, no_vllm=args.no_vllm)
    elif need_hf_for_step4:
        tokenizer, cache_kwargs = _E32._load_tokenizer()
        model, tokenizer, is_vllm = _E32.load_hf_model_for_generation(
            args.device, tokenizer, cache_kwargs)

    if args.skip_to is None and not (MODEL_DIR / "viability_report.json").exists():
        passed = _E32.stage_viability(args, model=model, tokenizer=tokenizer, is_vllm=is_vllm)
    else:
        rp = MODEL_DIR / "viability_report.json"
        if rp.exists():
            r = json.load(open(rp))
            passed = [s for s, v in r.items() if v.get("status") == "PASS"]
        else:
            passed = []
    if args.override_viability:
        passed = ["gsm8k", "mmlu", "bbh"]
    if not passed:
        logging.error("No sources passed viability. Exit.")
        return
    logging.info(f"Sources entering full pipeline: {passed}")

    if args.skip_to not in ("step2", "step3", "step4", "step5"):
        all_cot = _E32.step1_generate_cots(model, tokenizer, args, passed, is_vllm=is_vllm)
    else:
        all_cot = json.load(open(MODEL_DIR / "step1_cot_responses.json"))

    faithful = [r for r in all_cot if r["is_correct"]]
    logging.info(f"Faithful: {len(faithful)} / {len(all_cot)}")
    for src in passed:
        n = sum(1 for r in all_cot if r["source"] == src)
        c = sum(1 for r in all_cot if r["source"] == src and r["is_correct"])
        if n:
            logging.info(f"  {src}: {c}/{n} ({c/n*100:.1f}%)")

    if args.skip_to not in ("step3", "step4", "step5"):
        pairs = _E32.step2_create_perturbations(model, tokenizer, faithful, args, is_vllm=is_vllm)
    else:
        pairs = json.load(open(MODEL_DIR / "step2_pairs.json"))

    if args.skip_to not in ("step4", "step5"):
        pairs = _E32.step3_subclassify(pairs, args)
    else:
        pairs = json.load(open(MODEL_DIR / "step3_subclassified.json"))
    logging.info(f"Classification: {Counter(p.get('subtype','UNKNOWN') for p in pairs)}")

    if need_hf_for_step4 and is_vllm:
        logging.info("vLLM → HF swap for step 4...")
        import gc
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _, cache_kwargs = _E32._load_tokenizer()
        model, _, is_vllm = _E32.load_hf_model_for_generation(args.device, tokenizer, cache_kwargs)

    if args.skip_to not in ("step5",):
        feature_data = _E32.step4_extract_features(model, tokenizer, pairs, args)
    else:
        feature_data = dict(np.load(MODEL_DIR / "base_gemma_features.npz", allow_pickle=True))

    if model is not None:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not feature_data:
        return
    probe_results = _E32.step5_train_probes(feature_data, args)
    if not probe_results:
        return

    import pandas as pd
    df = pd.DataFrame(probe_results)
    df.to_csv(MODEL_DIR / "probe_results.csv", index=False)
    print("\nBest probe per task:")
    for task in df["task"].unique():
        sub = df[df["task"] == task]
        best = sub.loc[sub["accuracy"].idxmax()]
        print(f"  {task:32s} best={best['probe']:6s} L{int(best['layer']):2d} "
              f"acc={best['accuracy']:.3f} f1={best['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
