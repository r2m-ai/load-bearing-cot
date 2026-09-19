"""
exp_3_2_base_gemma_crossmodel.py — base Gemma-2-9B (NOT -IT) cross-model
validation (revision Theme 3, item 3.2).

Reviewer DzAa #3: the difficulty→faithfulness gradient is shown only on
instruction-tuned models. This script runs the full pipeline on the base
checkpoint to isolate the instruction-tuning effect at fixed architecture
and scale.

Per researcher sign-off 2026-05-19, we use **zero-shot prompting only**.
Base Gemma was never trained to follow chat templates or "Solution:"-style
instructions, so we use a `Question: … Step 1:` priming prefix that nudges
the base LM toward step-numbered generations without few-shot exemplars.
This means the base-vs-IT contrast jointly varies (a) instruction tuning
and (b) elicitation regime — call this the **3.5p prompting confound** and
state it explicitly in §5.5 / §6.3: the claim is scoped to "instruction
tuning + zero-shot elicitation jointly", not pure instruction-tuning.

Because the base model may not produce usable CoT at all on every task
(low base accuracy + thin step structure → no signal), we run an explicit
**viability go/no-go gate** as the first stage (item 3.4). The gate
samples ~50 questions per source and checks three conditions per source:
  (1) ≥40% of generations have ≥3 parseable steps (so perturbations exist),
  (2) ≥20% are correct (so there is a signal above floor),
  (3) ≥60% have an extractable Answer.
If any source fails, we skip it in step 1 and state that in the §5.5
write-up rather than reporting a pipeline that ran on noise.

Differences from script 14 (everything else is shared via import-by-path):

  1. NO chat template. `tokenizer.chat_template` is None on the base
     checkpoint; calling `apply_chat_template` would either error out or
     fall back to a template the model was never trained on. We build raw
     text prompts with `format_*_prompt_raw` and add a hard `Step 1:`
     priming token to coax step structure out of the base LM.

  2. Continuation prefix is raw concatenation:  `<raw_prompt>\\n<prefix>`.
     There are no chat-template suffix tokens to strip (the source of the
     `<|eot_id|>` / `<|im_end|>` handling in script 14 and exp_3_1).

  3. Feature anchor stays the same: hidden state at the last token of the
     raw prefix, across all 42 layers (Gemma-2-9B has same architecture as
     Gemma-2-9B-IT, just different weights, so the layer-21 finding from
     the IT model is a natural comparison point).

Caveats already wired into the response letter (DzAa §3 reply):
  - 3.5p prompting confound (above) — must surface in §5.5/§6.3.
  - 70B+ remains explicitly out of compute reach (3.3).

Usage:
  python experiments/models/base_gemma.py --stage viability
  python experiments/models/base_gemma.py --stage full
  python experiments/models/base_gemma.py --stage full \\
         --skip-to step4   # if step1-3 checkpoints exist

Outputs:
  data/processed/gemma2_9b_base/viability_report.json
  data/processed/gemma2_9b_base/step1_cot_responses.json
  data/processed/gemma2_9b_base/step2_pairs.json
  data/processed/gemma2_9b_base/step3_subclassified.json
  data/processed/gemma2_9b_base/base_gemma_features.npz
  data/processed/gemma2_9b_base/probe_results.csv
"""

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

random.seed(42)
np.random.seed(42)


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "experiments" / "pipeline" / "perturbation_helpers.py").exists():
            return p
    raise RuntimeError("Could not locate repo root from " + str(start))


ROOT = _find_repo_root(Path(__file__).resolve())
SCRIPTS = ROOT / "experiments"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_SLUG = "gemma2_9b_base"
MODEL_DIR = PROCESSED_DIR / MODEL_SLUG
HIDDEN_DIR = MODEL_DIR / "hidden_states"
LOG_DIR = ROOT / "logs"
for d in [MODEL_DIR, HIDDEN_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"exp_3_2_base_gemma_{timestamp}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "google/gemma-2-9b"   # base, not -it
# CACHE_DIR set to None forces vLLM + HF to use HF_HOME env var for caching
# (avoids creating a duplicate model_cache alongside hf_cache; 2026-05-20).
CACHE_DIR = None

# Base LMs are more verbose / less compliant; budget tokens accordingly,
# but cap to keep step 1 / step 2 wall-clock reasonable.
COT_MAX_NEW_TOKENS = 384
CONT_MAX_NEW_TOKENS_BASE = 256
CONT_MAX_NEW_TOKENS_CAP = 512

# Viability thresholds — fail-fast gates. Tune conservatively (item 3.4):
VIABILITY_N = 50            # samples per source for the gate
VIABILITY_MIN_STEPS_PCT = 0.40
VIABILITY_MIN_ACC = 0.20
VIABILITY_MIN_EXTRACT_PCT = 0.60


# ---- import script 14 by path (digit-prefixed; use the same canonical bits) ----

def _load_module(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LL = _load_module("models/llama.py", "llama_xmodel")

# Shared verbatim from script 14 (model-agnostic):
parse_cot_steps = _LL.parse_cot_steps
perturb_gsm8k_step = _LL.perturb_gsm8k_step
perturb_mmlu_step = _LL.perturb_mmlu_step
classify_pair = _LL.classify_pair
download_bbh = _LL.download_bbh
extract_gsm8k_answer = _LL.extract_gsm8k_answer
extract_mmlu_answer = _LL.extract_mmlu_answer
extract_bbh_answer = _LL.extract_bbh_answer
check_gsm8k_correct = _LL.check_gsm8k_correct
check_mmlu_correct = _LL.check_mmlu_correct
check_bbh_correct = _LL.check_bbh_correct
train_eval_probe = _LL.train_eval_probe
step5_train_probes = _LL.step5_train_probes


# ---- raw zero-shot prompt formatting (no chat template) ----
#
# Each prompt ends with a `Step 1:` (or `Step-by-step solution:\nStep 1:`)
# priming line. This is the cheapest defensible nudge: it does not provide
# any in-context solved examples (so it stays "zero-shot"), but it sets up
# the format the perturbation pipeline expects to parse.

def format_gsm8k_prompt_raw(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "After the final step, write the numeric answer on its own line as "
        "Answer: <number>.\n\n"
        f"Question: {question}\n\n"
        "Step-by-step solution:\n"
        "Step 1:"
    )


def format_mmlu_prompt_raw(question: str, choices: list[str]) -> str:
    choice_str = "\n".join(
        f"  {letter}) {text}"
        for letter, text in zip(["A", "B", "C", "D"], choices)
    )
    return (
        "Answer the following multiple choice question step by step. "
        "After the final step, write the final answer on its own line as "
        "Answer: <letter>.\n\n"
        f"Question: {question}\n{choice_str}\n\n"
        "Step-by-step solution:\n"
        "Step 1:"
    )


def format_bbh_prompt_raw(question: str) -> str:
    return (
        "Answer the following question step by step. "
        "After the final step, write the final answer on its own line as "
        "Answer: <your answer>.\n\n"
        f"Question: {question}\n\n"
        "Step-by-step solution:\n"
        "Step 1:"
    )


def build_raw_prompt(q: dict) -> str:
    src = q["source"]
    if src == "gsm8k":
        return format_gsm8k_prompt_raw(q["question"])
    if src == "mmlu":
        return format_mmlu_prompt_raw(q["question"], q["choices"])
    return format_bbh_prompt_raw(q["question"])


def _prepend_step1_marker(response: str) -> str:
    r"""
    Because the raw prompt ENDS with the literal "Step 1:" token, the
    model's continuation does not include that marker. For step-parsing to
    work we re-prepend it so the regex in `parse_cot_steps` sees the full
    first step (the parser keys on `\d+[.\)\:]` at line start).
    """
    if response.lstrip().startswith("Step "):
        return response
    return "Step 1:" + response


# ---- answer extraction & correctness (raw text; no chat tokens to strip) ----

def extract_answer_raw(response: str, source: str):
    full = _prepend_step1_marker(response)
    if source == "gsm8k":
        return extract_gsm8k_answer(full)
    if source == "mmlu":
        return extract_mmlu_answer(full)
    return extract_bbh_answer(full)


def check_correct_raw(extracted, example: dict) -> bool:
    if extracted is None:
        return False
    src = example["source"]
    if src == "gsm8k":
        return check_gsm8k_correct(extracted, example["reference_answer"])
    if src == "mmlu":
        return check_mmlu_correct(extracted, example.get("correct_letter", ""))
    return check_bbh_correct(extracted, example["reference_answer"])


# ---- model loading ----

def _load_tokenizer():
    from transformers import AutoTokenizer
    cache_kwargs = {"cache_dir": str(CACHE_DIR)} if CACHE_DIR else {}
    if CACHE_DIR:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **cache_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "chat_template", None):
        logging.warning("Base tokenizer unexpectedly has a chat_template; "
                        "ignored — zero-shot scope uses raw prompts.")
    return tokenizer, cache_kwargs


def load_model_for_generation(device="cuda", no_vllm=False):
    """
    Generation-only loader. Tries **vLLM** first (significantly faster on
    the long base-model continuations) and falls back to HuggingFace if
    vLLM is unavailable. Returns `(model, tokenizer, is_vllm)`. Step 4
    (hidden-state extraction) needs HF — main() reloads as HF before
    step 4 if generation used vLLM.
    """
    tokenizer, cache_kwargs = _load_tokenizer()

    if not no_vllm:
        try:
            # COMPAT PATCH (see exp_3_1 comment): vLLM 0.8.5 + transformers
            # 5.x missing-attr fix + force V0 engine (no flashinfer).
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
                PreTrainedTokenizerBase.all_special_tokens_extended = property(
                    lambda self: self.all_special_tokens)
            os.environ.setdefault("VLLM_USE_V1", "0")

            from vllm import LLM
            logging.info(f"Loading {MODEL_ID} with vLLM V0 (bf16, max_model_len=2560)...")
            model = LLM(
                model=MODEL_ID, dtype="bfloat16",
                gpu_memory_utilization=0.90,
                max_model_len=2560,
                enforce_eager=False,
                **({"download_dir": str(CACHE_DIR)} if CACHE_DIR else {}),
            )
            return model, tokenizer, True
        except Exception as e:
            logging.warning(f"vLLM unavailable / failed to load ({type(e).__name__}: {e}); "
                            f"falling back to HuggingFace.")

    return load_hf_model_for_generation(device, tokenizer, cache_kwargs)


def load_hf_model_for_generation(device, tokenizer, cache_kwargs):
    import torch
    from transformers import AutoModelForCausalLM

    logging.info(f"Loading {MODEL_ID} via HuggingFace (bfloat16)...")
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, device_map=device,
                attn_implementation=attn, **cache_kwargs)
            logging.info(f"Loaded with attn={attn}")
            break
        except Exception as e:
            logging.info(f"  {attn} unavailable: {type(e).__name__}: {e}")
    else:
        raise RuntimeError("Could not load base Gemma model")
    model.eval()
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                     f"mem after load: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    return model, tokenizer, False


def configure_h100():
    import torch
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _generate_batched(model, tokenizer, raw_prompts, max_new_tokens, batch_size, desc,
                       is_vllm=False):
    """
    Unified generation dispatcher. Raw text prompts (no chat template —
    base model). If `is_vllm`, runs vLLM batched inference (single call,
    all prompts at once); otherwise HF left-padded batched generation.
    """
    if is_vllm:
        return _generate_batched_vllm(model, raw_prompts, max_new_tokens, desc)
    return _generate_batched_hf(model, tokenizer, raw_prompts, max_new_tokens,
                                 batch_size, desc)


def _generate_batched_vllm(model, raw_prompts, max_new_tokens, desc):
    from vllm import SamplingParams
    logging.info(f"[{desc}] vLLM generate: {len(raw_prompts)} prompts, max_tokens={max_new_tokens}")
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outputs = model.generate(list(raw_prompts), sp)
    outputs = sorted(outputs, key=lambda o: int(o.request_id))
    return [o.outputs[0].text.strip() for o in outputs]


def _generate_batched_hf(model, tokenizer, raw_prompts, max_new_tokens, batch_size, desc):
    """HF batched generation; raw prompts (no chat template)."""
    import torch
    from tqdm import tqdm
    indexed = sorted(enumerate(raw_prompts), key=lambda x: len(x[1]))
    out_responses = [None] * len(raw_prompts)
    for bs in tqdm(range(0, len(indexed), batch_size), desc=desc):
        batch = indexed[bs: bs + batch_size]
        texts = [p for _, p in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, use_cache=True,
                                  pad_token_id=tokenizer.pad_token_id)
        in_len = inputs["input_ids"].shape[1]
        for (orig_idx, _), out in zip(batch, outs):
            out_responses[orig_idx] = tokenizer.decode(out[in_len:],
                                                       skip_special_tokens=True).strip()
    return out_responses


# ---- viability gate (item 3.4) ----

def _sample_for_viability(source: str) -> list[dict]:
    if source == "gsm8k":
        path = RAW_DIR / "gsm8k_full.json"
    elif source == "mmlu":
        path = RAW_DIR / "mmlu_full.json"
    elif source == "bbh":
        path = RAW_DIR / "bbh_full.json"
    if not path.exists() and source == "bbh":
        data = download_bbh(VIABILITY_N * 4)
        json.dump(data, open(path, "w"), indent=2)
    if not path.exists():
        return []
    data = json.load(open(path))
    random.Random(42).shuffle(data)
    return data[:VIABILITY_N]


def stage_viability(args, model=None, tokenizer=None, is_vllm=False):
    """
    Run a small probe to decide whether each source is worth full-pipeline
    treatment. Outputs a JSON report and prints a PASS/FAIL line per source.

    If `model` is provided (already loaded by main()), reuse it. Otherwise
    load fresh — supports the `--stage viability` standalone invocation.
    """
    report_path = MODEL_DIR / "viability_report.json"
    if model is None:
        model, tokenizer, is_vllm = load_model_for_generation(
            args.device, no_vllm=args.no_vllm)

    report = {}
    for src in ["gsm8k", "mmlu", "bbh"]:
        sample = _sample_for_viability(src)
        if not sample:
            logging.warning(f"[viability] no data for {src}; skipping")
            report[src] = {"status": "no_data"}
            continue
        for q in sample:
            q["source"] = src
        raw_prompts = [build_raw_prompt(q) for q in sample]
        t0 = time.time()
        responses = _generate_batched(
            model, tokenizer, raw_prompts, COT_MAX_NEW_TOKENS,
            args.batch_size, desc=f"viability-{src}", is_vllm=is_vllm)

        steps_counts = []
        extracts = []
        corrects = []
        for q, resp in zip(sample, responses):
            full = _prepend_step1_marker(resp or "")
            steps = parse_cot_steps(full)
            steps_counts.append(len(steps))
            ext = extract_answer_raw(resp or "", src)
            extracts.append(ext is not None)
            corrects.append(check_correct_raw(ext, q))

        n = len(sample)
        steps_pct = sum(1 for s in steps_counts if s >= 3) / n
        extract_pct = sum(extracts) / n
        acc = sum(corrects) / n
        pass_steps = steps_pct >= VIABILITY_MIN_STEPS_PCT
        pass_acc = acc >= VIABILITY_MIN_ACC
        pass_ext = extract_pct >= VIABILITY_MIN_EXTRACT_PCT
        status = "PASS" if (pass_steps and pass_acc and pass_ext) else "FAIL"

        report[src] = {
            "status": status,
            "n": n,
            "step_compliance_rate": steps_pct,
            "extract_rate": extract_pct,
            "accuracy": acc,
            "thresholds": {
                "step_compliance_rate_min": VIABILITY_MIN_STEPS_PCT,
                "extract_rate_min": VIABILITY_MIN_EXTRACT_PCT,
                "accuracy_min": VIABILITY_MIN_ACC,
            },
            "elapsed_s": time.time() - t0,
        }
        logging.info(f"[viability] {src}: {status}  n={n}  "
                     f"steps≥3 {steps_pct*100:.0f}% (>= {VIABILITY_MIN_STEPS_PCT*100:.0f}%)  "
                     f"extract {extract_pct*100:.0f}% (>= {VIABILITY_MIN_EXTRACT_PCT*100:.0f}%)  "
                     f"acc {acc*100:.0f}% (>= {VIABILITY_MIN_ACC*100:.0f}%)")

    json.dump(report, open(report_path, "w"), indent=2)
    logging.info(f"viability report -> {report_path}")

    passed_sources = [s for s, r in report.items() if r.get("status") == "PASS"]
    print("\n" + "=" * 60)
    print("VIABILITY GO/NO-GO")
    print("=" * 60)
    for s, r in report.items():
        print(f"  {s:6s}: {r.get('status', '?'):6s}  "
              f"steps={r.get('step_compliance_rate', 0)*100:5.1f}%  "
              f"extract={r.get('extract_rate', 0)*100:5.1f}%  "
              f"acc={r.get('accuracy', 0)*100:5.1f}%")
    print(f"PASSED sources: {passed_sources or '(none)'}")
    if not passed_sources:
        print("\n  -> All sources failed viability. Recommendation: state in §5.5 / §6.3 that")
        print("     base Gemma did not produce parseable ≥3-step CoT under zero-shot eliciting,")
        print("     making a cross-model comparison uninformative at this prompt regime.")
        print("     This is itself a finding (re: 3.5p prompting confound).")
    print("=" * 60)
    return passed_sources


# ---- step 1: full CoT generation, restricted to viability-PASS sources ----

def step1_generate_cots(model, tokenizer, args, allowed_sources, is_vllm=False):
    checkpoint_path = MODEL_DIR / "step1_cot_responses.json"
    if checkpoint_path.exists() and not args.force:
        all_results = json.load(open(checkpoint_path))
        logging.info(f"[Step 1] Loaded {len(all_results)} from checkpoint")
        return all_results

    all_results = []
    for src in allowed_sources:
        path = (RAW_DIR / f"{src}_full.json") if src != "bbh" else (RAW_DIR / "bbh_full.json")
        if not path.exists() and src == "bbh":
            data = download_bbh(args.max_examples)
            json.dump(data, open(path, "w"), indent=2)
        if not path.exists():
            logging.warning(f"{src} raw file missing; skipping")
            continue
        data = json.load(open(path))
        for q in data:
            q.setdefault("source", src)
        cap = args.max_examples or (5000 if src == "mmlu" else None)
        if cap and len(data) > cap:
            data = data[:cap]
            logging.info(f"Capped {src} to {cap}")
        prompts = [build_raw_prompt(q) for q in data]

        logging.info(f"[Step 1] Generating {len(data)} {src} CoTs ({COT_MAX_NEW_TOKENS} tok, "
                     f"backend={'vLLM' if is_vllm else 'HF'})...")
        t0 = time.time()
        responses = _generate_batched(model, tokenizer, prompts, COT_MAX_NEW_TOKENS,
                                       args.batch_size, desc=f"CoT-{src}", is_vllm=is_vllm)
        elapsed = time.time() - t0
        logging.info(f"  done in {elapsed:.0f}s ({len(data)/max(elapsed,1):.1f} ex/s)")

        for q, prompt, resp in zip(data, prompts, responses):
            full = _prepend_step1_marker(resp or "")
            ext = extract_answer_raw(resp or "", src)
            ok = check_correct_raw(ext, q)
            rec = {
                "id": q["id"],
                "source": src,
                "question": q["question"],
                "reference_answer": q.get("reference_answer", ""),
                "prompt": prompt,
                "cot_response": full,  # already step-1 prepended
                "raw_continuation": resp or "",
                "extracted_answer": ext,
                "is_correct": ok,
            }
            if src == "mmlu":
                rec["subject"] = q.get("subject", "")
                rec["choices"] = q.get("choices", [])
                rec["correct_letter"] = q.get("correct_letter", "")
            elif src == "bbh":
                rec["subject"] = q.get("subject", q.get("task", ""))
                rec["task"] = q.get("task", "")
            elif src == "gsm8k":
                rec["n_steps"] = q.get("n_steps", 0)
            all_results.append(rec)

        correct = sum(1 for r in all_results if r["source"] == src and r["is_correct"])
        logging.info(f"  {src} accuracy: {correct}/{sum(1 for r in all_results if r['source']==src)} "
                     f"({correct/max(1,sum(1 for r in all_results if r['source']==src))*100:.1f}%)")

    json.dump(all_results, open(checkpoint_path, "w"), indent=2)
    logging.info(f"[Step 1] Saved {len(all_results)} CoT responses to {checkpoint_path}")
    return all_results


# ---- step 2: perturb & continue (raw text; no chat template) ----

def _create_continuation_prefix_raw(example: dict, perturbation_point: str):
    steps = parse_cot_steps(example["cot_response"])
    if len(steps) < 4:
        return None
    pos_map = {
        "early": max(1, len(steps) // 4),
        "middle": len(steps) // 2,
        "late": min(len(steps) - 2, 3 * len(steps) // 4),
    }
    target_idx = pos_map.get(perturbation_point, len(steps) // 2)

    if example["source"] == "gsm8k":
        perturbed_step, desc, strategy = perturb_gsm8k_step(steps[target_idx])
    else:
        perturbed_step, desc, strategy = perturb_mmlu_step(
            steps[target_idx], example.get("correct_letter", "A"),
            example.get("choices"))
    if desc == "no_change":
        return None

    prefix_steps = steps[:target_idx] + [perturbed_step]
    prefix_text = "\n".join(prefix_steps)
    original_prefix_text = "\n".join(steps[: target_idx + 1])

    return {
        "prefix": prefix_text,
        "original_prefix": original_prefix_text,
        "perturbation_description": f"{desc} at step {target_idx}/{len(steps)} ({perturbation_point})",
        "perturbation_strategy": strategy,
        "perturbation_point": perturbation_point,
        "target_step_idx": target_idx,
        "total_steps": len(steps),
        "original_step": steps[target_idx],
        "perturbed_step": perturbed_step,
        "original_remaining": "\n".join(steps[target_idx + 1:]),
    }


_STEP1_PRIMING_RE = re.compile(r"Step\s*1\s*:\s*$")


def _build_continuation_input_raw(prompt: str, prefix: str) -> str:
    """
    Raw-text continuation prompt for a base LM.

    The original generation-time prompt ends with the `Step 1:` priming
    token (see `format_*_prompt_raw`) to coax step structure out of the
    base LM. The perturbed prefix, however, already CONTAINS the full
    "Step 1: …" line (since `parse_cot_steps` returned it as steps[0]).
    Naively concatenating would produce a duplicated marker:

        prompt  : "... Step-by-step solution:\nStep 1:"
        prefix  : "Step 1: Sarah has 5 apples.\nStep 2 (PERTURBED): …"
        concat  : "... Step 1: Step 1: Sarah has 5 apples. …"   ← duplicate

    Strip the trailing `Step 1:` priming from the prompt before joining
    so the continuation input is a clean single CoT trace.
    """
    cleaned_prompt = _STEP1_PRIMING_RE.sub("", prompt).rstrip()
    return f"{cleaned_prompt}\n{prefix}"


def _estimate_max_tokens_raw(item: dict) -> int:
    total_steps = item.get("total_steps", 6)
    target_idx = item.get("target_step_idx", total_steps // 2)
    remaining = (total_steps - target_idx) / max(total_steps, 1)
    est = int(CONT_MAX_NEW_TOKENS_BASE * remaining * 1.5) + 64
    return max(96, min(est, CONT_MAX_NEW_TOKENS_CAP))


def step2_create_perturbations(model, tokenizer, faithful_examples, args, is_vllm=False):
    checkpoint_path = MODEL_DIR / "step2_pairs.json"
    if checkpoint_path.exists() and not args.force:
        pairs = json.load(open(checkpoint_path))
        logging.info(f"[Step 2] Loaded {len(pairs)} from checkpoint")
        return pairs

    perturbation_points = ["early", "middle", "late"]
    pending = []
    for ex in faithful_examples:
        for point in perturbation_points:
            r = _create_continuation_prefix_raw(ex, point)
            if r is None:
                continue
            pending.append({
                "id": ex["id"],
                "source": ex["source"],
                "question": ex["question"],
                "reference_answer": ex.get("reference_answer", ""),
                "correct_letter": ex.get("correct_letter", ""),
                "subject": ex.get("subject", ""),
                "choices": ex.get("choices"),
                "prompt": ex["prompt"],
                "original_cot": ex["cot_response"],
                **r,
            })

    logging.info(f"[Step 2] {len(pending)} perturbation jobs from "
                 f"{len(faithful_examples)} faithful baselines")

    # Render continuation inputs once (shared by vLLM/HF paths and matches step 4).
    raw_inputs = [_build_continuation_input_raw(item["prompt"], item["prefix"])
                  for item in pending]
    # vLLM uses a single global max_tokens; HF allows per-batch maxima. Use the
    # max across all items so no example is silently truncated under vLLM.
    max_tok = max((_estimate_max_tokens_raw(item) for item in pending),
                  default=CONT_MAX_NEW_TOKENS_CAP)

    t0 = time.time()
    all_conts = _generate_batched(
        model, tokenizer, raw_inputs, max_tok, args.batch_size,
        desc="Continuing", is_vllm=is_vllm)
    logging.info(f"[Step 2] Done in {time.time()-t0:.0f}s "
                 f"({len(pending)/max(time.time()-t0,1):.1f} ex/s)")

    pairs = []
    for item, cont in zip(pending, all_conts):
        cont = cont or ""
        full_response = item["prefix"] + "\n" + cont
        if item["source"] == "gsm8k":
            ext = extract_gsm8k_answer(cont)
        elif item["source"] == "mmlu":
            ext = extract_mmlu_answer(cont)
        else:
            ext = extract_bbh_answer(cont)
        ok = check_correct_raw(ext, item)
        if ext is None:
            label = "unclear"
        elif ok:
            label = "self_corrects"
        else:
            label = "propagates_error"
        binary_label = {"self_corrects": "unfaithful",
                        "propagates_error": "faithful"}.get(label, "unclear")

        pairs.append({
            "id": item["id"],
            "source": item["source"],
            "question": item["question"],
            "reference_answer": item["reference_answer"],
            "prompt": item["prompt"],
            "faithful_cot": item["original_cot"],
            "faithful_answer": item["reference_answer"],
            "perturbed_cot": full_response,
            "perturbation_description": item["perturbation_description"],
            "perturbation_strategy": item["perturbation_strategy"],
            "perturbation_point": item["perturbation_point"],
            "target_step_idx": item["target_step_idx"],
            "total_steps": item["total_steps"],
            "original_step": item["original_step"],
            "perturbed_step": item["perturbed_step"],
            "original_prefix": item["original_prefix"],
            "prefix": item["prefix"],
            "continuation": cont,
            "perturbed_answer": ext,
            "model_ignores_perturbation": ok,
            "behavior": label,
            "label": binary_label,
            "correct_letter": item.get("correct_letter", ""),
            "subject": item.get("subject", ""),
            "choices": item.get("choices"),
        })

    json.dump(pairs, open(checkpoint_path, "w"), indent=2)
    logging.info(f"[Step 2] Saved {len(pairs)} pairs to {checkpoint_path}")
    return pairs


# ---- step 3: subclassify (canonical rule-based) ----

def step3_subclassify(pairs, args):
    checkpoint_path = MODEL_DIR / "step3_subclassified.json"
    if checkpoint_path.exists() and not args.force:
        classified = json.load(open(checkpoint_path))
        logging.info(f"[Step 3] Loaded {len(classified)} from checkpoint")
        return classified

    type_counts = Counter()
    for pair in pairs:
        subtype, reason = classify_pair(pair)
        pair["subtype"] = subtype
        pair["subtype_reason"] = reason
        pair["label_3class"] = {
            "TYPE_A": "silent_bypass", "TYPE_B": "self_correction",
            "TYPE_C": "error_propagation",
        }.get(subtype, "unclear")
        type_counts[subtype] += 1
    logging.info(f"[Step 3] Sub-classification: {dict(type_counts)}")
    json.dump(pairs, open(checkpoint_path, "w"), indent=2)
    return pairs


# ---- step 4: hidden state extraction at the raw-prefix boundary ----

def step4_extract_features(model, tokenizer, pairs, args):
    """
    Anchor at the LAST token of the raw continuation input (i.e., the end
    of the perturbed prefix). All 42 Gemma-2-9B layers; same architecture
    as -IT so the layer-21 finding from the IT model is the natural
    comparison point.
    """
    import torch
    from tqdm import tqdm

    features_path = MODEL_DIR / "base_gemma_features.npz"
    if features_path.exists() and not args.force:
        logging.info(f"[Step 4] Loading features from {features_path}")
        return dict(np.load(features_path, allow_pickle=True))

    valid = [p for p in pairs if p.get("label_3class") in
             ("silent_bypass", "self_correction", "error_propagation")]
    logging.info(f"[Step 4] {len(valid)} valid pairs (from {len(pairs)} total)")
    if not valid:
        return {}

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    logging.info(f"n_layers={n_layers}, hidden_dim={hidden_dim}")

    texts = [_build_continuation_input_raw(p["prompt"], p["prefix"]) for p in valid]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    bsz = args.batch_size

    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"  # last real token aligned at position -1

    feats = [None] * len(valid)
    for bs in tqdm(range(0, len(order), bsz), desc="Extract features (batched)"):
        idxs = order[bs: bs + bsz]
        batch_texts = [texts[i] for i in idxs]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=2560).to(model.device)
        with torch.no_grad():
            outs = model(**inputs, output_hidden_states=True, use_cache=False)
        per_layer = []
        for layer_idx in range(n_layers):
            h = outs.hidden_states[layer_idx + 1]
            per_layer.append(h[:, -1, :].float().cpu().numpy())  # (B, D)
        layer_stack = np.stack(per_layer, axis=1)  # (B, n_layers, D)
        for j, orig_i in enumerate(idxs):
            feats[orig_i] = layer_stack[j]
        del outs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tokenizer.padding_side = old_pad
    features = np.stack(feats)
    save_dict = {
        "pair_ids": np.array([p["id"] for p in valid]),
        "perturbation_points": np.array([p.get("perturbation_point", "") for p in valid]),
        "labels": np.array([p.get("label_3class", "unclear") for p in valid]),
        "sources": np.array([p["source"] for p in valid]),
    }
    for layer_idx in range(n_layers):
        save_dict[f"layer_{layer_idx}"] = features[:, layer_idx, :]
    np.savez_compressed(features_path, **save_dict)
    logging.info(f"[Step 4] Saved features to {features_path}")
    return save_dict


# ---- main ----

def main():
    parser = argparse.ArgumentParser(
        description=f"{MODEL_ID} base-model cross-model validation (item 3.2)")
    parser.add_argument("--stage", choices=["viability", "full"], default="full",
                        help="viability: gate only (item 3.4). full: viability → "
                             "step1..5; sources that fail viability are skipped.")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="per-dataset cap for smoke runs")
    parser.add_argument("--batch-size", type=int, default=24,
                        help="HF generation batch size. Ignored when vLLM is used. "
                             "Default 24 fits 9B on H200 141GB; bump if VRAM permits.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable the vLLM fast-path (faster on long generations). "
                             "Use only for debugging.")
    parser.add_argument("--skip-to", type=str, default=None,
                        choices=["step2", "step3", "step4", "step5"],
                        help="resume the full pipeline from a checkpointed stage")
    parser.add_argument("--force", action="store_true",
                        help="re-run even if a checkpoint exists")
    parser.add_argument("--override-viability", action="store_true",
                        help="run step1..5 for ALL sources regardless of the gate "
                             "(use only if you intend to report the failure case)")
    args = parser.parse_args()

    t_start = time.time()
    configure_h100()

    logging.info("=" * 64)
    logging.info(f"BASE GEMMA CROSS-MODEL PIPELINE — {MODEL_ID}")
    logging.info("=" * 64)
    logging.info(f"Stage: {args.stage}")
    logging.info(f"Max examples: {args.max_examples or 'all'}")
    logging.info(f"Output dir: {MODEL_DIR}")
    logging.info("NB: zero-shot prompting only. The base-vs-IT contrast confounds "
                 "instruction tuning with elicitation regime (3.5p) — state in §5.5.")

    if args.stage == "viability":
        stage_viability(args)
        return

    # full stage — model lifecycle: load once (vLLM if possible) → run
    # viability + step 1 + step 2 → swap to HF before step 4 (hidden states).
    model = None
    tokenizer = None
    is_vllm = False
    need_generation = args.skip_to not in ("step3", "step4", "step5")
    need_hf_for_step4 = args.skip_to not in ("step5",)

    if need_generation:
        model, tokenizer, is_vllm = load_model_for_generation(
            args.device, no_vllm=args.no_vllm)
    elif need_hf_for_step4:
        tokenizer, cache_kwargs = _load_tokenizer()
        model, tokenizer, is_vllm = load_hf_model_for_generation(
            args.device, tokenizer, cache_kwargs)

    # Viability — runs only if generation will run AND we don't already have a report
    if args.skip_to is None and not (MODEL_DIR / "viability_report.json").exists():
        passed = stage_viability(args, model=model, tokenizer=tokenizer, is_vllm=is_vllm)
    else:
        report_path = MODEL_DIR / "viability_report.json"
        if report_path.exists():
            report = json.load(open(report_path))
            passed = [s for s, r in report.items() if r.get("status") == "PASS"]
        else:
            passed = []
    if args.override_viability:
        logging.warning("--override-viability set; running full pipeline on all sources "
                        "regardless of gate result.")
        passed = ["gsm8k", "mmlu", "bbh"]
    if not passed:
        logging.error("No sources passed the viability gate. Exit. "
                      "Use --override-viability to run anyway (only if reporting the failure).")
        return
    logging.info(f"Sources entering full pipeline: {passed}")

    # Step 1
    if args.skip_to not in ("step2", "step3", "step4", "step5"):
        all_cot = step1_generate_cots(model, tokenizer, args, passed, is_vllm=is_vllm)
    else:
        all_cot = json.load(open(MODEL_DIR / "step1_cot_responses.json"))
        logging.info(f"Loaded {len(all_cot)} CoT responses from checkpoint")

    faithful = [r for r in all_cot if r["is_correct"]]
    logging.info(f"Faithful examples: {len(faithful)} / {len(all_cot)}")
    for src in passed:
        n = sum(1 for r in all_cot if r["source"] == src)
        c = sum(1 for r in all_cot if r["source"] == src and r["is_correct"])
        if n:
            logging.info(f"  {src}: {c}/{n} ({c/n*100:.1f}%)")

    # Step 2
    if args.skip_to not in ("step3", "step4", "step5"):
        pairs = step2_create_perturbations(model, tokenizer, faithful, args, is_vllm=is_vllm)
    else:
        pairs = json.load(open(MODEL_DIR / "step2_pairs.json"))

    # Step 3 (CPU-only)
    if args.skip_to not in ("step4", "step5"):
        pairs = step3_subclassify(pairs, args)
    else:
        pairs = json.load(open(MODEL_DIR / "step3_subclassified.json"))
    logging.info(f"Classification: {Counter(p.get('subtype','UNKNOWN') for p in pairs)}")

    # vLLM → HF swap before step 4 (vLLM can't expose hidden states).
    if need_hf_for_step4 and is_vllm:
        logging.info("Step 4 requires HF backend (vLLM cannot expose hidden states). "
                     "Freeing vLLM model and reloading as HuggingFace...")
        import torch, gc
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _, cache_kwargs = _load_tokenizer()
        model, _, is_vllm = load_hf_model_for_generation(args.device, tokenizer, cache_kwargs)

    # Step 4
    if args.skip_to not in ("step5",):
        feature_data = step4_extract_features(model, tokenizer, pairs, args)
    else:
        feature_data = dict(np.load(MODEL_DIR / "base_gemma_features.npz",
                                    allow_pickle=True))

    if model is not None:
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Freed GPU memory for probe training")

    # Step 5 — reuse script 14's probe training verbatim
    if not feature_data:
        logging.error("No features available -- cannot train probes")
        return
    probe_results = step5_train_probes(feature_data, args)
    if not probe_results:
        logging.error("No probe results")
        return

    import pandas as pd
    df = pd.DataFrame(probe_results)
    df.to_csv(MODEL_DIR / "probe_results.csv", index=False)
    logging.info(f"Saved probe results to {MODEL_DIR / 'probe_results.csv'}")
    print("\nBest probe per task:")
    for task in df["task"].unique():
        sub = df[df["task"] == task]
        best = sub.loc[sub["accuracy"].idxmax()]
        print(f"  {task:32s}  best={best['probe']:6s} L{int(best['layer']):2d}  "
              f"acc={best['accuracy']:.3f}  f1={best['f1_macro']:.3f}")
    print(f"\nTotal wall time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
