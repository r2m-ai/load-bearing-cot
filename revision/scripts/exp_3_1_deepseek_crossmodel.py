"""
exp_3_1_deepseek_crossmodel.py — DeepSeek-R1-Distill-Qwen-7B cross-model
validation (revision Theme 3, item 3.1).

Reviewers DzAa #3 + y8FE: replicate on a *reasoning-specialized* model to
test whether the difficulty→faithfulness gradient is specific to
instruction-tuned 8–9B models. Per researcher sign-off 2026-05-19, we use
**DeepSeek-R1-Distill-Qwen-7B** (closer scale match to the 8–9B main models
than the 1.5B distill; 7B distill is the compute driver).

DeepSeek-R1-Distill emits reasoning inside `<think>…</think>` blocks before
the final answer. The cross-model pipeline therefore differs from
`scripts/14_llama_crossmodel.py` in three load-bearing places — everywhere
else (perturbation strategies, rule-based subclassifier, probe training,
output formats) is matched verbatim by importing the canonical functions
from script 14 by path. Differences:

  1. parse_deepseek_cot_steps : extract the *content of the first
     `<think>…</think>` block* and parse it with script 14's
     `parse_cot_steps`. If the trace is unterminated (model ran out of
     tokens), we accept everything after `<think>` up to a soft cap.

  2. extract_*_answer (DeepSeek variants) : take the substring *after*
     `</think>` (the post-think final-answer slab) and apply script 14's
     existing regexes there; fall back to `\\boxed{…}` if no `Answer:`
     line is present. R1 distills frequently use `\\boxed`.

  3. _build_continuation_prompt_deepseek : the perturbed prefix is the
     content of `<think>` truncated at the perturbed step, *still inside
     the open `<think>` block*. We re-inject the opening `<think>` tag,
     do NOT close it, and let the model decide when to emit `</think>`
     and the answer. Trailing `<｜end▁of▁sentence｜>` and Qwen-style
     `<|im_end|>` are stripped to keep the open-think continuation valid.

Compute (rough budget on a single H100):
  step 1 (CoT generation, ~6.7K questions, ~1024 new tokens for `<think>`):
                                                ~2–3 h (the compute driver).
  step 2 (perturbed continuations, ~9–14K jobs, ~768 new tokens):  ~2 h.
  step 3 rule-based / step 4 features / step 5 probes:             ~30–40 min.
  Total ~5–6 h end-to-end. Sequence AFTER the confound result (item 1.1)
  is secured, per the within-P0-GPU priority order.

Caveat to surface in §5.5 / §6.3 (Method Spec 3.5): DeepSeek-R1 is RL-
trained — Chen2025a found this regime to be *more* faithful than IT
baselines — so `<think>`-block perturbation may not be 1-to-1 metric-
comparable with Gemma-IT / Llama-IT. State this; do not paper over it.

Usage:
  python revision/scripts/exp_3_1_deepseek_crossmodel.py
  python revision/scripts/exp_3_1_deepseek_crossmodel.py --skip-to step4
  python revision/scripts/exp_3_1_deepseek_crossmodel.py --max-examples 200 \\
         --batch-size 16    # smoke run

Outputs:
  data/processed/deepseek_r1_distill_qwen_7b/step1_cot_responses.json
  data/processed/deepseek_r1_distill_qwen_7b/step2_pairs.json
  data/processed/deepseek_r1_distill_qwen_7b/step3_subclassified.json
  data/processed/deepseek_r1_distill_qwen_7b/deepseek_features.npz
  data/processed/deepseek_r1_distill_qwen_7b/probe_results.csv
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
        if (p / "scripts" / "create_perturbations_v8_helpers.py").exists():
            return p
    raise RuntimeError("Could not locate repo root from " + str(start))


ROOT = _find_repo_root(Path(__file__).resolve())
SCRIPTS = ROOT / "scripts"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_SLUG = "deepseek_r1_distill_qwen_7b"
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
        logging.FileHandler(LOG_DIR / f"exp_3_1_deepseek_{timestamp}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
# CACHE_DIR set to None forces vLLM + HF to use HF_HOME env var for caching
# (avoids creating a duplicate model_cache alongside hf_cache; 2026-05-20).
CACHE_DIR = None

# DeepSeek-R1 traces are 2–3× longer than IT-baselines; budget accordingly.
# 2026-05-20: bumped from 1024 → 2048 after observing 38.3% GSM8K accuracy
# in the first H200 run — suspected `<think>` truncation. Public benchmarks
# report ~85% on GSM8K for this distill, so the truncation hypothesis is
# strong. vLLM handles longer outputs efficiently; expect ~30% wall-time
# increase for step 1.
COT_MAX_NEW_TOKENS = 2048
CONT_MAX_NEW_TOKENS_BASE = 768
CONT_MAX_NEW_TOKENS_CAP = 1024


# ---- import script 14 by path (digit prefix prevents normal import) ----

def _load_module(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Pull in canonical, well-tested helpers verbatim from script 14:
#   parse_cot_steps             — step regex (universal, model-agnostic)
#   format_gsm8k_prompt / format_mmlu_prompt / format_bbh_prompt
#                               — prompt scaffolding (instruction style is
#                                 compatible with R1 distills)
#   perturb_gsm8k_step / perturb_mmlu_step
#                               — 7 strategies (same as the paper)
#   classify_pair               — rule-based TYPE_A/B/C
#   download_bbh                — dataset download
#   train_eval_probe / step5_train_probes
#                               — probe training (identical metric reporting)
_LL = _load_module("14_llama_crossmodel.py", "llama_xmodel")

parse_cot_steps_inner = _LL.parse_cot_steps
format_gsm8k_prompt = _LL.format_gsm8k_prompt
format_mmlu_prompt = _LL.format_mmlu_prompt
format_bbh_prompt = _LL.format_bbh_prompt
perturb_gsm8k_step = _LL.perturb_gsm8k_step
perturb_mmlu_step = _LL.perturb_mmlu_step
classify_pair = _LL.classify_pair
download_bbh = _LL.download_bbh
train_eval_probe = _LL.train_eval_probe
step5_train_probes = _LL.step5_train_probes


# ---- DeepSeek-R1 specific: <think>...</think> block handling ----

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_think_and_answer(response: str) -> tuple[str, str]:
    """
    Return (think_content, post_think_slab).

    For DeepSeek-R1-Distill: the chat template's `add_generation_prompt=True`
    path auto-prepends `<｜Assistant｜><think>\\n` to the prompt — so the
    opening `<think>` lives in the PROMPT, not in the model's OUTPUT. The
    model's output starts directly with reasoning text and ends with
    `</think>\\n\\n[formal structured answer]`. We therefore split on
    `</think>` only.

    Verified 2026-05-20 on n=338 correct GSM8K responses: 0/338 contained
    `<think>` (the open tag) but 100% contained `</think>` (the close).
    """
    if _THINK_CLOSE in response:
        think, post = response.split(_THINK_CLOSE, 1)
        # Strip any leading `<think>` if the tokenizer didn't skip it
        # (defensive — recent transformers/vLLM versions sometimes preserve it).
        think = think.lstrip()
        if think.startswith(_THINK_OPEN):
            think = think[len(_THINK_OPEN):].lstrip()
        return think.strip(), post.strip()
    # No </think> — model either still thinking (truncated at cap) or
    # produced no formal answer yet. Treat the whole response as think
    # content; downstream step 2 will skip baselines whose post-think is
    # missing or too short.
    return response.strip(), ""


def parse_deepseek_cot_steps(response: str) -> list[str]:
    """
    For DeepSeek-R1, parse the **post-`</think>` structured presentation**,
    not the pre-`</think>` internal reasoning. Verified 2026-05-20:
    post-think text consistently contains numbered steps ("1. **...** ...
    2. **...** ...") matching the canonical step regex, while pre-think
    is free-form natural language ("Okay, so I need to figure out...") that
    the regex never picks up. Falls back to raw response if no `</think>`.
    """
    _, post = split_think_and_answer(response)
    text_for_parsing = post if post else response
    return parse_cot_steps_inner(text_for_parsing)


def _last_match(pattern, text):
    """Return the LAST regex match (closest to the final answer)."""
    matches = list(re.finditer(pattern, text))
    return matches[-1] if matches else None


def extract_deepseek_gsm8k_answer(response: str) -> str | None:
    """
    Score the slab *after* `</think>` first; fall back to `\\boxed{…}`
    anywhere; final fallback is the existing GSM8K extractor on the raw
    response (matches Llama's behavior for any tail that lacks a clean
    `Answer:` line).
    """
    _, post = split_think_and_answer(response)
    target = post if post else response
    m = _last_match(r"Answer:\s*\$?([\-\+]?[\d,]+\.?\d*)", target)
    if m:
        return m.group(1).replace(",", "")
    m = _last_match(r"\\boxed\{\s*([\-\+]?[\d,]+\.?\d*)\s*\}", target)
    if m:
        return m.group(1).replace(",", "")
    return _LL.extract_gsm8k_answer(target)


def extract_deepseek_mmlu_answer(response: str) -> str | None:
    _, post = split_think_and_answer(response)
    target = post if post else response
    m = _last_match(r"\\boxed\{\s*\(?([A-Da-d])\)?\s*\}", target)
    if m:
        return m.group(1).upper()
    return _LL.extract_mmlu_answer(target)


def extract_deepseek_bbh_answer(response: str) -> str | None:
    _, post = split_think_and_answer(response)
    target = post if post else response
    m = _last_match(r"\\boxed\{\s*(.+?)\s*\}", target)
    if m:
        return m.group(1).strip()
    return _LL.extract_bbh_answer(target)


def check_deepseek_correct(extracted: str | None, example: dict) -> bool:
    if extracted is None:
        return False
    src = example["source"]
    if src == "gsm8k":
        return _LL.check_gsm8k_correct(extracted, example["reference_answer"])
    if src == "mmlu":
        return _LL.check_mmlu_correct(extracted, example.get("correct_letter", ""))
    return _LL.check_bbh_correct(extracted, example["reference_answer"])


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
    return tokenizer, cache_kwargs


def load_model_for_generation(device="cuda", no_vllm=False):
    """
    Generation-only loader. Tries **vLLM** first (5–10× throughput on long
    `<think>` traces — the compute driver for this script) and falls back
    to HuggingFace if vLLM is unavailable or fails to load. Returns
    `(model, tokenizer, is_vllm)`. The `is_vllm` flag tells the generation
    helpers which dispatch path to use, and `step4_extract_features` knows
    to reload as HF (vLLM does not expose hidden states via forward hooks).
    """
    tokenizer, cache_kwargs = _load_tokenizer()

    if not no_vllm:
        try:
            # COMPAT PATCH (2026-05-19): vLLM 0.8.5 expects
            # `all_special_tokens_extended` on tokenizers, but transformers
            # 5.x removed it from the slow `*Tokenizer` classes (incl.
            # GemmaTokenizer, Qwen2Tokenizer). Add it back as a property
            # aliased to `all_special_tokens` before vLLM imports the
            # tokenizer. Also force the V0 engine — V1 pulls in flashinfer,
            # which on this stack has a CXX11-ABI mismatch with the torch
            # 2.6 downgrade we did to satisfy CUDA 12.4 driver compat.
            from transformers.tokenization_utils_base import PreTrainedTokenizerBase
            if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
                PreTrainedTokenizerBase.all_special_tokens_extended = property(
                    lambda self: self.all_special_tokens)
            os.environ.setdefault("VLLM_USE_V1", "0")

            from vllm import LLM
            logging.info(f"Loading {MODEL_ID} with vLLM V0 (bf16, max_model_len=4096)...")
            model = LLM(
                model=MODEL_ID, dtype="bfloat16",
                gpu_memory_utilization=0.90,
                # R1 distill <think> traces routinely exceed 2k; allow 4k.
                max_model_len=4096,
                enforce_eager=False,
                **({"download_dir": str(CACHE_DIR)} if CACHE_DIR else {}),
            )
            return model, tokenizer, True
        except Exception as e:
            logging.warning(f"vLLM unavailable / failed to load ({type(e).__name__}: {e}); "
                            f"falling back to HuggingFace.")

    return load_hf_model_for_generation(device, tokenizer, cache_kwargs)


def load_hf_model_for_generation(device, tokenizer, cache_kwargs):
    """HF generation loader (used as vLLM fallback AND for step 4 hidden states)."""
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
        raise RuntimeError("Could not load model")
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


def apply_chat_template_deepseek(tokenizer, prompt: str) -> str:
    """
    DeepSeek-R1-Distill-Qwen-7B ships a Qwen-style chat template. Use it
    via the official `apply_chat_template` API so we don't hard-code the
    role markers (which differ between distill variants).
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )


# ---- step 1: generate CoTs (with R1-style <think>) ----

def _generate_batched(model, tokenizer, prompts, max_new_tokens, batch_size, desc,
                       is_vllm=False, pre_rendered=False):
    """
    Unified generation dispatcher. If `is_vllm`, runs vLLM batched
    inference (single call, all prompts at once — vLLM handles scheduling
    internally). Otherwise falls back to HuggingFace left-padded batched
    generation. `pre_rendered=True` means `prompts` are already raw text
    (chat template / `<think>` injection already applied); otherwise we
    wrap each prompt in `apply_chat_template_deepseek`.
    """
    if is_vllm:
        return _generate_batched_vllm(model, prompts, max_new_tokens, desc,
                                       tokenizer=tokenizer, pre_rendered=pre_rendered)
    return _generate_batched_hf(model, tokenizer, prompts, max_new_tokens,
                                 batch_size, desc, pre_rendered=pre_rendered)


def _generate_batched_vllm(model, prompts, max_new_tokens, desc, tokenizer=None,
                            pre_rendered=False):
    """vLLM path: one batched call across all prompts; per-request max_tokens."""
    from vllm import SamplingParams
    formatted = (prompts if pre_rendered else
                 [apply_chat_template_deepseek(tokenizer, p) for p in prompts])
    logging.info(f"[{desc}] vLLM generate: {len(formatted)} prompts, max_tokens={max_new_tokens}")
    sp = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)
    outputs = model.generate(formatted, sp)
    # vLLM preserves submission order in the returned list, but the request_id
    # field is the canonical identifier; sort by it as a defensive measure
    # (matches the exp_1_1 pattern).
    outputs = sorted(outputs, key=lambda o: int(o.request_id))
    return [o.outputs[0].text.strip() for o in outputs]


def _generate_batched_hf(model, tokenizer, prompts, max_new_tokens, batch_size, desc,
                          pre_rendered=False):
    """HF path: left-padded batched generation, sorted by input length."""
    import torch
    from tqdm import tqdm
    formatted_all = (prompts if pre_rendered else
                     [apply_chat_template_deepseek(tokenizer, p) for p in prompts])
    indexed = sorted(enumerate(formatted_all), key=lambda x: len(x[1]))
    responses = [None] * len(formatted_all)
    for batch_start in tqdm(range(0, len(indexed), batch_size), desc=desc):
        batch = indexed[batch_start: batch_start + batch_size]
        inputs = tokenizer([t for _, t in batch], return_tensors="pt", padding=True,
                           truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.pad_token_id)
        in_len = inputs["input_ids"].shape[1]
        for (orig_idx, _), out in zip(batch, outputs):
            responses[orig_idx] = tokenizer.decode(out[in_len:], skip_special_tokens=True).strip()
    return responses


def generate_cot_for_source(model, tokenizer, questions, source, batch_size, is_vllm=False):
    logging.info(f"[Step 1] Generating CoTs for {len(questions)} {source} questions "
                 f"(batch={batch_size}, max_new_tokens={COT_MAX_NEW_TOKENS}, "
                 f"backend={'vLLM' if is_vllm else 'HF'})...")
    prompts = []
    for q in questions:
        if source == "gsm8k":
            prompts.append(format_gsm8k_prompt(q["question"]))
        elif source == "mmlu":
            prompts.append(format_mmlu_prompt(q["question"], q["choices"]))
        elif source == "bbh":
            prompts.append(format_bbh_prompt(q["question"]))

    t0 = time.time()
    responses = _generate_batched(model, tokenizer, prompts, COT_MAX_NEW_TOKENS,
                                  batch_size, desc=f"CoT-{source}", is_vllm=is_vllm)
    elapsed = time.time() - t0
    logging.info(f"{source} generation done in {elapsed:.0f}s ({len(questions)/max(elapsed,1):.1f} ex/s)")

    results = []
    correct = 0
    for q, prompt, response in zip(questions, prompts, responses):
        if source == "gsm8k":
            ext = extract_deepseek_gsm8k_answer(response or "")
            ok = _LL.check_gsm8k_correct(ext, q["reference_answer"])
        elif source == "mmlu":
            ext = extract_deepseek_mmlu_answer(response or "")
            ok = _LL.check_mmlu_correct(ext, q.get("correct_letter", ""))
        else:
            ext = extract_deepseek_bbh_answer(response or "")
            ok = _LL.check_bbh_correct(ext, q["reference_answer"])
        if ok:
            correct += 1

        # Also stash how the model used <think> for diagnostics.
        think, post = split_think_and_answer(response or "")
        rec = {
            "id": q["id"],
            "source": source,
            "question": q["question"],
            "reference_answer": q.get("reference_answer", ""),
            "prompt": prompt,
            "cot_response": response or "",
            "think_content": think,
            "post_think": post,
            "think_steps_count": len(parse_cot_steps_inner(think)) if think else 0,
            "extracted_answer": ext,
            "is_correct": ok,
        }
        if source == "gsm8k":
            rec["n_steps"] = q.get("n_steps", 0)
        elif source == "mmlu":
            rec["subject"] = q.get("subject", "")
            rec["choices"] = q.get("choices", [])
            rec["correct_letter"] = q.get("correct_letter", "")
        elif source == "bbh":
            rec["subject"] = q.get("subject", q.get("task", ""))
            rec["task"] = q.get("task", "")
        results.append(rec)

    acc = correct / max(len(results), 1) * 100
    logging.info(f"{source} accuracy: {correct}/{len(results)} ({acc:.1f}%)")
    return results


def step1_generate_cots(model, tokenizer, args, is_vllm=False):
    checkpoint_path = MODEL_DIR / "step1_cot_responses.json"
    if checkpoint_path.exists() and not args.force:
        with open(checkpoint_path) as f:
            all_results = json.load(f)
        logging.info(f"[Step 1] Loaded {len(all_results)} from checkpoint")
        return all_results

    all_results = []

    gsm8k_path = RAW_DIR / "gsm8k_full.json"
    if gsm8k_path.exists():
        gsm8k = json.load(open(gsm8k_path))
        if args.max_examples:
            gsm8k = gsm8k[: args.max_examples]
        all_results.extend(generate_cot_for_source(model, tokenizer, gsm8k, "gsm8k",
                                                   args.batch_size, is_vllm=is_vllm))
    else:
        logging.warning(f"GSM8K raw file not found: {gsm8k_path}")

    mmlu_path = RAW_DIR / "mmlu_full.json"
    if mmlu_path.exists():
        mmlu = json.load(open(mmlu_path))
        cap = args.max_examples or 5000
        if len(mmlu) > cap:
            mmlu = mmlu[:cap]
            logging.info(f"Capped MMLU to {cap}")
        all_results.extend(generate_cot_for_source(model, tokenizer, mmlu, "mmlu",
                                                   args.batch_size, is_vllm=is_vllm))

    bbh_path = RAW_DIR / "bbh_full.json"
    if not bbh_path.exists():
        bbh = download_bbh(args.max_examples)
        json.dump(bbh, open(bbh_path, "w"), indent=2)
    else:
        bbh = json.load(open(bbh_path))
        if args.max_examples:
            bbh = bbh[: args.max_examples]
    all_results.extend(generate_cot_for_source(model, tokenizer, bbh, "bbh",
                                               args.batch_size, is_vllm=is_vllm))

    json.dump(all_results, open(checkpoint_path, "w"), indent=2)
    logging.info(f"[Step 1] Saved {len(all_results)} CoT responses to {checkpoint_path}")
    return all_results


# ---- step 2: perturb & continue ----

def _create_continuation_prefix_deepseek(example: dict, perturbation_point: str):
    """
    Build the perturbed prefix for DeepSeek-R1 continuation.

    DeepSeek-R1 emits TWO portions in its response:
      1. **Pre-`</think>` internal reasoning** — free-form natural language
         ("Okay, so I need to figure out..."). NOT numbered, hard to parse
         into discrete steps cleanly.
      2. **Post-`</think>` structured presentation** — numbered formal
         answer ("1. **Regular Pay**: ... 2. **Overtime**: ..."). This IS
         the model's "reasoning chain" in a perturbable form.

    We therefore perturb the **post-think structured steps**, but keep the
    original internal reasoning intact in the prompt prefix. The model sees:

        <user>question<|Assistant|><think>\\n  ← from chat template
        [original internal reasoning verbatim]
        </think>\\n
        [steps so far with one perturbed]   ← model continues from here

    The model then continues writing the formal presentation. We measure
    whether it follows the perturbed step (TYPE_C) or self-corrects
    (TYPE_B) or bypasses (TYPE_A).

    Sanitization: replace any literal `</think>` in the perturbed step text
    with a placeholder, since the DeepSeek chat-template Jinja contains
    `content = content.split('</think>')[-1]` which would silently truncate
    upstream reasoning.
    """
    think, post = split_think_and_answer(example["cot_response"])
    if not post:
        # No structured post-think answer (model still thinking, truncated,
        # or skipped post-think). Skip — we can't apply a step-level
        # perturbation on the free-form pre-think text.
        return None
    steps = parse_cot_steps_inner(post)
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
    post_prefix = "\n".join(prefix_steps)
    # Defend against the template's `content.split('</think>')[-1]` clause.
    post_prefix = post_prefix.replace(_THINK_CLOSE, "[think-close-stripped]")
    # Build full prefix: think block (if any) + post-think prefix.
    # `_render_continuation_input_deepseek` will append this to the chat
    # template's auto-emitted `<think>\\n` opening.
    # Always close the think block: the chat template auto-opens `<think>\\n`,
    # so we need to close it before the post-think structured answer starts.
    if think:
        safe_think = think.replace(_THINK_CLOSE, "[think-close-stripped]")
        prefix = f"{safe_think}\n{_THINK_CLOSE}\n\n{post_prefix}"
    else:
        # Empty think (rare): immediately close the auto-opened <think> tag.
        prefix = f"{_THINK_CLOSE}\n\n{post_prefix}"

    original_post_prefix = "\n".join(steps[: target_idx + 1])
    original_prefix = (
        f"{think}\n{_THINK_CLOSE}\n\n{original_post_prefix}" if think
        else f"{_THINK_CLOSE}\n\n{original_post_prefix}")

    return {
        "prefix": prefix,
        "original_prefix": original_prefix,
        "perturbation_description": f"{desc} at step {target_idx}/{len(steps)} ({perturbation_point})",
        "perturbation_strategy": strategy,
        "perturbation_point": perturbation_point,
        "target_step_idx": target_idx,
        "total_steps": len(steps),
        "original_step": steps[target_idx],
        "perturbed_step": perturbed_step,
        "original_remaining": "\n".join(steps[target_idx + 1:]),
    }


def _render_continuation_input_deepseek(prompt: str, inner_prefix: str, tokenizer) -> str:
    """
    Canonical render path for DeepSeek-R1 continuation inputs. Used by
    BOTH step 2 (generation) and step 4 (feature extraction) so the
    boundary token positions align.

    Strategy: apply the chat template with the user message only and
    `add_generation_prompt=True`. The DeepSeek-R1 template's
    add-generation block ends with `<｜Assistant｜><think>\\n` — i.e., the
    opening `<think>` is emitted *by the template itself*. We then
    append the inner reasoning prefix verbatim (no `<think>` tag in it).

    If a future template revision drops the auto-`<think>` opening, we
    inject it ourselves as a fallback. This single check costs ~µs per
    call and prevents silent prompt corruption on template updates.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    # Belt-and-braces: ensure <think> is opened at the end of the rendered
    # template, since the canonical R1 template auto-adds it but variants
    # may not.
    rstripped = text.rstrip()
    if not (rstripped.endswith("<think>") or rstripped.endswith(_THINK_OPEN)):
        if not text.endswith("\n"):
            text += "\n"
        text += _THINK_OPEN + "\n"
    elif not text.endswith("\n"):
        text += "\n"
    return text + inner_prefix


# Backwards-compatible alias for any call site still referring to the old
# function name (kept until next refactor pass).
def _build_continuation_prompt_deepseek(prompt: str, prefix: str, tokenizer) -> str:
    return _render_continuation_input_deepseek(prompt, prefix, tokenizer)


def _estimate_max_tokens_deepseek(item: dict) -> int:
    """
    `<think>` continuations need more headroom than IT baselines: we have
    to (a) complete the reasoning, (b) emit `</think>`, and (c) produce
    the final answer. Budget scales with remaining-step fraction and caps
    at CONT_MAX_NEW_TOKENS_CAP.
    """
    total_steps = item.get("total_steps", 6)
    target_idx = item.get("target_step_idx", total_steps // 2)
    remaining = (total_steps - target_idx) / max(total_steps, 1)
    est = int(CONT_MAX_NEW_TOKENS_BASE * remaining * 1.5) + 128  # +128 for the post-think answer
    return max(192, min(est, CONT_MAX_NEW_TOKENS_CAP))


def step2_create_perturbations(model, tokenizer, faithful_examples, args, is_vllm=False):
    checkpoint_path = MODEL_DIR / "step2_pairs.json"
    if checkpoint_path.exists() and not args.force:
        pairs = json.load(open(checkpoint_path))
        logging.info(f"[Step 2] Loaded {len(pairs)} from checkpoint")
        return pairs

    perturbation_points = ["early", "middle", "late"]
    pending = []
    skipped_no_think = 0
    skipped_short = 0
    for ex in faithful_examples:
        if not ex.get("think_content"):
            skipped_no_think += 1
            continue
        for point in perturbation_points:
            r = _create_continuation_prefix_deepseek(ex, point)
            if r is None:
                skipped_short += 1
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

    logging.info(f"[Step 2] {len(pending)} perturbation jobs; "
                 f"skipped {skipped_no_think} for missing <think>, "
                 f"~{skipped_short // 3} baselines for <4 think-steps")

    # Render the continuation inputs ONCE (chat template + <think> injection)
    # so both vLLM and HF paths share the same boundary tokens.
    input_texts = [
        _render_continuation_input_deepseek(item["prompt"], item["prefix"], tokenizer)
        for item in pending
    ]
    # vLLM doesn't accept per-prompt max_tokens via SamplingParams; use the
    # global cap so no example is silently truncated. (Slightly more compute
    # but eliminates a fiddly per-request setting.)
    max_tok = max((_estimate_max_tokens_deepseek(item) for item in pending),
                  default=CONT_MAX_NEW_TOKENS_CAP)

    t0 = time.time()
    all_continuations = _generate_batched(
        model, tokenizer, input_texts, max_tok, args.batch_size,
        desc="Continuing", is_vllm=is_vllm, pre_rendered=True)
    logging.info(f"[Step 2] Continuation done in {time.time()-t0:.0f}s "
                 f"({len(pending)/max(time.time()-t0,1):.1f} ex/s)")

    pairs = []
    for item, cont in zip(pending, all_continuations):
        cont = cont or ""
        # Re-stitch the full response: the prefix (with open <think>) + the
        # model's continuation (which should close </think> and emit the
        # answer). This lets the answer extractor find the post-think slab.
        full_response = item["prefix"] + "\n" + cont
        if item["source"] == "gsm8k":
            ext = extract_deepseek_gsm8k_answer(cont)
        elif item["source"] == "mmlu":
            ext = extract_deepseek_mmlu_answer(cont)
        else:
            ext = extract_deepseek_bbh_answer(cont)

        ok = check_deepseek_correct(ext, item)
        if ext is None:
            label = "unclear"
        elif ok:
            label = "self_corrects"
        else:
            label = "propagates_error"
        binary_label = {"self_corrects": "unfaithful",
                        "propagates_error": "faithful"}.get(label, "unclear")

        # Did the model emit `</think>`? Useful diagnostic — a truncated
        # `<think>` should not be treated as a clean self-correction.
        closed_think = _THINK_CLOSE in cont

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
            "closed_think": closed_think,
        })

    json.dump(pairs, open(checkpoint_path, "w"), indent=2)
    logging.info(f"[Step 2] Saved {len(pairs)} pairs to {checkpoint_path}")
    return pairs


# ---- step 3: rule-based subclassify (reuses script 14's classify_pair) ----

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


# ---- step 4: hidden state extraction at the perturbed-prefix boundary ----

def step4_extract_features(model, tokenizer, pairs, args):
    """
    Extract hidden states at the LAST token of the perturbed prefix
    (i.e., the token where the perturbed step ends and the model's
    continuation would begin) across all layers. This is the same anchor
    point as script 14 — the absence of the `</think>` close tag means
    the model's hidden state still "sees" itself as mid-reasoning.
    """
    import torch
    from tqdm import tqdm

    features_path = MODEL_DIR / "deepseek_features.npz"
    if features_path.exists() and not args.force:
        logging.info(f"[Step 4] Loading features from {features_path}")
        return dict(np.load(features_path, allow_pickle=True))

    valid_pairs = [p for p in pairs if p.get("label_3class") in
                   ("silent_bypass", "self_correction", "error_propagation")]
    logging.info(f"[Step 4] {len(valid_pairs)} valid pairs (from {len(pairs)} total)")
    if not valid_pairs:
        return {}

    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    logging.info(f"n_layers={n_layers}, hidden_dim={hidden_dim}")

    # Pre-render all continuation inputs once (matches step 2 boundary tokens).
    texts = [_render_continuation_input_deepseek(p["prompt"], p["prefix"], tokenizer)
             for p in valid_pairs]
    # Sort by length so each batch has minimal padding waste.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    bsz = args.batch_size

    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"  # last real token aligned at position -1 across batch

    feats = [None] * len(valid_pairs)  # populated in original order
    for bs in tqdm(range(0, len(order), bsz), desc="Extract features (batched)"):
        idxs = order[bs: bs + bsz]
        batch_texts = [texts[i] for i in idxs]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True,
                           truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        # Left-padded: last real token is always at position -1 for every row.
        per_layer = []
        for layer_idx in range(n_layers):
            h = outputs.hidden_states[layer_idx + 1]  # (B, S, D)
            per_layer.append(h[:, -1, :].float().cpu().numpy())  # (B, D)
        layer_stack = np.stack(per_layer, axis=1)  # (B, n_layers, D)
        for j, orig_i in enumerate(idxs):
            feats[orig_i] = layer_stack[j]
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tokenizer.padding_side = old_pad
    features = np.stack(feats)  # (n_pairs, n_layers, hidden_dim)
    save_dict = {
        "pair_ids": np.array([p["id"] for p in valid_pairs]),
        "perturbation_points": np.array([p.get("perturbation_point", "") for p in valid_pairs]),
        "labels": np.array([p.get("label_3class", "unclear") for p in valid_pairs]),
        "sources": np.array([p["source"] for p in valid_pairs]),
    }
    for layer_idx in range(n_layers):
        save_dict[f"layer_{layer_idx}"] = features[:, layer_idx, :]
    np.savez_compressed(features_path, **save_dict)
    logging.info(f"[Step 4] Saved features to {features_path}")
    return save_dict


# ---- main ----

def main():
    parser = argparse.ArgumentParser(
        description=f"{MODEL_ID} cross-model validation (item 3.1)")
    parser.add_argument("--max-examples", type=int, default=None,
                        help="Per-dataset cap for smoke runs")
    parser.add_argument("--batch-size", type=int, default=24,
                        help="HF generation batch size. Ignored when vLLM is used "
                             "(vLLM batches internally). Default 24 fits a 7B distill "
                             "on H200 141GB comfortably; bump if VRAM permits.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-vllm", action="store_true",
                        help="Disable the vLLM fast-path (5–10× speedup on long "
                             "<think> traces). Use only for debugging.")
    parser.add_argument("--skip-to", type=str, default=None,
                        choices=["step2", "step3", "step4", "step5"])
    parser.add_argument("--force", action="store_true",
                        help="re-run even if a checkpoint exists")
    args = parser.parse_args()

    t_start = time.time()
    configure_h100()

    logging.info("=" * 64)
    logging.info(f"DEEPSEEK-R1-DISTILL CROSS-MODEL PIPELINE — {MODEL_ID}")
    logging.info("=" * 64)
    logging.info(f"Max examples: {args.max_examples or 'all'}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Skip to: {args.skip_to or 'none'}")
    logging.info(f"Output dir: {MODEL_DIR}")

    model = None
    tokenizer = None
    is_vllm = False
    # We need a model for steps 1, 2, and 4. Steps 1+2 can use vLLM
    # (faster); step 4 needs HF (hidden states via forward hook). We load
    # vLLM up front if step 1 or step 2 will run, then swap to HF before
    # step 4 if needed.
    need_generation = args.skip_to not in ("step3", "step4", "step5")
    need_hf_for_step4 = args.skip_to not in ("step5",)
    if need_generation:
        model, tokenizer, is_vllm = load_model_for_generation(args.device,
                                                              no_vllm=args.no_vllm)
    elif need_hf_for_step4:
        # Going straight to step 4 — load HF directly.
        tokenizer, cache_kwargs = _load_tokenizer()
        model, tokenizer, is_vllm = load_hf_model_for_generation(
            args.device, tokenizer, cache_kwargs)

    # Step 1
    if args.skip_to not in ("step2", "step3", "step4", "step5"):
        all_cot = step1_generate_cots(model, tokenizer, args, is_vllm=is_vllm)
    else:
        all_cot = json.load(open(MODEL_DIR / "step1_cot_responses.json"))
        logging.info(f"Loaded {len(all_cot)} CoT responses from checkpoint")

    faithful = [r for r in all_cot if r["is_correct"]]
    logging.info(f"Faithful examples: {len(faithful)} / {len(all_cot)}")
    for src in ["gsm8k", "mmlu", "bbh"]:
        n = sum(1 for r in all_cot if r["source"] == src)
        c = sum(1 for r in all_cot if r["source"] == src and r["is_correct"])
        if n:
            logging.info(f"  {src}: {c}/{n} ({c/n*100:.1f}%)")

    # Diagnostic: <think>-block usage rate
    with_think = sum(1 for r in all_cot if r.get("think_content"))
    logging.info(f"<think>-block emitted: {with_think}/{len(all_cot)} "
                 f"({with_think/max(len(all_cot),1)*100:.1f}%)")

    # Step 2
    if args.skip_to not in ("step3", "step4", "step5"):
        pairs = step2_create_perturbations(model, tokenizer, faithful, args,
                                            is_vllm=is_vllm)
    else:
        pairs = json.load(open(MODEL_DIR / "step2_pairs.json"))
        logging.info(f"Loaded {len(pairs)} pairs from checkpoint")

    # Step 3 (CPU-only)
    if args.skip_to not in ("step4", "step5"):
        pairs = step3_subclassify(pairs, args)
    else:
        pairs = json.load(open(MODEL_DIR / "step3_subclassified.json"))

    logging.info(f"Classification: {Counter(p.get('subtype','UNKNOWN') for p in pairs)}")

    # vLLM → HF swap before step 4 (vLLM cannot expose hidden states via hooks).
    if need_hf_for_step4 and is_vllm:
        logging.info("Step 4 requires HF backend (vLLM cannot expose hidden states). "
                     "Freeing vLLM model and reloading as HuggingFace...")
        import torch, gc
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _, cache_kwargs = _load_tokenizer()  # tokenizer already loaded; we just need cache_kwargs
        model, _, is_vllm = load_hf_model_for_generation(args.device, tokenizer, cache_kwargs)

    # Step 4
    if args.skip_to not in ("step5",):
        feature_data = step4_extract_features(model, tokenizer, pairs, args)
    else:
        feature_data = dict(np.load(MODEL_DIR / "deepseek_features.npz",
                                    allow_pickle=True))

    if model is not None:
        import torch
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Freed GPU memory for probe training")

    # Step 5 — reuse script 14's probe training verbatim (same metric reporting)
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
