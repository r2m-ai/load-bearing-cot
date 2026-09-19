"""
exp_1_4_strength_axis.py — perturbation-strength axis (item 1.4, A+B).

Reviewer y8FE #5 asked for "perturbation type **and strength**" across
datasets. The original plan addressed type (exp_1_1, exp_1_3) but not
strength. This script fills the gap by sweeping numerical-perturbation
*magnitude* on:

  - GSM8K (item A): re-uses the 879 faithful baselines from the paper's
    main pipeline. Three magnitudes:
      blatant   = num * 2 + 3   (current paper default; we re-use the
                                 paper's existing numbers, no new run)
      moderate  = num * 2       (no constant offset)
      subtle    = num + 1       (smallest possible non-trivial change)
  - BBH multistep_arithmetic_two (item B): same three magnitudes on the
    ~310 viable baselines.

If C rate rises with magnitude → confirms perturbation salience matters
(competence-floor hypothesis). If C is flat across magnitudes → the
task-difficulty driver is robust to perturbation magnitude (cleaner
"difficulty drives C" story).

Reuses the paper's canonical perturbation logic where possible; we add
two new variants of `perturb_gsm8k_arithmetic` for subtle and moderate
magnitudes. Output filenames are dedicated (no clobber).

Usage:
  # generate (GPU, ~10 min on H200 with vLLM)
  python experiments/controls/perturbation_strength.py --stage generate
  # judge (API, ~3 min)
  python experiments/controls/perturbation_strength.py --stage judge \\
         --api-key $ANTHROPIC_API_KEY
"""

import argparse
import importlib.util
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "experiments" / "pipeline" / "perturbation_helpers.py").exists():
            return p
    raise RuntimeError("Could not locate repo root from " + str(start))


ROOT = _find_repo_root(Path(__file__).resolve())
SCRIPTS = ROOT / "experiments"
PROCESSED = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"exp_1_4_strength_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "google/gemma-2-9b-it"
CACHE_DIR = None  # use HF_HOME env var (don't create duplicate model cache)

IN_BASELINES = PROCESSED / "expanded_pairs.json"
BBH_TASK_MAP = PROCESSED / "bbh_task_mapping.json"
OUT_SUBCLASSIFIED = PROCESSED / "confound_strength_subclassified.json"
OUT_LABELED = PROCESSED / "confound_strength_labeled.json"

sys.path.insert(0, str(SCRIPTS / "pipeline"))
from perturbation_helpers import parse_cot_steps, extract_gsm8k_answer  # noqa: E402


def _load_module(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- three magnitude variants of arithmetic_change ----

_MARKER_RE = re.compile(r"^(\s*(?:\d+[\.\):]|Step\s+\d+|[-*])\s+)")


def _split_marker(step: str) -> tuple[str, str]:
    m = _MARKER_RE.match(step)
    return (m.group(1), step[m.end():]) if m else ("", step)


def _perturb_number(step_text: str, transform, name: str) -> tuple[str, str, str]:
    """Find first number > 1 in step content (after stripping enumeration
    marker) and replace with transform(num). The marker is preserved
    untouched — we don't want to corrupt step indices."""
    marker, body = _split_marker(step_text)
    numbers = re.findall(r"\d+", body)
    for num_str in numbers:
        num = int(num_str)
        if num > 1:
            new_num = transform(num)
            if new_num == num:
                continue
            perturbed_body = body.replace(num_str, str(new_num), 1)
            return marker + perturbed_body, f"changed {num_str} to {new_num} ({name})", f"arithmetic_change_{name}"
    return step_text, "no_change", "none"


def perturb_subtle(step_text: str) -> tuple[str, str, str]:
    """Smallest non-trivial change: num + 1."""
    return _perturb_number(step_text, lambda n: n + 1, "subtle")


def perturb_moderate(step_text: str) -> tuple[str, str, str]:
    """Medium change: num * 2 (no constant offset; cleaner than ×2+3)."""
    return _perturb_number(step_text, lambda n: n * 2, "moderate")


def perturb_blatant(step_text: str) -> tuple[str, str, str]:
    """Current paper default: num * 2 + 3."""
    return _perturb_number(step_text, lambda n: n * 2 + 3, "blatant")


MAGNITUDE_FNS = {"subtle": perturb_subtle, "moderate": perturb_moderate, "blatant": perturb_blatant}


# ---- baseline loading ----

def load_target_baselines(max_per_group: int | None):
    """Load:
      - GSM8K baselines (all 879)
      - BBH multistep_arithmetic_two baselines (look up via bbh_task_mapping)
    Returns dict {group: [baselines]}.
    """
    if not BBH_TASK_MAP.exists():
        raise SystemExit(f"{BBH_TASK_MAP} missing — required to find multistep_arith")
    mapping = json.load(open(BBH_TASK_MAP))

    with open(IN_BASELINES) as f:
        data = json.load(f)

    # Use dedupe by question id (multiple perturbations of same baseline).
    gsm8k_seen, ma_seen = {}, {}
    for d in data:
        if d.get("source") == "gsm8k" and d["id"] not in gsm8k_seen:
            gsm8k_seen[d["id"]] = d
        elif d.get("source") == "bbh" and mapping.get(d["id"]) == "multistep_arithmetic_two":
            if d["id"] not in ma_seen:
                ma_seen[d["id"]] = d

    gsm8k = sorted(gsm8k_seen.values(), key=lambda d: d["id"])
    multistep = sorted(ma_seen.values(), key=lambda d: d["id"])
    if max_per_group:
        gsm8k = gsm8k[:max_per_group]
        multistep = multistep[:max_per_group]
    logging.info(f"Targets: gsm8k={len(gsm8k)} baselines, multistep_arith={len(multistep)} baselines")
    return {"gsm8k": gsm8k, "multistep_arith": multistep}


def build_jobs(targets: dict) -> list[dict]:
    """Apply 2 NEW magnitudes (subtle + moderate) at 3 positions to each
    baseline. Blatant is already in the main paper — we don't re-run it
    here, but cite the paper's numbers for comparison in the diagnostic."""
    jobs = []
    skipped = 0
    new_magnitudes = ["subtle", "moderate"]
    for group, baselines in targets.items():
        for d in baselines:
            cot = d.get("faithful_cot") or d.get("perturbed_cot") or ""
            steps = parse_cot_steps(cot)
            if len(steps) < 4:
                skipped += 1
                continue
            pos_map = {
                "early": max(1, len(steps) // 4),
                "middle": len(steps) // 2,
                "late": min(len(steps) - 2, 3 * len(steps) // 4),
            }
            for point, target_idx in pos_map.items():
                for mag in new_magnitudes:
                    perturbed_step, desc, strat = MAGNITUDE_FNS[mag](steps[target_idx])
                    if strat == "none":
                        continue
                    prefix_steps = steps[:target_idx] + [perturbed_step]
                    jobs.append({
                        "id": d["id"],
                        "source": d["source"],
                        "group": group,
                        "magnitude": mag,
                        "question": d.get("question", ""),
                        "reference_answer": str(d.get("reference_answer", d.get("faithful_answer", ""))),
                        "prompt": d.get("prompt", ""),
                        "faithful_cot": cot,
                        "faithful_answer": str(d.get("faithful_answer", d.get("reference_answer", ""))),
                        "perturbation_strategy": strat,
                        "perturbation_type": "numerical",
                        "perturbation_point": point,
                        "perturbation_description": f"{desc} at step {target_idx}/{len(steps)} ({point})",
                        "target_step_idx": target_idx,
                        "total_steps": len(steps),
                        "original_step": steps[target_idx],
                        "perturbed_step": perturbed_step,
                        "original_prefix": "\n".join(steps[: target_idx + 1]),
                        "prefix": "\n".join(prefix_steps),
                        "correct_letter": d.get("correct_letter", ""),  # gsm8k won't have; ok
                    })
    logging.info(f"Built {len(jobs)} jobs ({skipped} baselines skipped for <4 steps)")
    return jobs


# ---- model loading + generation (vLLM with monkey-patch for Gemma) ----

def build_continuation_prompt(prompt: str, prefix: str, tokenizer) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": prefix},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False)
    for suffix in ["<end_of_turn>\n", "<end_of_turn>", "<|eot_id|>\n", "<|eot_id|>"]:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return text


def estimate_max_tokens(total_steps: int, target_idx: int) -> int:
    remaining_fraction = (total_steps - target_idx) / max(total_steps, 1)
    return max(64, min(int(200 * remaining_fraction * 1.5), 384))


def load_model_vllm():
    """Force vLLM V0 + Gemma tokenizer monkey-patch (same as exp_3_1 fix).
    vLLM 0.8.5 + Gemma2 fails at `rope_theta` config init — fallback to HF."""
    import os
    from transformers import AutoTokenizer
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        PreTrainedTokenizerBase.all_special_tokens_extended = property(
            lambda self: self.all_special_tokens)
    os.environ.setdefault("VLLM_USE_V1", "0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        from vllm import LLM
        logging.info(f"Loading {MODEL_ID} with vLLM V0...")
        model = LLM(model=MODEL_ID, dtype="bfloat16",
                    gpu_memory_utilization=0.90, max_model_len=2560,
                    enforce_eager=False)
        return model, tokenizer, True
    except Exception as e:
        logging.warning(f"vLLM failed ({e}); falling back to HF")
        import torch
        from transformers import AutoModelForCausalLM
        for attn in ["flash_attention_2", "sdpa", "eager"]:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                    attn_implementation=attn)
                logging.info(f"HF attention: {attn}")
                break
            except Exception:
                continue
        model.eval()
        tokenizer.padding_side = "left"
        return model, tokenizer, False


def generate_vllm(model, tokenizer, jobs):
    from vllm import SamplingParams
    texts = [build_continuation_prompt(j["prompt"], j["prefix"], tokenizer) for j in jobs]
    max_tok = max(estimate_max_tokens(j["total_steps"], j["target_step_idx"]) for j in jobs)
    out = model.generate(texts, SamplingParams(max_tokens=max_tok, temperature=0))
    out = sorted(out, key=lambda o: int(o.request_id))
    return [o.outputs[0].text.strip() for o in out]


def generate_hf(model, tokenizer, jobs, batch_size=16):
    """HF batched generation, sorted by prefix length. Used when vLLM
    can't load the model (e.g., Gemma2 + vllm 0.8.5 rope_theta bug)."""
    import torch
    from tqdm import tqdm
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["prefix"]))
    res = [""] * len(jobs)
    for bs in tqdm(range(0, len(order), batch_size), desc="continue"):
        idxs = order[bs: bs + batch_size]
        texts = [build_continuation_prompt(jobs[i]["prompt"], jobs[i]["prefix"], tokenizer) for i in idxs]
        max_tok = max(estimate_max_tokens(jobs[i]["total_steps"], jobs[i]["target_step_idx"]) for i in idxs)
        inputs = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tok,
                                      do_sample=False, use_cache=True,
                                      pad_token_id=tokenizer.pad_token_id)
        for k, i in enumerate(idxs):
            plen = inputs["attention_mask"][k].sum().item()
            res[i] = tokenizer.decode(outputs[k][plen:], skip_special_tokens=True).strip()
    return res


# ---- subclassify (per-source: gsm8k → classify_gsm8k; bbh → classify_mmlu) ----

def stage_generate(args):
    import re as _re
    sub2b = _load_module("pipeline/classify_behaviors.py", "subclassify_2b")

    targets = load_target_baselines(args.max_per_group)
    jobs = build_jobs(targets)
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]

    records = []
    if OUT_SUBCLASSIFIED.exists():
        try:
            ck = json.load(open(OUT_SUBCLASSIFIED))
            if ck.get("status") == "in_progress":
                records = ck["records"]
                logging.info(f"Resuming: {len(records)} done")
        except Exception:
            pass
    done = {(r["id"], r["perturbation_strategy"], r["perturbation_point"]) for r in records}
    todo = [j for j in jobs if (j["id"], j["perturbation_strategy"], j["perturbation_point"]) not in done]
    logging.info(f"{len(todo)} jobs to run ({len(records)} already done)")
    if not todo:
        logging.info("Nothing to do.")
        return

    model, tokenizer, use_vllm = load_model_vllm()
    t0 = time.time()
    if use_vllm:
        conts = generate_vllm(model, tokenizer, todo)
    else:
        logging.info(f"Generating {len(todo)} continuations via HF (batch=16, ~2-3 ex/s)...")
        conts = generate_hf(model, tokenizer, todo, batch_size=16)
    for job, cont in zip(todo, conts):
        extracted = extract_gsm8k_answer(cont) if job["source"] == "gsm8k" else None
        if extracted is None and job["source"] == "bbh":
            # BBH multistep_arith answers are numeric — same regex works
            extracted = extract_gsm8k_answer(cont)

        ref = job["reference_answer"]
        try:
            correct = extracted is not None and float(extracted) == float(str(ref).replace(",", ""))
        except (ValueError, TypeError):
            correct = False

        if extracted is None:
            behavior = "unclear"
        elif correct:
            behavior = "self_corrects"
        else:
            behavior = "propagates_error"

        rec = {**{k: job[k] for k in (
            "id", "source", "group", "magnitude", "question", "reference_answer",
            "prompt", "faithful_cot", "faithful_answer", "perturbation_strategy",
            "perturbation_type", "perturbation_point", "perturbation_description",
            "target_step_idx", "total_steps", "original_step", "perturbed_step",
            "original_prefix", "prefix", "correct_letter",
        )},
            "perturbed_cot": job["prefix"] + "\n" + cont,
            "continuation": cont,
            "perturbed_answer": extracted,
            "perturbed_completion": cont,
            "model_ignores_perturbation": correct,
            "behavior": behavior,
            "label": {"self_corrects": "unfaithful",
                      "propagates_error": "faithful"}.get(behavior, "unclear"),
        }
        try:
            if job["source"] == "gsm8k":
                subtype, reason = sub2b.classify_gsm8k(rec)
            else:
                subtype, reason = sub2b.classify_mmlu(rec)
        except _re.error as e:
            subtype, reason = "NEEDS_JUDGE", f"rule_classifier_regex_error: {e}"
        rec["subtype"] = subtype
        rec["subtype_reason"] = reason
        rec["label_3class"] = {
            "TYPE_A": "silent_bypass", "TYPE_B": "self_correction",
            "TYPE_C": "error_propagation",
        }.get(subtype, "unclear")
        records.append(rec)

    elapsed = time.time() - t0
    logging.info(f"Generated {len(todo)} continuations in {elapsed:.0f}s "
                 f"({len(todo)/max(elapsed,1):.1f} ex/s)")
    json.dump({"timestamp": datetime.now().isoformat(), "n": len(records),
               "status": "complete", "records": records},
              open(OUT_SUBCLASSIFIED, "w"), indent=2)
    logging.info(f"Wrote {len(records)} records -> {OUT_SUBCLASSIFIED}")
    _print_pre_judge_summary(records)


def _print_pre_judge_summary(records):
    print("\nPre-judge rule-based subtype by (group × magnitude):")
    by = {}
    for r in records:
        by.setdefault((r["group"], r["magnitude"]), Counter())[r["subtype"]] += 1
    for k, c in sorted(by.items()):
        tot = sum(c.values())
        if tot:
            print(f"  {k[0]:20s} mag={k[1]:9s} n={tot:4d}  "
                  f"C={c['TYPE_C']:4d} ({c['TYPE_C']/tot*100:5.1f}%)  "
                  f"A={c['TYPE_A']:4d}  B={c['TYPE_B']:4d}  "
                  f"NEEDS_JUDGE={c['NEEDS_JUDGE']:4d}  UNCLEAR={c['UNCLEAR']:4d}")


def stage_judge(args):
    j9b = _load_module("pipeline/judge_behaviors.py", "judge_9b")
    if not OUT_SUBCLASSIFIED.exists():
        raise SystemExit(f"{OUT_SUBCLASSIFIED} missing — run --stage generate first")
    blob = json.load(open(OUT_SUBCLASSIFIED))
    records = blob["records"] if isinstance(blob, dict) else blob
    needs_ab = [r for r in records if r.get("subtype") == "NEEDS_JUDGE"]
    unclear = [r for r in records if r.get("subtype") == "UNCLEAR"]
    logging.info(f"Judge: {len(needs_ab)} NEEDS_JUDGE + {len(unclear)} UNCLEAR")

    import asyncio
    judged = []
    if needs_ab:
        t0 = time.time()
        ab = asyncio.run(j9b.run_judge_async(
            needs_ab, args.api_key, j9b.JUDGE_PROMPT_AB, j9b.parse_ab_response,
            model=args.judge_model, concurrency=args.concurrency, max_tokens=5))
        logging.info(f"A/B pass: {len(ab)} in {time.time()-t0:.0f}s")
        judged += ab
    if unclear:
        def parse_extract(t):
            t = t.strip()
            return None if t.upper() == "NONE" or not t else t
        t0 = time.time()
        ex = asyncio.run(j9b.run_judge_async(
            unclear, args.api_key, j9b.JUDGE_PROMPT_EXTRACT, parse_extract,
            model=args.judge_model, concurrency=args.concurrency, max_tokens=50))
        for r in ex:
            jl = r.get("judge_label")
            if not jl:
                continue
            try:
                eg = float(str(jl).replace(",", ""))
                rg = float(str(r.get("reference_answer", "")).replace(",", ""))
                correct = abs(eg - rg) < 1e-6
            except (ValueError, TypeError):
                correct = False
            r["judge_label"] = "A" if correct else "C"
            r["is_correct"] = correct
        logging.info(f"Extract pass: {len(ex)} in {time.time()-t0:.0f}s")
        judged += ex

    # j9b.run_judge_async returns dicts with only {id, perturbation_point,
    # judge_label, ...}. Since needs_ab and unclear are filtered subsets of
    # `records` (with reference semantics) and ab/ex preserve their order,
    # we can zip and update in place. This sidesteps the
    # (id, strategy, point) key collision we saw before.
    if needs_ab:
        ab_list = judged[: len(needs_ab)]
        for orig, jud in zip(needs_ab, ab_list):
            if orig.get("id") != jud.get("id"):
                continue  # safety; shouldn't happen
            jl = jud.get("judge_label")
            if jl == "A":
                orig["subtype"], orig["label_3class"] = "TYPE_A", "silent_bypass"
            elif jl == "B":
                orig["subtype"], orig["label_3class"] = "TYPE_B", "self_correction"
            orig["judge_label"] = jl
    if unclear:
        ex_list = judged[len(needs_ab):]
        for orig, jud in zip(unclear, ex_list):
            if orig.get("id") != jud.get("id"):
                continue
            jl = jud.get("judge_label")
            if jl == "C":
                orig["subtype"], orig["label_3class"] = "TYPE_C", "error_propagation"
            elif jl == "A":
                orig["subtype"], orig["label_3class"] = "TYPE_A", "silent_bypass"
            orig["judge_label"] = jl

    json.dump(records, open(OUT_LABELED, "w"), indent=2)
    logging.info(f"Wrote {len(records)} labeled -> {OUT_LABELED}")
    _print_strength_diagnostic(records)


def _print_strength_diagnostic(records):
    print("\n" + "=" * 70)
    print("ITEM 1.4 (A+B) — PERTURBATION STRENGTH AXIS DIAGNOSTIC")
    print("=" * 70)
    print("Reference (paper, blatant magnitude = num*2+3):")
    print("  GSM8K          : C =  3.9%")
    print("  multistep_arith: C = 64.5%  (current run, blatant)")
    print("-" * 70)
    clear = [r for r in records if r.get("label_3class") in
             ("silent_bypass", "self_correction", "error_propagation")]
    by = {}
    for r in clear:
        by.setdefault((r["group"], r["magnitude"]), Counter())[r["label_3class"]] += 1
    for k in sorted(by):
        c = by[k]
        tot = sum(c.values())
        print(f"  {k[0]:20s} mag={k[1]:9s} n={tot:4d}  "
              f"A={c['silent_bypass']/tot*100:5.1f}%  "
              f"B={c['self_correction']/tot*100:5.1f}%  "
              f"C={c['error_propagation']/tot*100:5.1f}%")
    print("=" * 70)
    print("Interpretation:")
    print("  * If C(subtle) ≈ C(moderate) ≈ C(blatant) → difficulty-driven, magnitude-robust")
    print("  * If C scales with magnitude → perturbation salience matters; report")
    print("    the scale; reframes the §5.1 claim slightly toward 'engagement × difficulty'")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Perturbation strength axis (item 1.4)")
    ap.add_argument("--stage", choices=["generate", "judge", "both"], default="generate")
    ap.add_argument("--max-per-group", type=int, default=None)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--judge-model", type=str, default="claude-haiku-4-5-20251001")
    ap.add_argument("--concurrency", type=int, default=40)
    args = ap.parse_args()
    if args.stage in ("generate", "both"):
        stage_generate(args)
    if args.stage in ("judge", "both"):
        if not args.api_key:
            raise SystemExit("--api-key required")
        stage_judge(args)


if __name__ == "__main__":
    main()
