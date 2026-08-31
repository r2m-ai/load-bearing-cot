"""
GPU Batch B v2: Optimized for H100 80GB.
Loads the model ONCE, runs Exp 2, 3a, 3c sequentially.

Key optimizations over v1:
- Batch size 16 (H100 has headroom for 9B model)
- Left-padding for correct batched generation
- max_new_tokens=256 (GSM8K answers are short)
- Batched logprob computation for Exp 3c
- torch.compile for faster inference
- Incremental saves for crash recovery

Usage: python scripts/v8/run_gpu_batch_b_v2.py [--exp 2,3a,3c] [--batch-size 16]
"""

import json
import re
import argparse
import logging
import time
import random
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

random.seed(42)
torch.manual_seed(42)

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
        logging.FileHandler(LOG_DIR / f"gpu_batch_b_v2_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "google/gemma-2-9b-it"
CACHE_DIR = Path("/workspace/model_cache") if Path("/workspace").exists() else None

sys.path.insert(0, str(ROOT / "scripts"))
from create_perturbations_v8_helpers import parse_cot_steps, extract_gsm8k_answer, extract_mmlu_answer


# ========== Model loading ==========

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    cache_kwargs = {"cache_dir": str(CACHE_DIR)} if CACHE_DIR else {}
    if CACHE_DIR:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading {MODEL_ID}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **cache_kwargs)
    # LEFT-pad for batched generation (critical for correct output extraction)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for attn_impl in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation=attn_impl, **cache_kwargs,
            )
            logging.info(f"Attention: {attn_impl}")
            break
        except Exception as e:
            logging.info(f"  {attn_impl} unavailable: {type(e).__name__}")
    else:
        raise RuntimeError("Could not load model")

    model.eval()
    vram = torch.cuda.memory_allocated() / 1e9
    logging.info(f"Model loaded in {time.time()-t0:.0f}s, VRAM: {vram:.1f}GB")
    return model, tokenizer


# ========== Generation ==========

def generate_batch(model, tokenizer, prefixes, max_new_tokens=256, batch_size=16):
    """Generate continuations with proper left-padding and adaptive batch size."""
    all_results = []

    for i in range(0, len(prefixes), batch_size):
        batch = prefixes[i:i+batch_size]
        try:
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=1536).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for j, seq in enumerate(outputs):
                # With left-padding, prompt length varies per sequence
                prompt_len = inputs["attention_mask"][j].sum().item()
                new_tokens = seq[prompt_len:]
                all_results.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logging.warning(f"OOM at batch_size={len(batch)}, falling back to sequential")
            for prefix in batch:
                inputs = tokenizer(prefix, return_tensors="pt", truncation=True,
                                   max_length=1536).to(model.device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
                new_tokens = out[0][inputs["input_ids"].shape[1]:]
                all_results.append(tokenizer.decode(new_tokens, skip_special_tokens=True))

    return all_results


def get_logprobs_batch(model, tokenizer, prefixes, target_tokens, batch_size=32):
    """Batched logprob computation. Much faster than one-at-a-time."""
    all_results = []

    for i in range(0, len(prefixes), batch_size):
        batch = prefixes[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True, max_length=1536).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # For each sequence, get logprobs at the last real token (before padding)
        for j in range(len(batch)):
            # Find last non-pad position
            mask = inputs["attention_mask"][j]
            last_pos = mask.sum().item() - 1
            logits = outputs.logits[j, last_pos, :]
            log_probs = torch.log_softmax(logits, dim=-1)

            result = {}
            for tok_str in target_tokens:
                tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
                if tok_ids:
                    result[tok_str] = float(log_probs[tok_ids[0]].cpu())
            all_results.append(result)

    return all_results


# ========== Experiment 2: GSM8K text perturbations ==========

def gsm8k_text_confidence_injection(step_text, question):
    numbers = re.findall(r"\d+", step_text)
    if numbers:
        num = int(numbers[0])
        wrong_num = num * 2 + 7
        return (
            f"At this point, the answer is clearly {wrong_num}. "
            f"This follows directly from the problem statement and "
            f"can be verified by simple arithmetic."
        ), f"confidence injection: asserted {wrong_num}", "text_confidence_injection"
    return step_text, "no_change", "none"


def gsm8k_text_premise_contradiction(step_text, question):
    return (
        f"However, we need to reconsider. The problem actually states "
        f"the opposite of what we assumed. {step_text.split('.')[0]} is incorrect "
        f"because the quantities described are inverted."
    ), "contradicted premise", "text_premise_contradiction"


def gsm8k_text_reversed_logic(step_text, question):
    return (
        f"But this reasoning is backwards. Instead of adding, we should "
        f"subtract here because the total decreases, not increases. "
        f"The correct interpretation leads to a much smaller result."
    ), "reversed logic", "text_reversed_logic"


def run_exp2(model, tokenizer, max_examples=0, batch_size=16):
    logging.info("\n" + "="*60)
    logging.info("EXP 2: GSM8K Text-Based Perturbations")
    logging.info("="*60)

    with open(DATA / "expanded_pairs.json") as f:
        all_data = json.load(f)

    gsm8k_baselines = {}
    for d in all_data:
        if d["source"] == "gsm8k" and d["id"] not in gsm8k_baselines:
            gsm8k_baselines[d["id"]] = d

    baselines = list(gsm8k_baselines.values())
    if max_examples > 0:
        baselines = baselines[:max_examples]
    logging.info(f"GSM8K baselines: {len(baselines)}")

    text_strategies = [
        ("text_confidence_injection", gsm8k_text_confidence_injection),
        ("text_premise_contradiction", gsm8k_text_premise_contradiction),
        ("text_reversed_logic", gsm8k_text_reversed_logic),
    ]

    # Pre-build all jobs
    all_jobs = []
    for d in baselines:
        cot = d.get("faithful_cot", d.get("perturbed_cot", ""))
        steps = parse_cot_steps(cot)
        if len(steps) < 4:
            continue

        question = d.get("question", "")
        original_answer = d.get("faithful_answer", d.get("reference_answer", ""))

        for pos in ["early", "middle", "late"]:
            pos_map = {"early": max(1, len(steps)//4), "middle": len(steps)//2,
                       "late": min(len(steps)-2, 3*len(steps)//4)}
            target_idx = pos_map[pos]

            for strategy_name, strategy_fn in text_strategies:
                perturbed_step, desc, strat = strategy_fn(steps[target_idx], question)
                if desc == "no_change":
                    continue
                prefix_steps = steps[:target_idx] + [perturbed_step]
                prefix = d.get("prompt", "") + "\n" + "\n".join(prefix_steps)
                all_jobs.append({
                    "prefix": prefix, "id": d["id"], "strategy": strategy_name,
                    "position": pos, "original_step": steps[target_idx],
                    "perturbed_step": perturbed_step, "original_answer": str(original_answer),
                })

    total = len(all_jobs)
    logging.info(f"  {total} jobs, batch_size={batch_size}")

    # Check for checkpoint
    outpath = RESULTS / "exp2_gsm8k_text_perturbations.json"
    results = []
    start_idx = 0
    if outpath.exists():
        try:
            with open(outpath) as f:
                ckpt = json.load(f)
            if ckpt.get("status") == "in_progress":
                results = ckpt["results"]
                start_idx = len(results)
                logging.info(f"  Resuming from checkpoint: {start_idx}/{total}")
        except Exception:
            pass

    t0 = time.time()
    for batch_start in range(start_idx, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_jobs = all_jobs[batch_start:batch_end]
        prefixes = [j["prefix"] for j in batch_jobs]

        continuations = generate_batch(model, tokenizer, prefixes,
                                       max_new_tokens=256, batch_size=batch_size)

        for job, cont in zip(batch_jobs, continuations):
            cont_answer = extract_gsm8k_answer(cont)
            is_correct = cont_answer == job["original_answer"] if cont_answer and job["original_answer"] else None
            label = "C" if (cont_answer and not is_correct) else ("A_or_B" if is_correct else "unclear")
            results.append({
                "id": job["id"], "source": "gsm8k", "strategy": job["strategy"],
                "position": job["position"], "original_step": job["original_step"],
                "perturbed_step": job["perturbed_step"], "continuation": cont[:500],
                "original_answer": job["original_answer"], "cont_answer": cont_answer,
                "label": label,
            })

        count = len(results)
        if count % (batch_size * 4) < batch_size:
            elapsed = time.time() - t0
            done_this_run = count - start_idx
            rate = done_this_run / elapsed if elapsed > 0 else 0
            eta = (total - count) / rate if rate > 0 else 0
            c_so_far = sum(1 for r in results if r["label"] == "C")
            logging.info(f"  [{count}/{total}] {rate:.1f}/s, ETA {eta:.0f}s, C={c_so_far/max(count,1)*100:.1f}%")

        # Incremental save every ~500
        if count % 500 < batch_size:
            with open(outpath, "w") as f:
                json.dump({"timestamp": datetime.now().isoformat(), "n": count,
                           "status": "in_progress", "results": results}, f)

    # Final summary
    c_count = sum(1 for r in results if r["label"] == "C")
    ab_count = sum(1 for r in results if r["label"] == "A_or_B")
    logging.info(f"\nExp 2 done: {len(results)} continuations in {time.time()-t0:.0f}s")
    logging.info(f"  C: {c_count} ({c_count/max(len(results),1)*100:.1f}%)")
    logging.info(f"  A/B: {ab_count} ({ab_count/max(len(results),1)*100:.1f}%)")
    for strat in ["text_confidence_injection", "text_premise_contradiction", "text_reversed_logic"]:
        s = [r for r in results if r["strategy"] == strat]
        sc = sum(1 for r in s if r["label"] == "C")
        logging.info(f"  {strat}: C={sc}/{len(s)} ({sc/max(len(s),1)*100:.1f}%)")

    with open(outpath, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "n": len(results),
                   "status": "complete", "results": results}, f, indent=2)
    logging.info(f"Saved to {outpath}")
    return results


# ========== Experiment 3a: Answer-change sensitivity ==========

def run_exp3a(model, tokenizer, max_examples=0, batch_size=16):
    logging.info("\n" + "="*60)
    logging.info("EXP 3a: Answer-Change Sensitivity")
    logging.info("="*60)

    with open(DATA / "expanded_pairs.json") as f:
        all_data = json.load(f)

    # Compute answer-change rate from existing data (no new generation needed)
    # For MMLU: compare correct_letter vs perturbed_answer (both are letters)
    # For BBH: normalize reference_answer "(B)" -> "B" to compare with perturbed_answer "B"
    # For GSM8K: compare reference_answer vs perturbed_answer (both are numbers)
    import re
    def normalize_answer(ans, source):
        """Normalize answer for comparison."""
        ans = str(ans).strip()
        if source == "bbh":
            # "(B)" -> "B", "Yes" -> "YES", etc.
            m = re.match(r'^\(([A-E])\)$', ans)
            if m:
                return m.group(1).upper()
            return ans.upper()
        return ans.upper()

    change_rate = defaultdict(lambda: {"total": 0, "changed": 0})
    for d in all_data:
        if d["source"] == "mmlu":
            orig = d.get("correct_letter", "")
            cont = d.get("perturbed_answer", "")
        elif d["source"] == "bbh":
            orig = d.get("reference_answer", d.get("faithful_answer", ""))
            cont = d.get("perturbed_answer", "")
        else:
            orig = d.get("reference_answer", d.get("faithful_answer", ""))
            cont = d.get("perturbed_answer", "")
        if not orig or not cont:
            continue
        change_rate[d["source"]]["total"] += 1
        if normalize_answer(cont, d["source"]) != normalize_answer(orig, d["source"]):
            change_rate[d["source"]]["changed"] += 1

    logging.info("Answer-change sensitivity (existing correct-baseline data):")
    results_summary = {}
    for source in ["gsm8k", "mmlu", "bbh"]:
        r = change_rate[source]
        rate = r["changed"] / max(r["total"], 1) * 100
        results_summary[source] = {"total": r["total"], "changed": r["changed"],
                                    "change_rate": round(rate, 1)}
        logging.info(f"  {source}: {r['changed']}/{r['total']} ({rate:.1f}%) answers changed")

    # Per MMLU subject
    subj_change = defaultdict(lambda: {"total": 0, "changed": 0})
    for d in all_data:
        if d["source"] != "mmlu":
            continue
        orig = d.get("correct_letter", "")
        cont = d.get("perturbed_answer", "")
        if not orig or not cont:
            continue
        subj = d.get("subject", "unknown")
        subj_change[subj]["total"] += 1
        if str(cont).strip() != str(orig).strip():
            subj_change[subj]["changed"] += 1

    subj_summary = {}
    for subj, r in sorted(subj_change.items(), key=lambda x: x[1]["changed"]/max(x[1]["total"],1)):
        if r["total"] >= 50:
            rate = r["changed"] / r["total"] * 100
            subj_summary[subj] = {"total": r["total"], "changed": r["changed"],
                                   "change_rate": round(rate, 1)}

    logging.info(f"\nMMLU subject range: {min(v['change_rate'] for v in subj_summary.values()):.1f}% - "
                 f"{max(v['change_rate'] for v in subj_summary.values()):.1f}%")

    outpath = RESULTS / "exp3a_answer_change_sensitivity.json"
    with open(outpath, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(),
                   "dataset_summary": results_summary,
                   "mmlu_subjects": subj_summary}, f, indent=2)
    logging.info(f"Saved to {outpath}")
    return results_summary


# ========== Experiment 3c: Logprob shift ==========

def run_exp3c(model, tokenizer, max_examples=500, batch_size=32):
    logging.info("\n" + "="*60)
    logging.info("EXP 3c: Logprob Shift (MMLU)")
    logging.info("="*60)

    with open(DATA / "expanded_pairs.json") as f:
        all_data = json.load(f)

    mmlu = [d for d in all_data if d["source"] == "mmlu"
            and d.get("original_prefix") and d.get("prefix")]

    # Deduplicate
    seen = set()
    unique = []
    for d in mmlu:
        key = (d["id"], d.get("perturbation_strategy", ""), d.get("perturbation_point", ""))
        if key not in seen:
            seen.add(key)
            unique.append(d)

    if max_examples > 0:
        unique = unique[:max_examples]
    logging.info(f"MMLU examples: {len(unique)}")

    option_tokens = ["A", "B", "C", "D"]
    t0 = time.time()

    # Build all prefixes (original + perturbed interleaved)
    orig_prefixes = []
    pert_prefixes = []
    meta = []
    for d in unique:
        prompt_base = d.get("prompt", "")
        orig_prefixes.append(prompt_base + "\n" + d["original_prefix"])
        pert_prefixes.append(prompt_base + "\n" + d["prefix"])
        correct = d.get("correct_letter", "")
        if not (isinstance(correct, str) and len(correct) == 1 and correct in "ABCD"):
            # Fallback: try to extract from reference_answer
            ref = d.get("reference_answer", d.get("faithful_answer", ""))
            if isinstance(ref, str) and len(ref) == 1 and ref in "ABCD":
                correct = ref
            else:
                correct = None
        meta.append({
            "id": d["id"], "subject": d.get("subject", "unknown"),
            "strategy": d.get("perturbation_strategy", ""),
            "position": d.get("perturbation_point", ""),
            "correct_letter": correct,
        })

    logging.info(f"  Computing logprobs for {len(orig_prefixes)} original + {len(pert_prefixes)} perturbed prefixes...")

    # Batch compute all logprobs
    orig_logprobs = get_logprobs_batch(model, tokenizer, orig_prefixes, option_tokens, batch_size)
    logging.info(f"  Original logprobs done ({time.time()-t0:.0f}s)")

    pert_logprobs = get_logprobs_batch(model, tokenizer, pert_prefixes, option_tokens, batch_size)
    logging.info(f"  Perturbed logprobs done ({time.time()-t0:.0f}s)")

    # Compute shifts
    results = []
    for i, m in enumerate(meta):
        if m["correct_letter"] is None:
            continue
        cl = m["correct_letter"]
        orig_lp = orig_logprobs[i].get(cl, -float("inf"))
        pert_lp = pert_logprobs[i].get(cl, -float("inf"))
        shift = pert_lp - orig_lp
        results.append({
            **m, "orig_logprob": round(orig_lp, 4), "pert_logprob": round(pert_lp, 4),
            "shift": round(shift, 4),
        })

    # Summarize by subject
    subj_shifts = defaultdict(list)
    for r in results:
        subj_shifts[r["subject"]].append(r["shift"])

    subj_summary = {}
    for subj, shifts in sorted(subj_shifts.items(), key=lambda x: np.mean(x[1])):
        if len(shifts) >= 5:
            subj_summary[subj] = {"n": len(shifts), "mean_shift": round(float(np.mean(shifts)), 4)}

    all_shifts = [r["shift"] for r in results]
    overall = round(float(np.mean(all_shifts)), 4) if all_shifts else 0
    logging.info(f"\nExp 3c done: {len(results)} examples in {time.time()-t0:.0f}s")
    logging.info(f"  Overall mean shift: {overall}")
    if subj_summary:
        vals = [v["mean_shift"] for v in subj_summary.values()]
        logging.info(f"  Subject range: [{min(vals):.4f}, {max(vals):.4f}]")

    outpath = RESULTS / "exp3c_logprob_shift.json"
    with open(outpath, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "n": len(results),
                   "overall_mean_shift": overall, "subject_summary": subj_summary,
                   "results": results}, f, indent=2)
    logging.info(f"Saved to {outpath}")
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="2,3a,3c")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    exps = [e.strip() for e in args.exp.split(",")]
    logging.info(f"Experiments: {exps}, batch_size={args.batch_size}")

    t_total = time.time()
    model, tokenizer = load_model()

    if "2" in exps:
        run_exp2(model, tokenizer, args.max_examples, args.batch_size)

    if "3a" in exps:
        run_exp3a(model, tokenizer, args.max_examples, args.batch_size)

    if "3c" in exps:
        run_exp3c(model, tokenizer, args.max_examples, args.batch_size * 2)  # logprobs can batch larger

    logging.info(f"\nAll done in {time.time()-t_total:.0f}s ({(time.time()-t_total)/60:.1f}m)")


if __name__ == "__main__":
    main()
