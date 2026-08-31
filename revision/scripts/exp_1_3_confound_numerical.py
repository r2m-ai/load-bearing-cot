"""
Script (revision Theme 1, items 1.3 + 1.2): Numerical perturbations on the
"hard" datasets — the other half of the perturbation x difficulty 2x2.

Reviewer DzAa #1 / y8FE #5: GSM8K used numerical perturbations, MMLU/BBH used
textual ones, so the difficulty gradient is confounded with perturbation TYPE.
exp_1_1 added the matched *textual* arm on GSM8K (easy + textual). This script
adds the matched *numerical* arm on harder tasks:

  1.3 (robust)  : numerical perturbations on BBH COMPUTATIONAL subtasks
                  (multistep_arithmetic_two, object_counting, boolean_expressions)
                  — the paper's GSM8K-like low-error-prop subtasks.
  1.2 (brittle) : arithmetic_change on numeric MMLU subjects
                  (high_school_mathematics, abstract_algebra, formal_logic,
                  college_mathematics). Gated by an explicit feasibility probe
                  (--stage probe) — only run if numeric steps are extractable.

Together with exp_1_1 this fills the 2x2:
                 numerical            textual
       easy      GSM8K (paper)        GSM8K (exp_1_1)
       hard      THIS SCRIPT          MMLU/BBH (paper)

The numerical strategies are the CANONICAL ones from the paper, loaded by path
from scripts/2_create_perturbations.py (perturb_gsm8k_arithmetic /
perturb_gsm8k_operation) so they are identical to the GSM8K rows; a boolean
operation-swap extension handles boolean_expressions (no +-*/ to swap). Same
matched pipeline as exp_1_1: chat-template continuation, the real
2b_subclassify, the real 9b judge. Output -> dedicated
data/processed/confound_numerical_*.json; expanded_pairs.json untouched.

Stages:
  probe     (CPU): report numeric-step extractability per target group; the
                   1.2 gate. No model load.
  generate  (GPU): build prefixes, continue, rule-based subclassify.
  judge     (API): LLM-judge NEEDS_JUDGE (A/B) and UNCLEAR (extract).
  both      : generate then judge.

Usage:
  python revision/scripts/exp_1_3_confound_numerical.py --stage probe
  python revision/scripts/exp_1_3_confound_numerical.py --stage generate \
        --groups bbh_computational            # add ,mmlu_numeric if probe passes
  python revision/scripts/exp_1_3_confound_numerical.py --stage judge \
        --api-key "$ANTHROPIC_API_KEY"
"""

import argparse
import importlib.util
import json
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "scripts" / "create_perturbations_v8_helpers.py").exists():
            return p
    raise RuntimeError("Could not locate repo root from " + str(start))


ROOT = _find_repo_root(Path(__file__).resolve())
SCRIPTS = ROOT / "scripts"
PROCESSED = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"exp_1_3_confound_numerical_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "google/gemma-2-9b-it"
CACHE_DIR = Path("/workspace/model_cache") if Path("/workspace").exists() else None

IN_BASELINES = PROCESSED / "expanded_pairs.json"
BBH_TASK_MAP = PROCESSED / "bbh_task_mapping.json"
OUT_SUBCLASSIFIED = PROCESSED / "confound_numerical_subclassified.json"
OUT_LABELED = PROCESSED / "confound_numerical_labeled.json"

sys.path.insert(0, str(SCRIPTS))
from create_perturbations_v8_helpers import parse_cot_steps, extract_gsm8k_answer, extract_mmlu_answer  # noqa: E402

BBH_COMPUTATIONAL = ["multistep_arithmetic_two", "object_counting", "boolean_expressions"]
MMLU_NUMERIC = ["high_school_mathematics", "abstract_algebra", "formal_logic", "college_mathematics"]


def _load_module(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- canonical numerical strategies (identical to the paper's GSM8K) ----
# perturb_gsm8k_arithmetic / perturb_gsm8k_operation come from
# scripts/2_create_perturbations.py so this arm is matched to the GSM8K rows.

_P2 = _load_module("2_create_perturbations.py", "create_perturbations_2")
perturb_gsm8k_arithmetic = _P2.perturb_gsm8k_arithmetic
perturb_gsm8k_operation = _P2.perturb_gsm8k_operation


def perturb_boolean_swap(step_text: str):
    """operation_swap archetype for boolean_expressions (no +-*/ to swap).
    Swap one True<->False or one and<->or, mirroring perturb_gsm8k_operation."""
    swaps = [(r"\bTrue\b", "False"), (r"\bFalse\b", "True"),
             (r"\band\b", "or"), (r"\bor\b", "and")]
    for pat, repl in swaps:
        if re.search(pat, step_text):
            perturbed = re.sub(pat, repl, step_text, count=1)
            tok = pat.strip("\\b")
            return perturbed, f"swapped boolean '{tok}'", "operation_swap"
    return step_text, "no_change", "none"


# Leading step marker the CoT parser keys on. We must NOT let a numerical
# perturbation corrupt the enumeration ("2." -> "7."): that is a no-op the
# model trivially ignores and would spuriously inflate bypass.
_MARKER_RE = re.compile(r"^(\s*(?:\d+[\.\):]|Step\s+\d+|[-*])\s+)")


def _split_marker(step: str) -> tuple[str, str]:
    m = _MARKER_RE.match(step)
    return (m.group(1), step[m.end():]) if m else ("", step)


def apply_numerical(step_text: str, strategy: str, is_boolean: bool):
    """Apply the canonical numerical strategy to the step *content* only
    (leading enumeration marker stripped then re-attached)."""
    marker, body = _split_marker(step_text)
    if strategy == "arithmetic_change":
        perturbed, desc, name = perturb_gsm8k_arithmetic(body)
    elif is_boolean:
        perturbed, desc, name = perturb_boolean_swap(body)
    else:
        perturbed, desc, name = perturb_gsm8k_operation(body)
    if name == "none" or desc == "no_change":
        return step_text, "no_change", "none"
    return marker + perturbed, desc, name


# strategies applicable per target group
GROUP_STRATEGIES = {
    "multistep_arithmetic_two": ["arithmetic_change", "operation_swap"],
    "object_counting": ["arithmetic_change"],
    "boolean_expressions": ["operation_swap"],          # boolean swap
    "_mmlu_numeric": ["arithmetic_change"],
}


# ---- baseline loading ----

def _bbh_taskmap():
    return json.load(open(BBH_TASK_MAP)) if BBH_TASK_MAP.exists() else {}


def load_targets(groups: list[str], max_baselines: int | None):
    with open(IN_BASELINES) as f:
        data = json.load(f)
    taskmap = _bbh_taskmap()

    seen_bbh, seen_mmlu = {}, {}
    for d in data:
        s = d.get("source")
        if s == "bbh" and d["id"] not in seen_bbh:
            seen_bbh[d["id"]] = d
        elif s == "mmlu" and d["id"] not in seen_mmlu:
            seen_mmlu[d["id"]] = d

    targets = []  # (record, group_key, subtask_or_subject, is_boolean)
    if "bbh_computational" in groups:
        for rid, rec in seen_bbh.items():
            task = taskmap.get(rid)
            if task in BBH_COMPUTATIONAL:
                targets.append((rec, task, task, task == "boolean_expressions"))
    if "mmlu_numeric" in groups:
        for rec in seen_mmlu.values():
            if rec.get("subject") in MMLU_NUMERIC:
                targets.append((rec, "_mmlu_numeric", rec.get("subject"), False))

    targets.sort(key=lambda t: t[0]["id"])
    if max_baselines:
        targets = targets[:max_baselines]
    return targets


# ---- feasibility probe (item 1.2 gate; CPU, no model) ----

def stage_probe():
    with open(IN_BASELINES) as f:
        data = json.load(f)
    taskmap = _bbh_taskmap()
    seen_bbh, seen_mmlu = {}, {}
    for d in data:
        if d.get("source") == "bbh" and d["id"] not in seen_bbh:
            seen_bbh[d["id"]] = d
        elif d.get("source") == "mmlu" and d["id"] not in seen_mmlu:
            seen_mmlu[d["id"]] = d

    def perturbable(rec, want_ops):
        steps = parse_cot_steps(rec.get("faithful_cot") or "")
        if len(steps) < 4:
            return False
        # strip enumeration markers so the gate reflects *content* numerals,
        # not the "1." / "2." step indices (honest 1.2 feasibility).
        body = [_split_marker(s)[1] for s in steps[1:]]
        has_num = any(re.search(r"\d", s) for s in body)
        has_op = any(re.search(r"[+\-*/]|\b(and|or|not|True|False)\b", s) for s in body)
        return (has_num or has_op) if want_ops else has_num

    print("\n" + "=" * 64)
    print("FEASIBILITY PROBE (item 1.2 gate)")
    print("=" * 64)
    print("BBH computational subtasks (1.3 — expected robust):")
    for task in BBH_COMPUTATIONAL:
        recs = [r for rid, r in seen_bbh.items() if taskmap.get(rid) == task]
        ok = sum(1 for r in recs if perturbable(r, want_ops=True))
        pct = ok / max(len(recs), 1) * 100
        print(f"  {task:26s}: {ok}/{len(recs)} perturbable ({pct:.0f}%)")

    print("\nMMLU numeric subjects (1.2 — brittle; gate threshold = 50%):")
    gate_pass = True
    for subj in MMLU_NUMERIC:
        recs = [r for r in seen_mmlu.values() if r.get("subject") == subj]
        ok = sum(1 for r in recs if perturbable(r, want_ops=False))
        pct = ok / max(len(recs), 1) * 100
        flag = "OK" if pct >= 50 and len(recs) >= 20 else "WEAK"
        if flag == "WEAK":
            gate_pass = False
        print(f"  {subj:26s}: {ok}/{len(recs)} with numeric step ({pct:.0f}%)  [{flag}]")
    print("-" * 64)
    if gate_pass:
        print("  1.2 GATE: PASS — include mmlu_numeric in --groups.")
    else:
        print("  1.2 GATE: PARTIAL — keep mmlu_numeric subjects that are OK only,")
        print("  and report the descope explicitly in the response letter.")
    print("=" * 64)


# ---- continuation prompt (matched to exp_1_1 / Script 2) ----

def build_continuation_prompt(prompt: str, prefix: str, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix}]
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False)
    if text.endswith("<end_of_turn>\n"):
        text = text[: -len("<end_of_turn>\n")]
    elif text.endswith("<end_of_turn>"):
        text = text[: -len("<end_of_turn>")]
    return text


def estimate_max_tokens(total_steps: int, target_idx: int) -> int:
    rf = (total_steps - target_idx) / max(total_steps, 1)
    return max(64, min(int(200 * rf * 1.5), 384))


def build_jobs(targets):
    jobs = []
    skipped = 0
    for rec, group_key, label, is_bool in targets:
        cot = rec.get("faithful_cot") or rec.get("perturbed_cot") or ""
        steps = parse_cot_steps(cot)
        if len(steps) < 4:
            skipped += 1
            continue
        pos_map = {
            "early": max(1, len(steps) // 4),
            "middle": len(steps) // 2,
            "late": min(len(steps) - 2, 3 * len(steps) // 4),
        }
        for point, tidx in pos_map.items():
            for strat in GROUP_STRATEGIES[group_key]:
                perturbed_step, desc, sname = apply_numerical(steps[tidx], strat, is_bool)
                if desc == "no_change" or sname == "none":
                    continue
                src = rec.get("source")
                jobs.append({
                    "id": rec["id"],
                    "source": src,
                    "question": rec.get("question", ""),
                    "reference_answer": str(rec.get("reference_answer", rec.get("faithful_answer", ""))),
                    "faithful_answer": str(rec.get("faithful_answer", rec.get("reference_answer", ""))),
                    "correct_letter": rec.get("correct_letter") or "",
                    "subject": rec.get("subject") or "",
                    "bbh_subtask": label if src == "bbh" else "",
                    "prompt": rec.get("prompt", ""),
                    "faithful_cot": cot,
                    "perturbation_strategy": strat,   # canonical name (pairs w/ GSM8K)
                    "perturbation_type": "numerical",
                    "perturbation_point": point,
                    "perturbation_description": f"{desc} at step {tidx}/{len(steps)} ({point})",
                    "target_step_idx": tidx,
                    "total_steps": len(steps),
                    "original_step": steps[tidx],
                    "perturbed_step": perturbed_step,
                    "original_prefix": "\n".join(steps[: tidx + 1]),
                    "prefix": "\n".join(steps[:tidx] + [perturbed_step]),
                })
    logging.info(f"Built {len(jobs)} jobs from {len(targets)} baselines ({skipped} skipped)")
    return jobs


# ---- answer extraction / behavior (matched to the paper's handling) ----

def _num(x):
    try:
        return float(str(x).replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _norm_bool(x):
    t = str(x).strip().lower()
    if t in ("true", "yes"):
        return "true"
    if t in ("false", "no"):
        return "false"
    return None


def behavior_for(job, cont):
    """Set behavior so that 2b_subclassify (classify_gsm8k for gsm8k, else
    classify_mmlu) produces matched A/B/C/NEEDS_JUDGE/UNCLEAR labels.
    Lean to 'unclear' when uncertain so the 9b judge Pass-2 decides (exactly
    how the paper labelled BBH)."""
    src = job["source"]
    sub = job.get("bbh_subtask", "")
    ref = job["reference_answer"]
    if src == "mmlu":
        ext = extract_mmlu_answer(cont)
        if ext is None:
            return "unclear", False
        ok = ext.upper() == str(job.get("correct_letter", "")).upper()
        return ("self_corrects", True) if ok else ("propagates_error", False)
    if sub == "boolean_expressions":
        m = re.findall(r"\b(True|False|Yes|No)\b", cont)
        if not m:
            return "unclear", False
        ok = _norm_bool(m[-1]) == _norm_bool(ref)
        return ("self_corrects", True) if ok else ("propagates_error", False)
    # multistep_arithmetic_two / object_counting -> numeric
    ext = extract_gsm8k_answer(cont)
    en, rn = _num(ext), _num(ref)
    if en is None or rn is None:
        return "unclear", False
    ok = abs(en - rn) < 1e-6
    return ("self_corrects", True) if ok else ("propagates_error", False)


# ---- model loading (shared shape with exp_1_1) ----

def load_model(no_vllm: bool):
    from transformers import AutoTokenizer
    ck = {"cache_dir": str(CACHE_DIR)} if CACHE_DIR else {}
    if CACHE_DIR:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, **ck)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if not no_vllm:
        try:
            from vllm import LLM
            logging.info(f"Loading {MODEL_ID} with vLLM...")
            return LLM(model=MODEL_ID, dtype="bfloat16",
                       gpu_memory_utilization=0.90, max_model_len=2560,
                       enforce_eager=False), tok, True
        except Exception as e:
            logging.warning(f"vLLM failed ({e}); using HuggingFace")
    import torch
    from transformers import AutoModelForCausalLM
    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation=attn, **ck)
            logging.info(f"HF attention: {attn}")
            break
        except Exception as e:
            logging.info(f"  {attn} unavailable: {type(e).__name__}")
    else:
        raise RuntimeError("Could not load model")
    model.eval()
    return model, tok, False


def generate_vllm(model, tok, jobs):
    from vllm import SamplingParams
    order = sorted(range(len(jobs)),
                   key=lambda i: estimate_max_tokens(jobs[i]["total_steps"], jobs[i]["target_step_idx"]))
    texts = [build_continuation_prompt(jobs[i]["prompt"], jobs[i]["prefix"], tok) for i in order]
    mt = max(estimate_max_tokens(j["total_steps"], j["target_step_idx"]) for j in jobs)
    out = sorted(model.generate(texts, SamplingParams(max_tokens=mt, temperature=0)),
                 key=lambda x: int(x.request_id))
    res = [""] * len(jobs)
    for pos, o in zip(order, out):
        res[pos] = o.outputs[0].text.strip()
    return res


def generate_hf(model, tok, jobs, bs):
    import torch
    from tqdm import tqdm
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i]["prefix"]))
    res = [""] * len(jobs)
    for b in tqdm(range(0, len(order), bs), desc="continue"):
        idxs = order[b:b + bs]
        texts = [build_continuation_prompt(jobs[i]["prompt"], jobs[i]["prefix"], tok) for i in idxs]
        mt = max(estimate_max_tokens(jobs[i]["total_steps"], jobs[i]["target_step_idx"]) for i in idxs)
        inp = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=2048).to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=mt, do_sample=False,
                                  use_cache=True, pad_token_id=tok.pad_token_id)
        for k, i in enumerate(idxs):
            plen = inp["attention_mask"][k].sum().item()
            res[i] = tok.decode(out[k][plen:], skip_special_tokens=True).strip()
    return res


# ---- generate stage ----

def stage_generate(args):
    sub2b = _load_module("2b_subclassify.py", "subclassify_2b")
    groups = [g.strip() for g in args.groups.split(",")]
    targets = load_targets(groups, args.max_baselines)
    logging.info(f"Groups={groups}  targets(baselines)={len(targets)}")
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
        logging.info("Nothing to generate.")
        return

    model, tok, use_vllm = load_model(args.no_vllm)
    t0 = time.time()
    chunk = len(todo) if use_vllm else args.batch_size * 8
    for cs in range(0, len(todo), chunk):
        batch = todo[cs:cs + chunk]
        conts = (generate_vllm(model, tok, batch) if use_vllm
                 else generate_hf(model, tok, batch, args.batch_size))
        for job, cont in zip(batch, conts):
            behavior, gives_correct = behavior_for(job, cont)
            ext = (extract_mmlu_answer(cont) if job["source"] == "mmlu"
                   else extract_gsm8k_answer(cont))
            rec = {
                **{k: job[k] for k in (
                    "id", "source", "question", "reference_answer",
                    "faithful_answer", "correct_letter", "subject",
                    "bbh_subtask", "prompt", "faithful_cot",
                    "perturbation_strategy", "perturbation_type",
                    "perturbation_point", "perturbation_description",
                    "target_step_idx", "total_steps", "original_step",
                    "perturbed_step", "original_prefix", "prefix")},
                "perturbed_cot": job["prefix"] + "\n" + cont,
                "continuation": cont,
                "perturbed_answer": ext,
                "perturbed_completion": cont,
                "model_ignores_perturbation": gives_correct,
                "behavior": behavior,
                "label": {"self_corrects": "unfaithful",
                          "propagates_error": "faithful"}.get(behavior, "unclear"),
            }
            # source decides classify_gsm8k vs classify_mmlu (matched to 2b).
            # Wrap in try/except to handle the known re.error in classify_mmlu
            # when perturbed_val is a single regex-special char (`+`, `*`,
            # `?`, `(`, etc.) — happens for operation_swap descriptions like
            # "swapped '+' to '-'". Fallback: defer to the judge.
            try:
                if rec["source"] == "gsm8k":
                    subtype, reason = sub2b.classify_gsm8k(rec)
                else:
                    subtype, reason = sub2b.classify_mmlu(rec)
            except re.error as e:
                subtype = "NEEDS_JUDGE"
                reason = f"rule_classifier_regex_error: {e}"
            rec["subtype"] = subtype
            rec["subtype_reason"] = reason
            rec["label_3class"] = {"TYPE_A": "silent_bypass",
                                   "TYPE_B": "self_correction",
                                   "TYPE_C": "error_propagation"}.get(subtype, "unclear")
            records.append(rec)

        n = len(records)
        rate = (n - len(done)) / max(time.time() - t0, 1e-6)
        c = sum(1 for r in records if r["label_3class"] == "error_propagation")
        logging.info(f"[{n}/{len(jobs)}] {rate:.1f}/s  C so far={c/max(n,1)*100:.1f}%")
        json.dump({"timestamp": datetime.now().isoformat(), "n": n,
                   "status": "in_progress", "records": records},
                  open(OUT_SUBCLASSIFIED, "w"))

    json.dump({"timestamp": datetime.now().isoformat(), "n": len(records),
               "status": "complete", "records": records},
              open(OUT_SUBCLASSIFIED, "w"), indent=2)
    logging.info(f"Wrote {len(records)} -> {OUT_SUBCLASSIFIED}")
    _summary(records)


def _summary(records):
    by = defaultdict(Counter)
    for r in records:
        key = r.get("bbh_subtask") or r.get("subject") or r["source"]
        by[key][r["subtype"]] += 1
    print("\nRule-based (pre-judge) by target group:")
    for k, c in sorted(by.items()):
        tot = sum(c.values())
        print(f"  {k:26s} n={tot:5d}  C={c['TYPE_C']}({c['TYPE_C']/max(tot,1)*100:.1f}%)  "
              f"NEEDS_JUDGE={c['NEEDS_JUDGE']}  UNCLEAR={c['UNCLEAR']}")
    print("NOTE: final A/B only after --stage judge.")


# ---- judge stage (reuses 9b verbatim; matched to exp_1_1) ----

def stage_judge(args):
    j9b = _load_module("9b_run_judge.py", "judge_9b")
    if not OUT_SUBCLASSIFIED.exists():
        raise SystemExit(f"{OUT_SUBCLASSIFIED} not found — run --stage generate first.")
    blob = json.load(open(OUT_SUBCLASSIFIED))
    pairs = blob["records"] if isinstance(blob, dict) else blob

    needs_ab = [p for p in pairs if p.get("subtype") == "NEEDS_JUDGE"]
    unclear = [p for p in pairs if p.get("subtype") == "UNCLEAR"]
    logging.info(f"Judge: {len(needs_ab)} NEEDS_JUDGE, {len(unclear)} UNCLEAR")

    import asyncio
    judged = []
    if needs_ab:
        t0 = time.time()
        judged += asyncio.run(j9b.run_judge_async(
            needs_ab, args.api_key, j9b.JUDGE_PROMPT_AB, j9b.parse_ab_response,
            model=args.judge_model, concurrency=args.concurrency, max_tokens=5))
        logging.info(f"A/B pass {len(needs_ab)} in {time.time()-t0:.0f}s")
    if unclear:
        def parse_extract(t):
            t = t.strip()
            return None if t.upper() == "NONE" or not t else t
        t0 = time.time()
        ex = asyncio.run(j9b.run_judge_async(
            unclear, args.api_key, j9b.JUDGE_PROMPT_EXTRACT, parse_extract,
            model=args.judge_model, concurrency=args.concurrency, max_tokens=50))
        ul = {(p["id"], p.get("perturbation_point", "")): p for p in unclear}
        for r in ex:
            p = ul.get((r["id"], r["perturbation_point"]))
            if p and r.get("judge_label"):
                guess, ref = r["judge_label"], str(p.get("reference_answer", ""))
                gn, rn = _num(guess), _num(ref)
                if gn is not None and rn is not None:
                    correct = abs(gn - rn) < 1e-6
                else:
                    gb, rb = _norm_bool(guess), _norm_bool(ref)
                    correct = (gb is not None and gb == rb) or \
                              guess.strip().lower() == ref.strip().lower()
                r["judge_label"] = "A" if correct else "C"
                r["is_correct"] = correct
        logging.info(f"Extract pass {len(unclear)} in {time.time()-t0:.0f}s")
        judged += ex

    jmap = {(r["id"], r["perturbation_point"]): r for r in judged}
    for p in pairs:
        r = jmap.get((p["id"], p.get("perturbation_point", "")))
        if not r:
            continue
        jl = r.get("judge_label")
        if p.get("subtype") == "NEEDS_JUDGE":
            if jl == "A":
                p["subtype"], p["label_3class"] = "TYPE_A", "silent_bypass"
            elif jl == "B":
                p["subtype"], p["label_3class"] = "TYPE_B", "self_correction"
            p["judge_label"] = jl
        elif p.get("subtype") == "UNCLEAR":
            if jl == "C":
                p["subtype"], p["label_3class"] = "TYPE_C", "error_propagation"
            elif jl == "A":
                p["subtype"], p["label_3class"] = "TYPE_A", "silent_bypass"
            p["judge_label"] = jl

    json.dump(pairs, open(OUT_LABELED, "w"), indent=2)
    logging.info(f"Wrote {len(pairs)} labeled -> {OUT_LABELED}")
    _print_numerical_hard_diagnostic(pairs)


def _print_numerical_hard_diagnostic(pairs):
    clear = [p for p in pairs if p.get("label_3class") in
             ("silent_bypass", "self_correction", "error_propagation")]
    print("\n" + "=" * 64)
    print("ITEM 1.3/1.2 — NUMERICAL PERTURBATIONS ON HARD TASKS")
    print("=" * 64)
    grp = defaultdict(Counter)
    for p in clear:
        key = p.get("bbh_subtask") or p.get("subject") or p["source"]
        grp[key][p["label_3class"]] += 1
    for k, c in sorted(grp.items()):
        tot = sum(c.values())
        if not tot:
            continue
        print(f"  {k:26s} n={tot:5d}  "
              f"A={c['silent_bypass']/tot*100:5.1f}%  "
              f"B={c['self_correction']/tot*100:5.1f}%  "
              f"C={c['error_propagation']/tot*100:5.1f}%")
    print("-" * 64)
    print("  Compare C to: GSM8K-numerical 3.9% | MMLU-textual 22.2% | BBH-textual 40.9%.")
    print("  numerical+hard staying LOW-C alongside textual+hard HIGH-C is")
    print("  evidence the gradient is perturbation-TYPE driven; numerical+hard")
    print("  rising with difficulty supports the task-difficulty reading.")
    print("  Feed all cells to the 1.5 variance decomposition before concluding.")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description="Confound: numerical perturbations on hard tasks (1.3 + 1.2)")
    ap.add_argument("--stage", choices=["probe", "generate", "judge", "both"], default="probe")
    ap.add_argument("--groups", type=str, default="bbh_computational",
                    help="comma list: bbh_computational,mmlu_numeric")
    ap.add_argument("--max-baselines", type=int, default=None)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--judge-model", type=str, default="claude-haiku-4-5-20251001")
    ap.add_argument("--concurrency", type=int, default=40)
    args = ap.parse_args()

    if args.stage == "probe":
        stage_probe()
        return
    if args.stage in ("generate", "both"):
        stage_generate(args)
    if args.stage in ("judge", "both"):
        if not args.api_key:
            raise SystemExit("--api-key required for the judge stage")
        stage_judge(args)


if __name__ == "__main__":
    main()
