"""
exp_2_0_judge_persistence.py — re-run 4 judge prompt variants on every
non-Type-C example and persist (id, variant, judge_label) per row.

Why: experiments/validation/judge_sensitivity.py only aggregates A/B distributions
across variants. Method Spec M3 needs the per-example labels so the human
annotation budget (item 2.1) can be stratified by agree vs disagree across
the 4 variants — far more informative per annotation than a random sample.

Outputs:
  data/processed/judge_variants_per_example.json
      One row per (id, variant): {id, source, variant, judge_label, raw}.
  results/validation/judge_prompt_variants/
      Disagreement summary (per-source + per-strategy) and stratification
      input for the annotation step.

Reuses 4 A/B variants from experiments/validation/judge_sensitivity.py verbatim
(we drop "causal_final_answer" — it's a C-vs-NC variant, not an A/B
classifier, and 2.1 only re-labels non-C records).

Usage:
  python experiments/validation/judge_prompt_variants.py --api-key $ANTHROPIC_API_KEY
  python experiments/validation/judge_prompt_variants.py --sample 500 --api-key $KEY   # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
OUTDIR = ROOT / "results" / "validation" / "judge_prompt_variants"
OUTDIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"exp_2_0_judge_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

OUT_JSONL = DATA / "judge_variants_per_example.json"

# Four A/B variants from experiments/validation/judge_sensitivity.py.
PROMPTS = {
    "strict_bypass": """Classify how a language model handled a perturbed reasoning step.
The model saw a WRONG step but still gave the CORRECT final answer.

TYPE_A (SILENT BYPASS): The model shows NO evidence of noticing the perturbation. It continues with correct reasoning as if the wrong step never happened. Even if it uses correct values in computation, classify as A ONLY if there is no reference to or engagement with the perturbation.

TYPE_B (SELF-CORRECTION): The model shows ANY evidence of engaging with the perturbation — using correct values where the perturbation gave wrong ones counts as correction, even without explicit language like "wait" or "actually."

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "A" or "B".""",
    "strict_correction": """Classify how a language model handled a perturbed reasoning step.
The model saw a WRONG step but still gave the CORRECT final answer.

TYPE_A (SILENT BYPASS): The model continues without explicitly acknowledging or correcting the error. This includes cases where the model happens to use correct values through computation — unless it explicitly flags the error, it's bypass.

TYPE_B (SELF-CORRECTION): The model EXPLICITLY acknowledges the error using language like "wait", "actually", "but that's wrong", "however the correct value is", etc. Simply computing with correct values without verbal correction is NOT Type B.

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "A" or "B".""",
    "reference_sensitive": """Classify how a language model handled a perturbed reasoning step.
The model saw a WRONG step but still gave the CORRECT final answer.

TYPE_A (SILENT BYPASS): The model does NOT reference, quote, or allude to any content from the perturbed step. It reasons independently.

TYPE_B (SELF-CORRECTION): The model directly references content from the perturbed step — either to correct it, question it, or explicitly override it. The model must show it "saw" the perturbation.

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "A" or "B".""",
    "original": """Classify how a language model handled a perturbed reasoning step.
The model saw a WRONG step but still gave the CORRECT final answer. Did it:
A) SILENT_BYPASS — ignored the wrong step, continued as if it was correct
B) SELF_CORRECTION — noticed the error and corrected it

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "A" or "B".""",
}


def get_final_label(d: dict) -> str:
    jl = d.get("judge_label")
    if jl in {"A", "B", "C"}:
        return jl
    lc = d.get("label_3class", "")
    if lc == "error_propagation":
        return "C"
    if lc == "silent_bypass":
        return "A"
    if lc == "self_correction":
        return "B"
    return "unclear"


def parse_label(text: str) -> str | None:
    t = (text or "").strip().upper()
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    return None


async def run_variant(
    pairs: list[dict],
    prompt_template: str,
    prompt_name: str,
    api_key: str,
    concurrency: int = 40,
) -> tuple[list[dict], int]:
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(concurrency)
    results: list[dict | None] = [None] * len(pairs)
    total_tokens = 0
    lock = asyncio.Lock()

    async def one(idx: int, pair: dict) -> None:
        nonlocal total_tokens
        async with sem:
            prompt = prompt_template.format(
                original_step=(pair.get("original_step") or "N/A")[:500],
                perturbed_step=(pair.get("perturbed_step") or "N/A")[:500],
                continuation=(pair.get("continuation") or "N/A")[:800],
            )
            label = None
            raw = ""
            for attempt in range(3):
                try:
                    resp = await client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=5,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = resp.content[0].text.strip()
                    label = parse_label(raw)
                    async with lock:
                        total_tokens += resp.usage.input_tokens + resp.usage.output_tokens
                    break
                except Exception as e:
                    if attempt == 2:
                        raw = f"error: {e}"
                    else:
                        msg = str(e).lower()
                        await asyncio.sleep(
                            min(2.0, 0.1 * (2 ** attempt))
                            if ("rate" in msg or "429" in msg)
                            else 0.5
                        )
            # Composite key: (base_question_id, strategy, position) uniquely
            # identifies a perturbed continuation. The base `id` field alone
            # is shared across multiple continuations of the same base
            # question — collapsing on it loses information.
            results[idx] = {
                "id": pair["id"],
                "perturbation_strategy": pair.get("perturbation_strategy"),
                "perturbation_point": pair.get("perturbation_point"),
                "source": pair.get("source"),
                "variant": prompt_name,
                "judge_label": label,
                "raw": raw,
            }

    t0 = time.time()
    BATCH = 500
    for i in range(0, len(pairs), BATCH):
        end = min(i + BATCH, len(pairs))
        await asyncio.gather(*[one(j, pairs[j]) for j in range(i, end)])
        done = sum(1 for r in results if r is not None)
        logging.info(f"  [{prompt_name}] {done}/{len(pairs)} done ({time.time()-t0:.0f}s)")

    return [r for r in results if r is not None], total_tokens


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    ap.add_argument("--sample", type=int, default=0, help="0=all non-C rows")
    ap.add_argument("--concurrency", type=int, default=40)
    args = ap.parse_args()

    if not args.api_key:
        logging.error("missing --api-key / ANTHROPIC_API_KEY")
        return

    src = DATA / "expanded_pairs.json"
    logging.info(f"loading {src}")
    with open(src) as f:
        data = json.load(f)

    # Resolve final_label, keep only A/B rows for re-judging.
    non_c: list[dict] = []
    for d in data:
        d["_final_label"] = get_final_label(d)
        if d["_final_label"] in {"A", "B"}:
            non_c.append(d)
    logging.info(f"non-C rows: {len(non_c)}; total: {len(data)}")

    if args.sample > 0:
        import random
        random.seed(42)
        non_c = random.sample(non_c, min(args.sample, len(non_c)))
        logging.info(f"sampled {len(non_c)} rows for smoke test")

    # Run each prompt variant
    all_rows: list[dict] = []
    total_tokens = 0
    for vname, vtemplate in PROMPTS.items():
        logging.info(f"=== {vname} ===")
        rows, tok = await run_variant(
            non_c, vtemplate, vname, args.api_key, args.concurrency
        )
        all_rows.extend(rows)
        total_tokens += tok
        labels = Counter(r["judge_label"] for r in rows)
        logging.info(f"  {vname}: label dist = {dict(labels)}; tokens = {tok}")

    OUT_JSONL.write_text(json.dumps(all_rows, indent=None))
    logging.info(f"wrote {OUT_JSONL} ({len(all_rows)} rows)")

    # Build composite-key -> {variant: label}.
    # Each (id, strategy, position) uniquely identifies a perturbed continuation.
    by_key: dict[tuple, dict[str, str | None]] = defaultdict(dict)
    metadata: dict[tuple, dict] = {}
    for r in all_rows:
        ck = (r["id"], r.get("perturbation_strategy"), r.get("perturbation_point"))
        by_key[ck][r["variant"]] = r["judge_label"]
        metadata.setdefault(ck, {
            "id": ck[0],
            "perturbation_strategy": ck[1],
            "perturbation_point": ck[2],
            "source": r.get("source"),
        })

    # Agreement stratification: across the 4 variants, how many unique labels?
    strata: dict[str, list[dict]] = {"agree": [], "split_2_2": [], "majority_3_1": [], "incomplete": []}
    for ck, labels_by_v in by_key.items():
        labels = [l for l in labels_by_v.values() if l in {"A", "B"}]
        meta = {**metadata[ck], "labels": labels_by_v}
        if len(labels) < 4:
            strata["incomplete"].append(meta)
            continue
        cnt = Counter(labels)
        if len(cnt) == 1:
            strata["agree"].append({**meta, "consensus": labels[0]})
        elif cnt.most_common(1)[0][1] == 3:
            strata["majority_3_1"].append({
                **meta,
                "majority": cnt.most_common(1)[0][0],
                "minority": cnt.most_common()[1][0],
            })
        else:  # 2-2
            strata["split_2_2"].append(meta)

    summary = {
        "n_records_unique_composite": len(by_key),
        "n_per_variant": {
            v: sum(1 for r in all_rows if r["variant"] == v) for v in PROMPTS
        },
        "stratum_sizes": {k: len(v) for k, v in strata.items()},
        "agreement_rate": (
            len(strata["agree"]) / len(by_key) if by_key else 0.0
        ),
        "estimated_cost_usd_haiku": round(total_tokens * 0.00000025, 2),
        "total_tokens": total_tokens,
    }
    by_source = defaultdict(lambda: Counter())
    for ck, info in by_key.items():
        labels = [l for l in info.values() if l in {"A", "B"}]
        if len(labels) < 4:
            continue
        cnt = Counter(labels)
        src = metadata[ck]["source"]
        if len(cnt) == 1:
            by_source[src]["agree"] += 1
        elif cnt.most_common(1)[0][1] == 3:
            by_source[src]["majority_3_1"] += 1
        else:
            by_source[src]["split_2_2"] += 1
    summary["by_source"] = {k: dict(v) for k, v in by_source.items()}

    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTDIR / "stratification.json").write_text(json.dumps(strata, indent=2))
    logging.info("wrote summary.json + stratification.json")
    logging.info(f"agreement rate: {summary['agreement_rate']*100:.1f}%")
    logging.info(
        f"strata: agree={summary['stratum_sizes']['agree']}, "
        f"majority_3_1={summary['stratum_sizes']['majority_3_1']}, "
        f"split_2_2={summary['stratum_sizes']['split_2_2']}"
    )
    logging.info(f"estimated cost (Haiku pricing): ${summary['estimated_cost_usd_haiku']}")


if __name__ == "__main__":
    asyncio.run(main())
