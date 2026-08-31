"""exp_3_1_judge_deepseek.py — run the canonical judge stage on the DeepSeek
exp_3_1 step3_subclassified.json file.

The DeepSeek pipeline (revision/scripts/exp_3_1_deepseek_crossmodel.py)
stops at step 3 (rule-based subclassification). For the paper's
load-bearing claim about C-vs-non-C invariance and for the §5.5
cross-model table, we need judge-resolved A/B/C labels on the
NEEDS_JUDGE records, the same way the main pipeline does
(scripts/9b_run_judge.py).

This script is a thin wrapper: load step3_subclassified.json, run the
canonical A/B judge prompt on every NEEDS_JUDGE record, then write
step3_labeled.json with the resolved labels.

Usage:
  python revision/scripts/exp_3_1_judge_deepseek.py --api-key $ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "revision" / "revision_exp_results" / "exp_3_1" / "step3_subclassified.json"
OUTPUT = ROOT / "revision" / "revision_exp_results" / "exp_3_1" / "step3_labeled.json"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"exp_3_1_judge_deepseek_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

JUDGE_PROMPT_AB = """Classify how a language model handled a perturbed reasoning step.

The model saw a WRONG step but still gave the CORRECT final answer. Did it:
A) SILENT_BYPASS — ignored the wrong step, continued as if it was correct
B) SELF_CORRECTION — noticed the error and corrected it

ORIGINAL STEP: {original_step}
PERTURBED STEP: {perturbed_step}
CONTINUATION: {continuation}

Reply with ONLY "A" or "B"."""


def parse_ab(text: str) -> str | None:
    t = (text or "").strip().upper()
    if t.startswith("A"):
        return "A"
    if t.startswith("B"):
        return "B"
    return None


async def judge_one(client, sem, lock, totals, pair, idx, results):
    async with sem:
        prompt = JUDGE_PROMPT_AB.format(
            original_step=(pair.get("original_step") or "N/A")[:500],
            perturbed_step=(pair.get("perturbed_step") or "N/A")[:500],
            continuation=(pair.get("continuation") or "N/A")[:800],
        )
        for attempt in range(3):
            try:
                resp = await client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=5,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.content[0].text.strip()
                label = parse_ab(raw)
                async with lock:
                    totals["tokens"] += resp.usage.input_tokens + resp.usage.output_tokens
                results[idx] = (label, raw)
                return
            except Exception as e:
                msg = str(e).lower()
                if attempt == 2:
                    results[idx] = (None, f"error: {e}")
                    return
                await asyncio.sleep(min(2.0, 0.1 * (2 ** attempt))
                                     if ("rate" in msg or "429" in msg) else 0.5)


async def main_async(args) -> None:
    import anthropic

    data = json.load(open(INPUT))
    logging.info(f"Loaded {len(data)} records from {INPUT.name}")
    logging.info(f"Subtype distribution: {Counter(r.get('subtype') for r in data)}")

    needs = [(i, r) for i, r in enumerate(data) if r.get("subtype") == "NEEDS_JUDGE"]
    logging.info(f"NEEDS_JUDGE records to re-classify: {len(needs)}")

    client = anthropic.AsyncAnthropic(api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    totals = {"tokens": 0}
    results: list = [None] * len(needs)

    t0 = time.time()
    BATCH = 500
    for i in range(0, len(needs), BATCH):
        end = min(i + BATCH, len(needs))
        chunk = [judge_one(client, sem, lock, totals, needs[j][1], j, results) for j in range(i, end)]
        await asyncio.gather(*chunk)
        done = sum(1 for r in results if r is not None)
        logging.info(f"  {done}/{len(needs)} done ({time.time()-t0:.0f}s)")

    # Write back into data
    upgraded = Counter()
    for (idx_in_data, _), (label, raw) in zip(needs, results):
        if label in {"A", "B"}:
            data[idx_in_data]["judge_label"] = label
            data[idx_in_data]["judge_raw"] = raw
            data[idx_in_data]["label_3class"] = (
                "silent_bypass" if label == "A" else "self_correction"
            )
            data[idx_in_data]["subtype"] = "TYPE_A" if label == "A" else "TYPE_B"
            upgraded[label] += 1
        else:
            data[idx_in_data]["judge_label"] = None
            data[idx_in_data]["judge_raw"] = raw
            upgraded["unresolved"] += 1

    OUTPUT.write_text(json.dumps(data, indent=None))
    logging.info(
        f"Wrote {OUTPUT} — judge resolved {upgraded['A']} A, "
        f"{upgraded['B']} B, {upgraded.get('unresolved', 0)} still unresolved"
    )
    cost = totals["tokens"] * 0.00000025
    logging.info(f"Tokens: {totals['tokens']:,}; estimated Haiku cost: ${cost:.2f}")

    # Final breakdown by source
    print("\n=== Final A/B/C by source (post-judge) ===")
    by_src = {}
    for r in data:
        st = r.get("subtype")
        lc = r.get("label_3class")
        if st == "TYPE_A":
            lab = "A"
        elif st == "TYPE_B":
            lab = "B"
        elif st == "TYPE_C":
            lab = "C"
        else:
            continue
        by_src.setdefault(r.get("source", "?"), []).append(lab)
    for src, lbls in sorted(by_src.items()):
        n = len(lbls)
        a = lbls.count("A")
        b = lbls.count("B")
        c = lbls.count("C")
        print(f"  {src}: n={n}  A={a/n*100:.1f}%  B={b/n*100:.1f}%  C={c/n*100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    ap.add_argument("--concurrency", type=int, default=30)
    args = ap.parse_args()
    if not args.api_key:
        logging.error("missing --api-key / ANTHROPIC_API_KEY")
        return
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
