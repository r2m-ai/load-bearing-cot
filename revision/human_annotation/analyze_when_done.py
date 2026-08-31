#!/usr/bin/env python3
"""
Post-annotation analysis (item 2.1 + 2.3).

Run this after the annotator returns `data_to_label_DONE.csv` (filename
free — pass via --annotated argument). Computes:

  1. Calibration check on unambiguous strata (GSM_type_c / BBH_type_a /
     BBH_type_c). Human–auto agreement should be ≥90%. Below that
     suggests the annotation protocol itself drifted.
  2. Overall human↔auto and human↔judge agreement, with 95% Wilson CIs.
  3. Per-source, per-stratum agreement breakdown.
  4. (Item 2.3 add-on, optional) Probe-vs-human evaluation: load the
     existing N=2,000 bypass probe and evaluate its predictions on the
     200 human-labeled examples. Requires probe checkpoint path.

Output: prints a console report and writes
`revision/human_annotation/results_n200.json` with all numbers.

Usage:
  python3 revision/human_annotation/analyze_when_done.py \\
      --annotated revision/human_annotation/data_to_label_DONE.csv \\
      [--probe-checkpoint path/to/bypass_probe.pkl]
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0, centre - half), min(1, centre + half))


def fmt_pct_ci(k: int, n: int) -> str:
    if n == 0:
        return "—"
    lo, hi = wilson_ci(k, n)
    return f"{k/n*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}] (n={n})"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_and_join(annotated_path: Path, metadata_path: Path) -> list[dict]:
    """Join the annotator's filled CSV with the hidden metadata."""
    annotated = load_csv(annotated_path)
    meta = {r["annotation_id"]: r for r in load_csv(metadata_path)}
    joined = []
    for r in annotated:
        aid = r.get("annotation_id", "")
        m = meta.get(aid)
        if not m:
            print(f"  warn: no metadata for annotation_id={aid}", file=sys.stderr)
            continue
        # Merge — annotator fields win on collision (they shouldn't collide).
        rec = {**m, **r}
        joined.append(rec)
    return joined


def normalize_label(s: str) -> str:
    """Normalize a label to A / B / C / U / ''."""
    s = (s or "").strip().upper()
    if s in ("A", "B", "C", "U"):
        return s
    # Accept "TYPE_A", "Type A", "silent_bypass" -> A, etc.
    if "TYPE_A" in s or "BYPASS" in s:
        return "A"
    if "TYPE_B" in s or "SELF" in s and "CORR" in s:
        return "B"
    if "TYPE_C" in s or "PROPAGAT" in s:
        return "C"
    if "UNCLEAR" in s:
        return "U"
    return ""


def report(joined: list[dict]) -> dict:
    """Compute and print the full agreement report; return as dict."""
    out: dict = {}
    print("=" * 70)
    print(f"HUMAN-ANNOTATION AGREEMENT REPORT  (n={len(joined)})")
    print("=" * 70)

    # ---- Coverage ----
    labels_present = Counter(normalize_label(r.get("human_label", "")) for r in joined)
    print(f"\nHuman labels filled: {dict(labels_present)}")
    n_labeled = sum(v for k, v in labels_present.items() if k)
    n_total = len(joined)
    print(f"Coverage: {n_labeled}/{n_total} ({n_labeled/n_total*100:.1f}%)")
    out["coverage"] = {"labeled": n_labeled, "total": n_total}
    if n_labeled < n_total * 0.9:
        print("⚠️  <90% of rows have a human_label; results below are partial.")

    # ---- 1. Calibration check on unambiguous strata ----
    print(f"\n{'─' * 70}")
    print("1. CALIBRATION on unambiguous strata (human should agree ≥90%)")
    print(f"{'─' * 70}")
    unambig_strata = {"GSM_type_c", "BBH_type_a", "BBH_type_c"}
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for r in joined:
        by_stratum[r.get("stratum", "?")].append(r)
    out["calibration"] = {}
    for s in sorted(unambig_strata):
        sub = by_stratum.get(s, [])
        agree = sum(
            1 for r in sub
            if normalize_label(r["human_label"]) == normalize_label(r["auto_label"])
        )
        line = f"  {s:18s}  human=auto: {fmt_pct_ci(agree, len(sub))}"
        if sub and agree / len(sub) < 0.9:
            line += "   ⚠️  below 90% — protocol drift?"
        print(line)
        out["calibration"][s] = {"n": len(sub), "agree": agree}

    # ---- 2. Overall agreement: human vs auto, human vs judge ----
    print(f"\n{'─' * 70}")
    print("2. OVERALL AGREEMENT (95% Wilson CIs)")
    print(f"{'─' * 70}")
    labeled = [r for r in joined if normalize_label(r.get("human_label", ""))]
    n = len(labeled)
    agree_auto = sum(
        1 for r in labeled
        if normalize_label(r["human_label"]) == normalize_label(r["auto_label"])
    )
    print(f"  human ↔ auto (rule-based):   {fmt_pct_ci(agree_auto, n)}")
    out["overall"] = {"n": n, "agree_auto": agree_auto}

    judge_labeled = [r for r in labeled if normalize_label(r.get("judge_label", ""))]
    agree_judge = sum(
        1 for r in judge_labeled
        if normalize_label(r["human_label"]) == normalize_label(r["judge_label"])
    )
    print(f"  human ↔ judge (LLM Haiku):    {fmt_pct_ci(agree_judge, len(judge_labeled))}")
    out["overall"]["agree_judge"] = agree_judge
    out["overall"]["n_judge"] = len(judge_labeled)

    # ---- 3. Per-source breakdown ----
    print(f"\n{'─' * 70}")
    print("3. PER-SOURCE BREAKDOWN")
    print(f"{'─' * 70}")
    out["by_source"] = {}
    for src in ["gsm8k", "mmlu", "bbh"]:
        sub = [r for r in labeled if r["source"] == src]
        if not sub:
            continue
        agr_a = sum(
            1 for r in sub
            if normalize_label(r["human_label"]) == normalize_label(r["auto_label"])
        )
        sub_j = [r for r in sub if normalize_label(r.get("judge_label", ""))]
        agr_j = sum(
            1 for r in sub_j
            if normalize_label(r["human_label"]) == normalize_label(r["judge_label"])
        )
        print(f"  {src:6s}  human↔auto: {fmt_pct_ci(agr_a, len(sub))}   "
              f"human↔judge: {fmt_pct_ci(agr_j, len(sub_j))}")
        out["by_source"][src] = {
            "n": len(sub), "agree_auto": agr_a,
            "n_judge": len(sub_j), "agree_judge": agr_j,
        }

    # ---- 4. By-stratum (gives the M3 "disagree" approximation) ----
    print(f"\n{'─' * 70}")
    print("4. BY STRATUM (the borderline cells matter most)")
    print(f"{'─' * 70}")
    out["by_stratum"] = {}
    for s in sorted(by_stratum):
        sub = [r for r in by_stratum[s] if normalize_label(r.get("human_label", ""))]
        if not sub:
            continue
        agr_a = sum(
            1 for r in sub
            if normalize_label(r["human_label"]) == normalize_label(r["auto_label"])
        )
        sub_j = [r for r in sub if normalize_label(r.get("judge_label", ""))]
        agr_j = sum(
            1 for r in sub_j
            if normalize_label(r["human_label"]) == normalize_label(r["judge_label"])
        )
        print(f"  {s:18s}  human↔auto: {fmt_pct_ci(agr_a, len(sub))}   "
              f"human↔judge: {fmt_pct_ci(agr_j, len(sub_j))}")
        out["by_stratum"][s] = {
            "n": len(sub), "agree_auto": agr_a,
            "n_judge": len(sub_j), "agree_judge": agr_j,
        }

    # ---- 5. Confidence distribution ----
    print(f"\n{'─' * 70}")
    print("5. CONFIDENCE DISTRIBUTION (annotator's self-reported confidence)")
    print(f"{'─' * 70}")
    conf = Counter(r.get("human_confidence", "").lower() for r in labeled)
    for k in ["high", "medium", "low", ""]:
        if conf.get(k):
            print(f"  {k or '(blank)':8s}  {conf[k]:4d}  ({conf[k]/len(labeled)*100:.1f}%)")
    out["confidence"] = dict(conf)

    return out


def probe_vs_human(joined, probe_ckpt_path: Path) -> dict | None:
    """Item 2.3 add-on. Evaluate existing N=2,000 bypass probe on the
    200 human-labeled subset. Requires a sklearn probe checkpoint at
    `probe_ckpt_path` that exposes .predict(X) and the X features for
    these 200 examples. See `scripts/5b_reframed_probe.py` for the
    matching feature-extraction code.

    This is a stub — wire up to the specific probe artifact in your
    repo (the path depends on which probe ran in the paper's main
    pipeline). Returns None if not available."""
    if not probe_ckpt_path or not probe_ckpt_path.exists():
        print(f"\n(probe checkpoint not provided / not found — skipping item 2.3 add-on)")
        return None
    # Implement once the probe checkpoint path is fixed.
    print(f"\nItem 2.3 (probe-vs-human eval) — TODO: wire up to {probe_ckpt_path}")
    return None


def main():
    ap = argparse.ArgumentParser(description="Post-annotation agreement analysis (item 2.1 + 2.3)")
    ap.add_argument("--annotated", type=Path, required=True,
                    help="Path to the annotator's filled CSV")
    ap.add_argument("--metadata", type=Path,
                    default=Path(__file__).parent / "_analysis_metadata.csv",
                    help="Hidden metadata CSV (default: same dir)")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "results_n200.json",
                    help="Output JSON path")
    ap.add_argument("--probe-checkpoint", type=Path, default=None,
                    help="Optional: bypass-probe checkpoint for item 2.3")
    args = ap.parse_args()

    joined = load_and_join(args.annotated, args.metadata)
    print(f"Joined: {len(joined)} rows from {args.annotated.name} + {args.metadata.name}\n")

    results = report(joined)
    probe_results = probe_vs_human(joined, args.probe_checkpoint)
    if probe_results:
        results["probe_vs_human"] = probe_results

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
