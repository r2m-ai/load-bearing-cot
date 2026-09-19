"""
Plot judge prompt sensitivity using the FULL-sample item-2.0 results
(15,336 non-Type-C records × 4 prompt variants), showing the order-of-
magnitude shift in A/B split versus the construction-invariant C count.

Inputs:
  results/validation/judge_prompt_variants/summary.json  -- per-variant
    A/B counts on the full non-C set, plus stratum sizes at the composite
    (id, strategy, position) key.
  data/processed/expanded_pairs.json  -- to obtain the fixed C count and
    the per-source breakdown.

Output:
  figures/judge_sensitivity.png + .pdf
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "results" / "validation" / "judge_prompt_variants" / "summary.json"
EXPANDED = ROOT / "data" / "processed" / "expanded_pairs.json"
FIGURES = ROOT / "figures"


def label_of(d: dict) -> str:
    jl = d.get("judge_label")
    if jl in {"A", "B", "C"}:
        return jl
    lc = d.get("label_3class", "")
    return {
        "error_propagation": "C",
        "silent_bypass": "A",
        "self_correction": "B",
    }.get(lc, "unclear")


def main() -> None:
    summary = json.load(open(SUMMARY))
    expanded = json.load(open(EXPANDED))

    # Number of Type-C records (judge-invariant by construction)
    c_count_by_source = Counter()
    for d in expanded:
        if label_of(d) == "C":
            c_count_by_source[d.get("source", "?")] += 1
    c_total = sum(c_count_by_source.values())

    variants = ["strict_bypass", "strict_correction", "reference_sensitive", "original"]
    short_names = ["Strict\nbypass", "Strict\ncorrection", "Reference\nsensitive", "Original\nprompt"]

    a_counts, b_counts = [], []
    for v in variants:
        dist = summary["per_variant_label_distribution"][v]
        a_counts.append(dist.get("A", 0))
        b_counts.append(dist.get("B", 0))
    a_arr = np.array(a_counts, dtype=float)
    b_arr = np.array(b_counts, dtype=float)
    c_arr = np.full(len(variants), float(c_total))
    totals = a_arr + b_arr + c_arr

    a_pct = 100 * a_arr / totals
    b_pct = 100 * b_arr / totals
    c_pct = 100 * c_arr / totals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={"width_ratios": [3, 2]})

    x = np.arange(len(variants))
    width = 0.55

    ax1.bar(x, c_pct, width, color="#d45f5f", label="Error Prop (C)", alpha=0.85)
    ax1.bar(x, b_pct, width, bottom=c_pct, color="#e8b84a",
            label="Self-Correct (B)", alpha=0.85)
    ax1.bar(x, a_pct, width, bottom=c_pct + b_pct, color="#4a90d9",
            label="Bypass (A)", alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, fontsize=8.5)
    ax1.set_ylabel("Proportion (%)", fontsize=10)
    ax1.set_title("A/B boundary shifts; C stays fixed (full sample, $n = 21{,}238$)",
                  fontsize=10.5, pad=18)
    ax1.legend(loc="center right", fontsize=8, framealpha=0.9)
    ax1.set_ylim(0, 118)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Annotate the within-non-C A% per variant — this is the metric the
    # paper text reports (the "22× spread" claim is on this denominator).
    n_nonC = (a_arr + b_arr)
    a_pct_within_nonC = 100 * a_arr / n_nonC
    for i, ap in enumerate(a_pct_within_nonC):
        top = c_pct[i] + b_pct[i] + a_pct[i]
        ax1.annotate(f"{ap:.1f}% A\n(of non-C)",
                     xy=(i, top + 1.5),
                     ha="center", fontsize=8, color="#235a87")

    # Right panel: C rate by source, across 4 variants — should be flat lines
    sources = ["gsm8k", "mmlu", "bbh"]
    src_n = {s: 0 for s in sources}
    src_c = {s: 0 for s in sources}
    for d in expanded:
        s = d.get("source")
        lab = label_of(d)
        if s in src_n and lab in {"A", "B", "C"}:
            src_n[s] += 1
            if lab == "C":
                src_c[s] += 1
    src_rate = {s: 100 * src_c[s] / src_n[s] for s in sources}

    colors = {"gsm8k": "#4a90d9", "mmlu": "#e07040", "bbh": "#5cb85c"}
    for s in sources:
        rate = src_rate[s]
        ax2.plot(x, [rate] * len(x), "-o", color=colors[s], label=s.upper(),
                 linewidth=1.8, markersize=6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, fontsize=8.5)
    ax2.set_ylabel("Error propagation rate (%)", fontsize=10)
    ax2.set_title("C rate invariant across variants", fontsize=10.5)
    ax2.set_ylim(0, 50)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    plt.savefig(FIGURES / "judge_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES / "judge_sensitivity.pdf", bbox_inches="tight")
    print(f"Saved {FIGURES / 'judge_sensitivity.png'} and .pdf")
    print(f"Per-variant A%: {[(v, f'{p:.1f}%') for v, p in zip(variants, a_pct)]}")
    print(f"C rate per source: {src_rate}")


if __name__ == "__main__":
    main()
