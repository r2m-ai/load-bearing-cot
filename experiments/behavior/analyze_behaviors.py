"""
Script 7 (v3): Behavioral analysis — position gradient and strategy effectiveness.

Produces figures and tables for the paper's behavioral results section:
- Position gradient: how causal influence varies with perturbation position
- Strategy comparison: which perturbation strategies produce more faithful behavior
- Type A/B/C breakdown by position and strategy
- Source comparison: GSM8K vs MMLU behavioral differences

Input: data/processed/subclassified_pairs.json
Output: figures/figure5_behavioral.png, figures/table4_behavioral.csv
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Behavioral analysis")
    args = parser.parse_args()

    # Load subclassified pairs
    sub_path = PROCESSED_DIR / "subclassified_pairs.json"
    if sub_path.exists():
        with open(sub_path) as f:
            pairs = json.load(f)
    else:
        with open(PROCESSED_DIR / "all_continuation_pairs.json") as f:
            pairs = json.load(f)
    print(f"Loaded {len(pairs)} pairs")

    # ---- Analysis 1: Position Gradient with Type A/B/C breakdown ----
    positions = ["early", "middle", "late"]
    type_labels = ["TYPE_A", "TYPE_B", "TYPE_C", "UNCLEAR", "NEEDS_JUDGE"]
    type_names = {
        "TYPE_A": "Silent Bypass",
        "TYPE_B": "Self-Correction",
        "TYPE_C": "Error Propagation",
        "UNCLEAR": "Unclear",
        "NEEDS_JUDGE": "Needs Judge",
    }
    type_colors = {
        "TYPE_A": "#F44336",   # Red — unfaithful
        "TYPE_B": "#FF9800",   # Orange — robust
        "TYPE_C": "#2196F3",   # Blue — faithful
        "UNCLEAR": "#9E9E9E",
        "NEEDS_JUDGE": "#BDBDBD",
    }

    pos_data = {}
    for pos in positions:
        subset = [p for p in pairs if p.get("perturbation_point") == pos]
        counts = Counter(p.get("subtype", "UNCLEAR") for p in subset)
        pos_data[pos] = {"total": len(subset), **{t: counts.get(t, 0) for t in type_labels}}

    # ---- Analysis 2: Strategy Effectiveness ----
    strategies = sorted(set(p.get("perturbation_strategy", "unknown") for p in pairs))
    strat_data = {}
    for strat in strategies:
        subset = [p for p in pairs if p.get("perturbation_strategy") == strat]
        counts = Counter(p.get("subtype", "UNCLEAR") for p in subset)
        strat_data[strat] = {"total": len(subset), **{t: counts.get(t, 0) for t in type_labels}}

    # ---- Analysis 3: Source Comparison ----
    source_data = {}
    for source in ["gsm8k", "mmlu", "bbh"]:
        subset = [p for p in pairs if p.get("source") == source]
        counts = Counter(p.get("subtype", "UNCLEAR") for p in subset)
        source_data[source] = {"total": len(subset), **{t: counts.get(t, 0) for t in type_labels}}

    # ---- Figure 5: Multi-panel behavioral analysis ----
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel A: Position gradient (stacked bar)
    ax = axes[0]
    x = np.arange(len(positions))
    width = 0.6
    bottom = np.zeros(len(positions))

    for t in ["TYPE_C", "TYPE_B", "TYPE_A"]:  # Stack order: faithful at bottom
        values = [pos_data[pos].get(t, 0) for pos in positions]
        totals = [pos_data[pos]["total"] for pos in positions]
        pcts = [v / t_val * 100 if t_val > 0 else 0 for v, t_val in zip(values, totals)]
        ax.bar(x, pcts, width, bottom=bottom, label=type_names[t], color=type_colors[t], alpha=0.85)
        bottom += np.array(pcts)

    ax.set_xticks(x)
    ax.set_xticklabels(["Early\n(25%)", "Middle\n(50%)", "Late\n(75%)"])
    ax.set_ylabel("% of Examples")
    ax.set_title("A) Behavior by Perturbation Position")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)

    # Add counts as annotations
    for i, pos in enumerate(positions):
        total = pos_data[pos]["total"]
        ax.text(i, 102, f"n={total}", ha="center", fontsize=9, color="gray")

    # Panel B: Strategy comparison
    ax = axes[1]
    # Only show strategies with enough data
    valid_strats = [s for s in strategies if strat_data[s]["total"] >= 10]
    x = np.arange(len(valid_strats))
    width = 0.6
    bottom = np.zeros(len(valid_strats))

    for t in ["TYPE_C", "TYPE_B", "TYPE_A"]:
        values = [strat_data[s].get(t, 0) for s in valid_strats]
        totals = [strat_data[s]["total"] for s in valid_strats]
        pcts = [v / t_val * 100 if t_val > 0 else 0 for v, t_val in zip(values, totals)]
        ax.bar(x, pcts, width, bottom=bottom, label=type_names[t], color=type_colors[t], alpha=0.85)
        bottom += np.array(pcts)

    ax.set_xticks(x)
    strat_labels = [s.replace("_", "\n") for s in valid_strats]
    ax.set_xticklabels(strat_labels, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("% of Examples")
    ax.set_title("B) Behavior by Perturbation Strategy")
    ax.set_ylim(0, 105)

    # Panel C: GSM8K vs MMLU
    ax = axes[2]
    sources = [s for s in ["gsm8k", "mmlu", "bbh"] if source_data.get(s, {}).get("total", 0) > 0]
    x = np.arange(len(sources))
    width = 0.5
    bottom = np.zeros(len(sources))

    for t in ["TYPE_C", "TYPE_B", "TYPE_A"]:
        values = [source_data[s].get(t, 0) for s in sources]
        totals = [source_data[s]["total"] for s in sources]
        pcts = [v / t_val * 100 if t_val > 0 else 0 for v, t_val in zip(values, totals)]
        ax.bar(x, pcts, width, bottom=bottom, label=type_names[t], color=type_colors[t], alpha=0.85)
        bottom += np.array(pcts)

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in sources])
    ax.set_ylabel("% of Examples")
    ax.set_title("C) Behavior by Task Type")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 105)

    for i, src in enumerate(sources):
        ax.text(i, 102, f"n={source_data[src]['total']}", ha="center", fontsize=9, color="gray")

    plt.suptitle("Behavioral Analysis: How Models Process Perturbed Chain-of-Thought", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure5_behavioral.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure5_behavioral.pdf", bbox_inches="tight")
    print("Saved Figure 5")

    # ---- Table 4: Full behavioral breakdown ----
    rows = []
    for pos in positions:
        d = pos_data[pos]
        total = d["total"]
        for t in ["TYPE_A", "TYPE_B", "TYPE_C"]:
            rows.append({
                "Category": "Position",
                "Setting": f"{pos} ({['25%','50%','75%'][positions.index(pos)]})",
                "Behavior": type_names[t],
                "Count": d.get(t, 0),
                "Percentage": f"{d.get(t,0)/total*100:.1f}%" if total > 0 else "0%",
            })

    for strat in valid_strats:
        d = strat_data[strat]
        total = d["total"]
        for t in ["TYPE_A", "TYPE_B", "TYPE_C"]:
            rows.append({
                "Category": "Strategy",
                "Setting": strat,
                "Behavior": type_names[t],
                "Count": d.get(t, 0),
                "Percentage": f"{d.get(t,0)/total*100:.1f}%" if total > 0 else "0%",
            })

    for src in sources:
        d = source_data[src]
        total = d["total"]
        for t in ["TYPE_A", "TYPE_B", "TYPE_C"]:
            rows.append({
                "Category": "Source",
                "Setting": src.upper(),
                "Behavior": type_names[t],
                "Count": d.get(t, 0),
                "Percentage": f"{d.get(t,0)/total*100:.1f}%" if total > 0 else "0%",
            })

    df = pd.DataFrame(rows)
    df.to_csv(FIGURES_DIR / "table4_behavioral.csv", index=False)
    print("Saved Table 4")

    # ---- Print Summary ----
    print(f"\n{'='*60}")
    print("BEHAVIORAL ANALYSIS SUMMARY")
    print(f"{'='*60}")

    print("\nPosition gradient (Type A / Type B / Type C):")
    for pos in positions:
        d = pos_data[pos]
        total = d["total"]
        if total > 0:
            a_pct = d.get("TYPE_A", 0) / total * 100
            b_pct = d.get("TYPE_B", 0) / total * 100
            c_pct = d.get("TYPE_C", 0) / total * 100
            print(f"  {pos:6s}: A={a_pct:.0f}% B={b_pct:.0f}% C={c_pct:.0f}% (n={total})")

    print("\nStrategy effectiveness:")
    for strat in valid_strats:
        d = strat_data[strat]
        total = d["total"]
        if total > 0:
            c_pct = d.get("TYPE_C", 0) / total * 100
            print(f"  {strat:25s}: {c_pct:.0f}% error propagation (n={total})")

    print("\nSource comparison:")
    for src in sources:
        d = source_data[src]
        total = d["total"]
        if total > 0:
            a_pct = d.get("TYPE_A", 0) / total * 100
            b_pct = d.get("TYPE_B", 0) / total * 100
            c_pct = d.get("TYPE_C", 0) / total * 100
            print(f"  {src.upper():6s}: A={a_pct:.0f}% B={b_pct:.0f}% C={c_pct:.0f}% (n={total})")


if __name__ == "__main__":
    main()
