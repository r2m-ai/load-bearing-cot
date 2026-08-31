"""
Plot judge sensitivity results: A/B split shifts across prompt variants while C stays fixed.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
RESULTS = ROOT / "results" / "v8"
FIGURES = ROOT / "figures"

with open(RESULTS / "exp1ab_judge_sensitivity.json") as f:
    data = json.load(f)

# Extract data for A/B variants only (skip causal_final_answer which has a bug)
variants = ["strict_bypass", "strict_correction", "reference_sensitive", "original"]
short_names = ["Strict\nbypass", "Strict\ncorrection", "Reference\nsensitive", "Original\nprompt"]

a_counts = []
b_counts = []
c_counts = []

# Use corrected C count (v9 fix: 3,229 BBH examples reclassified)
corrected_c = 5902

for v in variants:
    d = data["variants"][v]["label_distribution"]
    a_counts.append(d.get("A", 0))
    b_counts.append(d.get("B", 0))
    c_counts.append(corrected_c)  # Fixed: same across all variants

a_arr = np.array(a_counts)
b_arr = np.array(b_counts)
c_arr = np.array(c_counts)
totals = a_arr + b_arr + c_arr

a_pct = a_arr / totals * 100
b_pct = b_arr / totals * 100
c_pct = c_arr / totals * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [3, 2]})

# Left panel: stacked bars showing A/B/C proportions
x = np.arange(len(variants))
width = 0.55

bars_c = ax1.bar(x, c_pct, width, color="#d45f5f", label="Error Prop (C)", alpha=0.85)
bars_b = ax1.bar(x, b_pct, width, bottom=c_pct, color="#e8b84a", label="Self-Correct (B)", alpha=0.85)
bars_a = ax1.bar(x, a_pct, width, bottom=c_pct + b_pct, color="#4a90d9", label="Bypass (A)", alpha=0.85)

ax1.set_xticks(x)
ax1.set_xticklabels(short_names, fontsize=8.5)
ax1.set_ylabel("Proportion (%)", fontsize=10)
ax1.set_title("A/B boundary shifts; C stays fixed", fontsize=11)
ax1.legend(loc="upper right", fontsize=8, framealpha=0.9)
ax1.set_ylim(0, 105)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Add C% annotation on each bar
for i, cp in enumerate(c_pct):
    ax1.text(i, cp / 2, f"C={cp:.0f}%", ha="center", va="center", fontsize=7.5,
             color="white", fontweight="bold")

# Right panel: dataset-level C rates (identical across variants)
# Use corrected C rates from v9-fixed expanded_pairs.json
datasets = ["GSM8K", "MMLU", "BBH"]
rates = [3.9, 22.3, 40.9]  # Corrected C rates (v9 fix)

colors = ["#4a90d9", "#e8b84a", "#d45f5f"]
bars = ax2.bar(datasets, rates, color=colors, alpha=0.85, width=0.5, edgecolor="white", linewidth=0.5)

for bar, rate in zip(bars, rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
             f"{rate:.1f}%", ha="center", fontsize=9, fontweight="bold")

ax2.set_ylabel("Error propagation rate (%)", fontsize=10)
ax2.set_title("C rate invariant to prompt", fontsize=11)
ax2.set_ylim(0, 105)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# Add "identical across all variants" annotation
ax2.text(1, 80, "Same across all\n4 prompt variants", ha="center", fontsize=8,
         style="italic", alpha=0.6)

plt.tight_layout()
plt.savefig(FIGURES / "judge_sensitivity.png", dpi=300, bbox_inches="tight")
plt.savefig(FIGURES / "judge_sensitivity.pdf", bbox_inches="tight")
print(f"Saved to {FIGURES / 'judge_sensitivity.png'}")
