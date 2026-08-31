"""Generate two supplemental figures:
  figures/variance_decomp.png  -- 98.8/0.8/0.4 deviance partition
  figures/human_agreement.png  -- per-source human-vs-judge / human-vs-rule
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]


def wilson(p, n, z=1.96):
    if n == 0:
        return 0, 0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, center - halfwidth), min(1, center + halfwidth)


# ===========================================================
# FIGURE A: Variance decomposition
# ===========================================================
res = json.load(open(REPO / "revision/revision_exp_results/exp_1_5_variance/results.json"))
parts = res["deviance_partition"]["components"]
total = res["deviance_partition"]["total_explained"]
n = res["deviance_partition"]["n"]

frac = {
    "Task difficulty": parts["difficulty"]["fraction_of_total"] * 100,
    "Perturbation type": parts["type"]["fraction_of_total"] * 100,
    "Type $\\times$ difficulty": parts["interaction"]["fraction_of_total"] * 100,
}
dev = {
    "Task difficulty": parts["difficulty"]["deviance_explained"],
    "Perturbation type": parts["type"]["deviance_explained"],
    "Type $\\times$ difficulty": parts["interaction"]["deviance_explained"],
}
pvals = {
    "Task difficulty": parts["difficulty"]["p_value"],
    "Perturbation type": parts["type"]["p_value"],
    "Type $\\times$ difficulty": parts["interaction"]["p_value"],
}

fig, ax = plt.subplots(figsize=(11, 2.6))
colors = ["#2196F3", "#FF9800", "#F44336"]   # match figure5_behavioral Material palette
left = 0
for i, (label, color) in enumerate(zip(frac.keys(), colors)):
    f = frac[label]
    ax.barh(0, f, left=left, color=color, edgecolor="white", linewidth=1.2, alpha=0.85)
    # Annotate large slice in-bar, small slices with arrow
    if f > 5:
        ax.text(left + f / 2, 0, f"{label}\n{f:.1f}%  ($\\Delta = {dev[label]:.0f}$)",
                ha="center", va="center", color="white",
                fontsize=11, fontweight="bold")
    else:
        # offset annotation above bar
        ax.annotate(
            f"{label}\n{f:.2f}%  ($\\Delta = {dev[label]:.1f}$)",
            xy=(left + f / 2, 0.45),
            xytext=(left + f / 2, 1.0 + i * 0.4),
            ha="center", va="bottom",
            fontsize=9,
            arrowprops=dict(arrowstyle="-", color="black", lw=0.6),
        )
    left += f

ax.set_xlim(0, 100)
ax.set_ylim(-0.6, 2.0)
ax.set_yticks([])
ax.set_xlabel("Fraction of explained deviance (%)")
ax.set_xticks([0, 25, 50, 75, 100])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_title(
    f"Logistic mixed model on $n = {n:,}$ continuations "
    f"($\\sim$type $\\times$ difficulty + (1|base\\_question)); "
    f"all components $p < 10^{{-3}}$.",
    fontsize=10,
)
plt.tight_layout()
out_a = REPO / "figures/variance_decomp.png"
plt.savefig(out_a, dpi=200, bbox_inches="tight")
plt.close()
print(f"saved -> {out_a}")
print(f"  partition: {frac}")

# ===========================================================
# FIGURE B: Human annotation per-source agreement
# ===========================================================
hr = json.load(open(REPO / "revision/human_annotation/results_n200.json"))

sources = ["gsm8k", "mmlu", "bbh"]
src_labels = ["GSM8K", "MMLU", "BBH"]

n_auto, agree_auto = [], []
n_judge, agree_judge = [], []
for s in sources:
    v = hr["by_source"][s]
    n_auto.append(v["n"])
    agree_auto.append(v["agree_auto"])
    n_judge.append(v["n_judge"])
    agree_judge.append(v["agree_judge"])

# Overall
ov = hr["overall"]
n_auto_ov, agree_auto_ov = ov["n"], ov["agree_auto"]
n_judge_ov, agree_judge_ov = ov["n_judge"], ov["agree_judge"]

p_auto = [a / n for a, n in zip(agree_auto, n_auto)]
p_judge = [a / n for a, n in zip(agree_judge, n_judge)]
ci_auto = [wilson(p, n) for p, n in zip(p_auto, n_auto)]
ci_judge = [wilson(p, n) for p, n in zip(p_judge, n_judge)]

p_auto_ov = agree_auto_ov / n_auto_ov
p_judge_ov = agree_judge_ov / n_judge_ov

fig, ax = plt.subplots(figsize=(8.5, 4.0))
x = np.arange(len(sources))
width = 0.36

JUDGE_COLOR = "#2196F3"   # match figure5_behavioral Material blue (Error-Prop / faithful tracking)
RULE_COLOR = "#FF9800"    # match figure5_behavioral Material orange (Self-Correction)

bars_judge = ax.bar(
    x - width / 2,
    [p * 100 for p in p_judge],
    width,
    yerr=[[(p - lo) * 100 for p, (lo, _) in zip(p_judge, ci_judge)],
          [(hi - p) * 100 for p, (_, hi) in zip(p_judge, ci_judge)]],
    capsize=4,
    color=JUDGE_COLOR,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.85,
    label=f"human $\\leftrightarrow$ LLM judge (overall {p_judge_ov*100:.1f}%, $n={n_judge_ov}$)",
)
bars_auto = ax.bar(
    x + width / 2,
    [p * 100 for p in p_auto],
    width,
    yerr=[[(p - lo) * 100 for p, (lo, _) in zip(p_auto, ci_auto)],
          [(hi - p) * 100 for p, (_, hi) in zip(p_auto, ci_auto)]],
    capsize=4,
    color=RULE_COLOR,
    edgecolor="black",
    linewidth=0.6,
    alpha=0.85,
    label=f"human $\\leftrightarrow$ rule-based (overall {p_auto_ov*100:.1f}%, $n={n_auto_ov}$)",
)

# Annotate bar tops
for i, (pj, na, nj, pa) in enumerate(zip(p_judge, n_auto, n_judge, p_auto)):
    ax.text(i - width / 2, pj * 100 + 2, f"{pj*100:.0f}%\n($n={nj}$)",
            ha="center", va="bottom", fontsize=8)
    ax.text(i + width / 2, pa * 100 + 2, f"{pa*100:.0f}%\n($n={na}$)",
            ha="center", va="bottom", fontsize=8)

# Reference line at the overall rates
ax.axhline(p_judge_ov * 100, color=JUDGE_COLOR, linestyle="--", alpha=0.35, lw=1)
ax.axhline(p_auto_ov * 100, color=RULE_COLOR, linestyle="--", alpha=0.35, lw=1)

ax.set_xticks(x)
ax.set_xticklabels(src_labels)
ax.set_ylabel("Agreement with human (%)")
ax.set_ylim(0, 110)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(True, alpha=0.25, axis="y")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2,
          fontsize=9, framealpha=0.95)
ax.set_title(
    "Per-source human-validation agreement ($n = 200$, one ML-literate annotator, "
    "stratified design over-weights variant-disagreement cells)",
    fontsize=9.5,
)

plt.tight_layout()
out_b = REPO / "figures/human_agreement.png"
plt.savefig(out_b, dpi=200, bbox_inches="tight")
plt.close()
print(f"saved -> {out_b}")
print(f"  GSM:  judge {p_judge[0]*100:.1f}%  auto {p_auto[0]*100:.1f}%")
print(f"  MMLU: judge {p_judge[1]*100:.1f}%  auto {p_auto[1]*100:.1f}%")
print(f"  BBH:  judge {p_judge[2]*100:.1f}%  auto {p_auto[2]*100:.1f}%")
