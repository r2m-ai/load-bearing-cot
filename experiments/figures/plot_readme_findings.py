"""README findings figure: gradient, within-MMLU spread, and the matched controls.

One panel per headline finding. Nothing here reruns inference.

Sources
  (a) paper/main.tex, tab:by_dataset and tab:crossmodel (base accuracy, C rate).
  (b) results/controls/variance_decomposition/results.json, within_mmlu_subject.
  (c) results/controls/variance_decomposition/results.json, cell_summary and
      deviance_partition; the text-on-multistep-arithmetic cell is the BBH
      subtask breakdown in the paper appendix (n = 612), which the released
      snapshot pools into bbh_other.

Run from the repository root:
    python experiments/figures/plot_readme_findings.py
"""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "results/controls/variance_decomposition/results.json"
DESTINATION = ROOT / "docs/assets/research_findings.svg"

# Okabe-Ito, colourblind-safe.
GEMMA, LLAMA, DEEPSEEK = "#0072B2", "#D55E00", "#009E73"
NUMERICAL, TEXTUAL = "#C9686B", "#7B8799"
INK, RULE, GRID = "#30343b", "#c5c8cc", "#eceef0"

# (dataset, base accuracy %, error propagation %) per model, paper tables 1 and 3.
BY_MODEL = {
    "Gemma-2-9B-IT": [("GSM8K", 86.5, 3.9), ("MMLU", 74.8, 22.3), ("BBH", 57.9, 40.9)],
    "Llama-3.1-8B-Instruct": [("GSM8K", 55.3, 12.4), ("MMLU", 40.8, 62.8), ("BBH", 40.1, 65.5)],
    "DeepSeek-R1-Distill-7B": [("GSM8K", 38.5, 5.1), ("MMLU", 52.4, 2.7), ("BBH", 20.2, 16.8)],
}
STYLE = {"Gemma-2-9B-IT": (GEMMA, "o"),
         "Llama-3.1-8B-Instruct": (LLAMA, "^"),
         "DeepSeek-R1-Distill-7B": (DEEPSEEK, "s")}
# Label offsets in points, keyed by (model, dataset), where the default would collide.
NUDGE = {("Llama-3.1-8B-Instruct", "MMLU"): (-6, -16),
         ("DeepSeek-R1-Distill-7B", "MMLU"): (6, -16),
         ("DeepSeek-R1-Distill-7B", "GSM8K"): (-4, -16)}

# Text perturbations on BBH multistep arithmetic; paper appendix, BBH subtask table.
TEXT_ON_MULTISTEP_ARITH = 18.6


def rate(cells, key, expected):
    """Percentage C rate for a cell of the released snapshot, checked against the paper."""
    value = round(cells[key]["c_rate"] * 100, 1)
    assert value == expected, f"{key}: snapshot {value}% vs paper {expected}%"
    return value


def panel_gradient(ax):
    """a. Error propagation rises as the model's own accuracy on the task falls."""
    for model, points in BY_MODEL.items():
        color, marker = STYLE[model]
        ordered = sorted(points, key=lambda p: p[1])
        ax.plot([p[1] for p in ordered], [p[2] for p in ordered],
                color=color, linewidth=1.1, alpha=.4, zorder=2)
        ax.scatter([p[1] for p in points], [p[2] for p in points], s=62, color=color,
                   marker=marker, label=model, edgecolors="white", linewidths=1.1, zorder=3)
        for name, x, y in points:
            ax.annotate(name, (x, y), xytext=NUDGE.get((model, name), (0, 9)),
                        textcoords="offset points", ha="center", fontsize=7.5, color=color)
    ax.set_xlim(15, 95)
    ax.set_ylim(-4, 78)
    ax.set_xlabel("Baseline accuracy on the task (%)")
    ax.set_ylabel("Error propagation (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper right", handletextpad=.4,
              borderpad=0, labelspacing=.35)
    ax.set_title("a   Harder for the model $\\rightarrow$ more load-bearing",
                 loc="left", fontweight="bold", pad=14)


def panel_mmlu(ax, subjects):
    """b. The same gradient inside MMLU, with format and strategies held fixed."""
    ordered = sorted(subjects.items(), key=lambda kv: kv[1]["c_rate"])
    values = np.array([v["c_rate"] * 100 for _, v in ordered])
    x = np.arange(len(values))
    ax.bar(x, values, width=.74, color=GEMMA, alpha=.8, linewidth=0)
    endpoints = [(0, "HS psychology", (6, 66), "left"),
                 (len(x) - 1, "Global facts", (-8, 8), "right")]
    for index, label, offset, align in endpoints:
        ax.annotate(f"{label}\n{values[index]:.1f}%", (index, values[index]),
                    xytext=offset, textcoords="offset points", ha=align, fontsize=8,
                    color=INK, linespacing=1.4,
                    arrowprops=dict(arrowstyle="-", color=RULE, linewidth=.8,
                                    shrinkA=2, shrinkB=3) if align == "left" else None)
    ax.set_xlim(-1.2, len(x) + .2)
    ax.set_ylim(0, 66)
    ax.set_xticks([])
    ax.set_xlabel("29 MMLU subjects, ordered by error propagation")
    ax.set_ylabel("Error propagation (%)")
    ax.set_title("b   A $7\\times$ spread at fixed question format",
                 loc="left", fontweight="bold", pad=14)


def panel_controls(ax, cells, partition):
    """c. Matched 2x2: difficulty moves the rate, perturbation type barely does."""
    numerical = [rate(cells, "gsm8k|numerical", 3.9),
                 rate(cells, "bbh_multistep_arith|numerical", 64.5)]
    textual = [rate(cells, "gsm8k|text", 6.5), TEXT_ON_MULTISTEP_ARITH]
    x = np.arange(2)
    for shift, values, label, color in [(-.19, numerical, "Numerical perturbation", NUMERICAL),
                                        (.19, textual, "Text perturbation", TEXTUAL)]:
        bars = ax.bar(x + shift, values, width=.34, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=3, fontsize=8.5)
    ax.annotate("$16\\times$ within\nnumerical", xy=(1 - .32, 62), xytext=(.30, 44),
                fontsize=8, color="#682f34", ha="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color="#682f34", linewidth=.9,
                                connectionstyle="arc3,rad=-.25"))
    share = partition["components"]["difficulty"]["fraction_of_total"] * 100
    type_share = partition["components"]["type"]["fraction_of_total"] * 100
    ax.text(.5, -.245, f"Explained deviance over $n = {partition['n']:,}$ continuations:\n"
                       f"{share:.1f}% task difficulty, {type_share:.1f}% perturbation type",
            transform=ax.transAxes, ha="center", va="top", fontsize=8,
            color="#5b6068", linespacing=1.5)
    ax.set_xticks(x, ["GSM8K\n(easy)", "BBH multistep\narithmetic (hard)"])
    ax.set_ylim(0, 78)
    ax.set_ylabel("Error propagation (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", handletextpad=.5,
              borderpad=0, labelspacing=.35)
    ax.set_title("c   Difficulty, not perturbation type",
                 loc="left", fontweight="bold", pad=14)


def main():
    snapshot = json.loads(SNAPSHOT.read_text())
    for key, expected in [("mmlu|text", 22.3), ("bbh_other|text", 40.9)]:
        rate(snapshot["cell_summary"], key, expected)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": INK, "text.color": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         # Glyphs as paths: the README renders identically for
                         # viewers without DejaVu Sans installed.
                         "svg.fonttype": "path"})
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1),
                             gridspec_kw={"width_ratios": [1.12, 1, .92]})
    fig.patch.set_facecolor("white")
    panel_gradient(axes[0])
    panel_mmlu(axes[1], snapshot["within_mmlu_subject"]["by_subject"])
    panel_controls(axes[2], snapshot["cell_summary"], snapshot["deviance_partition"])
    for ax in axes:
        ax.spines["left"].set_color(RULE)
        ax.spines["bottom"].set_color(RULE)
        ax.tick_params(length=0, pad=6)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=.7)
    fig.subplots_adjust(left=.055, right=.99, top=.87, bottom=.235, wspace=.30)

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINATION, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {DESTINATION.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
