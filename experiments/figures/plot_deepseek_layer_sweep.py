"""
Plot the per-layer probe accuracy for DeepSeek-R1-Distill-Qwen-7B
from the precomputed probe_results.csv. Three panels match the
main paper's Gemma layer sweep: 3-class, binary bypass, binary
error propagation. Linear (solid) and MLP (dashed) for each task.

Input:  results/models/deepseek/probe_results.csv
Output: figures/deepseek_layer_sweep.png + .pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "models" / "deepseek" / "probe_results.csv"
FIGURES = ROOT / "figures"


def main() -> None:
    df = pd.read_csv(CSV)

    # The CSV has columns: layer, task, probe, accuracy, accuracy_std, f1_macro, f1_std, n
    tasks = [
        ("3-class (A/B/C)", "3-class A/B/C", 0.33),
        ("Binary bypass (A vs non-A)", "Bypass (A vs non-A)", 0.50),
        ("Binary faithful (C vs non-C)", "Error prop (C vs non-C)", 0.50),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    colors = {"linear": "#3a6ea5", "mlp": "#c83a3a"}

    for ax, (csv_task, panel_title, chance) in zip(axes, tasks):
        sub = df[df["task"] == csv_task]
        if sub.empty:
            ax.set_title(f"{panel_title} (no data)", fontsize=10)
            continue
        for probe_name in ("linear", "mlp"):
            d = sub[sub["probe"] == probe_name].sort_values("layer")
            if d.empty:
                continue
            ax.plot(
                d["layer"], d["accuracy"] * 100,
                "-o" if probe_name == "linear" else "--s",
                color=colors[probe_name],
                markersize=4, linewidth=1.5,
                label=f"{probe_name.upper()}",
            )

        ax.axhline(chance * 100, color="gray", linestyle=":", linewidth=1.0, alpha=0.6,
                   label=f"chance ({int(chance * 100)}%)")
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_title(panel_title, fontsize=11)
        ax.set_ylim(20, 100)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Layer-wise probe accuracy on DeepSeek-R1-Distill-Qwen-7B (28 layers; $n = 1{,}194$)",
        fontsize=11.5,
    )
    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    plt.savefig(FIGURES / "deepseek_layer_sweep.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES / "deepseek_layer_sweep.pdf", bbox_inches="tight")
    print(f"Saved {FIGURES / 'deepseek_layer_sweep.png'} and .pdf")
    # Best layer per task
    for csv_task, panel_title, _ in tasks:
        sub = df[df["task"] == csv_task]
        for p in ("linear", "mlp"):
            d = sub[sub["probe"] == p]
            if d.empty:
                continue
            best = d.loc[d["accuracy"].idxmax()]
            print(f"  {panel_title:30s} {p:6s}: best acc = {best['accuracy']*100:.1f}% at layer {best['layer']}")


if __name__ == "__main__":
    main()
