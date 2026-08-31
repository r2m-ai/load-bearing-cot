#!/usr/bin/env python3
"""Scatter: base accuracy vs error propagation (Gemma + Llama). Values match paper tables."""
from pathlib import Path

import matplotlib.pyplot as plt

# (dataset label, base_acc %, error_prop %)
GEMMA = [("GSM8K", 86.5, 3.9), ("MMLU", 74.8, 22.2), ("BBH", 57.9, 79.0)]
LLAMA = [("GSM8K", 55.3, 12.4), ("MMLU", 40.8, 62.8), ("BBH", 40.1, 65.5)]

# Okabe–Ito (colorblind-friendly)
C_GEMMA = "#0072B2"
C_LLAMA = "#D55E00"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "figures" / "accuracy_vs_errorprop.png"

    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=200)

    for name, pts, c, m in (
        ("Gemma-2-9B-IT", GEMMA, C_GEMMA, "o"),
        ("Llama-3.1-8B-Instruct", LLAMA, C_LLAMA, "s"),
    ):
        xa = [t[1] for t in pts]
        ya = [t[2] for t in pts]
        ax.scatter(
            xa,
            ya,
            c=c,
            s=85,
            marker=m,
            label=name,
            zorder=3,
            edgecolors="white",
            linewidths=0.8,
        )
        for ds, x, y in pts:
            ax.annotate(
                ds,
                (x, y),
                textcoords="offset points",
                xytext=(6, 4 if ds != "BBH" else -12),
                fontsize=8,
                color=c,
            )

    ax.set_xlabel("Model accuracy on unperturbed questions (%)")
    ax.set_ylabel("Error propagation under perturbation (%)")
    ax.set_xlim(35, 92)
    ax.set_ylim(0, 88)
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
