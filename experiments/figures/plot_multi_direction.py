"""Generate figures/multidir_steering.png — per-type flip rate vs alpha across
(basis, k) combos for exp_4_2 (multi-direction steering)."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "results/steering/multi_direction/multidir_steering_subclassified.json"
OUT = REPO / "figures/multidir_steering.png"

records = json.load(open(DATA))["records"]
print(f"loaded {len(records)} records")

# flip = baseline subtype != final subtype, conditional on a clear final label
# Treat NEEDS_JUDGE / UNCLEAR as "did not resolve" -> drop for flip rate
def is_flip(r):
    b = r["baseline_subtype"]
    f = r["subtype"]
    if f in ("NEEDS_JUDGE", "UNCLEAR"):
        return None
    return b != f

# Aggregate: (basis, k, alpha, baseline_type) -> [resolved_count, flip_count]
agg = defaultdict(lambda: [0, 0])
for r in records:
    f = is_flip(r)
    if f is None:
        continue
    key = (r["basis"], r["k"], r["alpha"], r["baseline_subtype"])
    agg[key][0] += 1
    agg[key][1] += int(f)

# Wilson CI
def wilson(p, n, z=1.96):
    if n == 0:
        return 0, 0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    halfwidth = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, center - halfwidth), min(1, center + halfwidth)

# Build per-type figures
types = ["TYPE_A", "TYPE_B", "TYPE_C"]
type_titles = {
    "TYPE_A": "Type A (bypass)",
    "TYPE_B": "Type B (self-correction)",
    "TYPE_C": "Type C (error propagation)",
}
bases = ["M2", "probe"]
ks = [2, 4, 8]
alphas = sorted({k_[2] for k_ in agg.keys()})

# Color = basis, linestyle = k
basis_colors = {"M2": "#1f77b4", "probe": "#d62728"}
k_styles = {2: ":", 4: "--", 8: "-"}
k_lw = {2: 1.4, 4: 1.6, 8: 2.2}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)

for ax, t in zip(axes, types):
    for basis in bases:
        for k in ks:
            xs, ys, lo, hi = [], [], [], []
            for a in alphas:
                key = (basis, k, a, t)
                n, f = agg.get(key, [0, 0])
                if n == 0:
                    continue
                p = f / n
                cl, ch = wilson(p, n)
                xs.append(a)
                ys.append(p * 100)
                lo.append(cl * 100)
                hi.append(ch * 100)
            xs, ys, lo, hi = map(np.array, (xs, ys, lo, hi))
            label = f"{basis}, k={k}"
            ax.plot(
                xs, ys,
                color=basis_colors[basis],
                linestyle=k_styles[k],
                linewidth=k_lw[k],
                label=label,
                marker="o" if k == 8 else None,
                markersize=4 if k == 8 else 0,
                alpha=0.85 if k == 8 else 0.65,
            )
            # Show CI band only for k=8 (the headline curves)
            if k == 8:
                ax.fill_between(xs, lo, hi, color=basis_colors[basis], alpha=0.12)

    ax.set_title(type_titles[t], fontsize=11)
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.grid(True, alpha=0.25)

axes[0].set_ylabel("flip rate (%)\nbaseline subtype $\\to$ different subtype")
axes[0].set_ylim(-2, 35)

# Annotate the headline cell on TYPE_C panel
type_c_ax = axes[2]
# Find (probe, k=8, alpha=-10, TYPE_C)
n_c, f_c = agg[("probe", 8, -10.0, "TYPE_C")]
peak = 100 * f_c / n_c if n_c else 0
type_c_ax.annotate(
    f"probe k=8, $\\alpha=-10$\n{peak:.1f}% flip (n={n_c})",
    xy=(-10, peak),
    xytext=(-8.5, peak + 4),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
)

# Single legend, top of figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    ncol=6,
    bbox_to_anchor=(0.5, 1.05),
    frameon=False,
    fontsize=9,
)

plt.suptitle(
    "Multi-direction steering: per-type flip rate at layer 21 "
    "across two $k$-vector bases (M2 = class-mean + within-C PCs; "
    "probe = multinomial probe directions)",
    fontsize=10, y=1.13,
)
plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"saved -> {OUT}")

# Print key cells for sanity check
print("\n=== Key headline cells ===")
for key in [
    ("probe", 8, -10.0, "TYPE_C"),
    ("M2", 8, 10.0, "TYPE_C"),
    ("probe", 8, -10.0, "TYPE_A"),
    ("M2", 8, -10.0, "TYPE_A"),
    ("probe", 8, -10.0, "TYPE_B"),
]:
    n, f = agg.get(key, [0, 0])
    p = 100 * f / n if n else 0
    print(f"  {key}: flip={f}/{n} = {p:.1f}%")
