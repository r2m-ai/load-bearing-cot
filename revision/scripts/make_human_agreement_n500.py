"""figures/human_agreement.png -- n=500 two-annotator human-validation study.

Left : per-source agreement of the human consensus (499 rows: 487 agreed +
       12 adjudicated in favour of the annotator matching the production label) with the paper's production label, three-way (A/B/C)
       and binary (C vs non-C); Cohen's kappa per source above the bars.
Right: per-source agreement of each judge prompt variant (and the production
       label) with the human consensus on the 407 agreed rows covered by the
       four-variant re-judging (revision item 2.0).

Reads revision/human_annotation_n500/analysis/merged_labels_n500.csv.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "revision/human_annotation_n500/analysis/merged_labels_n500.csv"
OUT = REPO / "figures/human_agreement.png"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    c = (p + z * z / (2 * n)) / denom
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, c - h), min(1.0, c + h)


def kappa(a, b):
    n = len(a)
    po = np.mean(np.array(a) == np.array(b))
    cats = set(a) | set(b)
    pe = sum(a.count(c) * b.count(c) for c in cats) / (n * n)
    return (po - pe) / (1 - pe)


rows = list(csv.DictReader(open(SRC)))
# use the adjudicated consensus where present (falls back to pre-adjudication)
for r in rows:
    if r.get("consensus_label_adjudicated"):
        r["consensus_label_pre_adjudication"] = r["consensus_label_adjudicated"]
SOURCES = ["gsm8k", "mmlu", "bbh"]
LABELS = ["GSM8K", "MMLU", "BBH"]
VARIANTS = ["original", "reference_sensitive", "strict_bypass", "strict_correction"]

# three-way: all agreed rows (A/B/C/U; U never matches a paper label) -- same
# denominator as analyze_when_done_n500.py; binary: agreed non-U rows.
agreed = [r for r in rows if r["consensus_label_pre_adjudication"] in ("A", "B", "C", "U")]
stats = {}
for s in SOURCES:
    rs = [r for r in agreed if r["source"] == s]
    rb = [r for r in rs if r["consensus_label_pre_adjudication"] != "U"]
    n3, nb = len(rs), len(rb)
    k3 = sum(r["consensus_label_pre_adjudication"] == r["final_label"] for r in rs)
    kb = sum((r["consensus_label_pre_adjudication"] == "C") == (r["final_label"] == "C") for r in rb)
    allrows = [r for r in rows if r["source"] == s]
    kap = kappa([r["annotator1_label"] for r in allrows], [r["annotator2_label"] for r in allrows])
    cov = [r for r in rs if r["variant_labels"]]
    var = {}
    for v in VARIANTS:
        hits = 0
        for r in cov:
            lab = dict(kv.split("=") for kv in r["variant_labels"].split("|"))
            hits += lab.get(v) == r["consensus_label_pre_adjudication"]
        var[v] = (hits, len(cov))
    var["production"] = (sum(r["consensus_label_pre_adjudication"] == r["final_label"] for r in cov), len(cov))
    stats[s] = dict(n3=n3, nb=nb, k3=k3, kb=kb, kappa=kap, var=var)
    print(f"{s}: 3-way {k3}/{n3}={k3/n3:.3f}  binary {kb}/{nb}={kb/nb:.3f}  kappa={kap:.3f}  "
          + "  ".join(f"{v} {h}/{n}={h/n:.3f}" for v, (h, n) in var.items()))

k_all = kappa([r["annotator1_label"] for r in rows], [r["annotator2_label"] for r in rows])
n_all = len(agreed)
nb_all = sum(r["consensus_label_pre_adjudication"] != "U" for r in agreed)
k3_all = sum(r["consensus_label_pre_adjudication"] == r["final_label"] for r in agreed)
kb_all = sum((r["consensus_label_pre_adjudication"] == "C") == (r["final_label"] == "C") for r in agreed if r["consensus_label_pre_adjudication"] != "U")
print(f"overall: kappa={k_all:.3f}  3-way {k3_all}/{n_all}={k3_all/n_all:.3f}  binary {kb_all}/{nb_all}={kb_all/nb_all:.3f}")

C3, CB = "#4C72B0", "#55A868"
VCOL = {"original": "#4C72B0", "reference_sensitive": "#8172B2",
        "strict_bypass": "#C44E52", "strict_correction": "#55A868", "production": "#7F7F7F"}
VNAME = {"original": "original (production judge prompt)", "reference_sensitive": "reference-sensitive",
         "strict_bypass": "strict bypass", "strict_correction": "strict correction",
         "production": "paper label (rule + original judge)"}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1, 1.35]})

# ---- left panel
x = np.arange(3)
w = 0.36
for i, s in enumerate(SOURCES):
    st = stats[s]
    for off, k, n, col, lab in [(-w / 2, st["k3"], st["n3"], C3, "three-way (A/B/C)"), (w / 2, st["kb"], st["nb"], CB, "binary (C vs non-C)")]:
        p = k / n
        lo, hi = wilson(k, n)
        axL.bar(i + off, p * 100, w, color=col, alpha=0.85,
                yerr=[[max(0.0, p - lo) * 100], [max(0.0, hi - p) * 100]], capsize=3,
                error_kw=dict(lw=1, alpha=0.7), label=lab if i == 0 else None)
        axL.text(i + off, hi * 100 + 1.5, f"{p*100:.0f}%", ha="center", va="bottom", fontsize=8)
    axL.text(i, 106, f"$\\kappa$ = {st['kappa']:.2f}", ha="center", va="bottom", fontsize=8.5,
             color="#333333")
axL.axhline(k3_all / n_all * 100, color=C3, ls="--", lw=1, alpha=0.35)
axL.axhline(kb_all / nb_all * 100, color=CB, ls="--", lw=1, alpha=0.35)
axL.set_xticks(x); axL.set_xticklabels([f"{l}\n($n$={stats[s]['n3']})" for l, s in zip(LABELS, SOURCES)])
axL.set_ylabel("Human consensus $\\leftrightarrow$ paper label (%)")
axL.set_ylim(0, 118); axL.set_yticks([0, 25, 50, 75, 100])
axL.grid(True, alpha=0.25, axis="y")
axL.spines["top"].set_visible(False); axL.spines["right"].set_visible(False)
axL.legend(loc="lower left", fontsize=8, framealpha=0.95)
axL.set_title(f"(a) Agreement with the paper's labels\n(overall {k3_all/n_all*100:.1f}% three-way, "
              f"{kb_all/nb_all*100:.1f}% binary; inter-annotator $\\kappa$ = {k_all:.2f})", fontsize=9.5)

# ---- right panel
order = ["strict_correction", "production", "original", "reference_sensitive", "strict_bypass"]
w2 = 0.15
for j, v in enumerate(order):
    for i, s in enumerate(SOURCES):
        h, n = stats[s]["var"][v]
        p = h / n
        lo, hi = wilson(h, n)
        axR.bar(i + (j - 2) * w2, p * 100, w2, color=VCOL[v], alpha=0.85,
                yerr=[[max(0.0, p - lo) * 100], [max(0.0, hi - p) * 100]], capsize=2,
                error_kw=dict(lw=0.8, alpha=0.6), label=VNAME[v] if i == 0 else None)
        axR.text(i + (j - 2) * w2, hi * 100 + 1.5, f"{p*100:.0f}", ha="center", va="bottom", fontsize=6.8)
axR.set_xticks(x); axR.set_xticklabels([f"{l}\n($n$={stats[s]['var']['original'][1]})" for l, s in zip(LABELS, SOURCES)])
axR.set_ylabel("Agreement with human consensus (%)")
axR.set_ylim(0, 118); axR.set_yticks([0, 25, 50, 75, 100])
axR.grid(True, alpha=0.25, axis="y")
axR.spines["top"].set_visible(False); axR.spines["right"].set_visible(False)
axR.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.2, framealpha=0.95)
axR.set_title("(b) Which judge prompt matches human judgment?\n(rows covered by the four-variant re-judging)", fontsize=9.5)

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"saved -> {OUT}")
