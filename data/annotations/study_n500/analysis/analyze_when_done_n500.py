#!/usr/bin/env python3
"""
Post-annotation analysis for the camera-ready n=500 two-annotator study
(revision item 2.6: n >= 500, two annotators, Cohen's kappa).

Run after BOTH annotators return their filled CSVs. Computes:

  1. Coverage per annotator.
  2. Inter-annotator reliability: raw agreement + Cohen's kappa
     (4-class A/B/C/U; 3-class on rows where both chose A/B/C; and
     binary C vs non-C), overall and per-source.
  3. Consensus labels: rows where both annotators agree. Disagreements
     are exported to `disagreements_for_adjudication.csv`; after the two
     annotators discuss and settle them, pass the filled file back via
     --adjudicated (columns: annotation_id, consensus_label) for the
     final consensus-based numbers.
  4. Human(consensus) vs paper labels: agreement with the paper's final
     labels (`final_label`) and with the LLM-judge subset
     (`judge_label`), Wilson 95% CIs, per-source and per-stratum.
  5. Calibration check on the anchor strata (all-4-variant-agree +
     type_c): consensus-vs-paper agreement should be high; a low number
     flags protocol drift (or, for BBH_type_c, the known rule bug).
  6. Confidence distributions.

Output: console report + `results_n500.json` (+ the adjudication CSV
when there are unresolved disagreements).

Annotator identity is carried by the numbered filename/folder
(annotator1 / annotator2); files come back with their names unchanged.
The join runs on `annotation_id`, so a renamed file still works.

Usage:
  python3 data/annotations/study_n500/analysis/analyze_when_done_n500.py \\
      --annotator1 path/to/data_to_label_annotator1.csv \\
      --annotator2 path/to/data_to_label_annotator2.csv \\
      [--adjudicated path/to/disagreements_adjudicated.csv]
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ---------- small stats helpers ----------

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


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa for two raters over the same items.
    `pairs` = [(label1, label2), ...]. Returns None if undefined."""
    n = len(pairs)
    if n == 0:
        return None
    cats = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    po = sum(1 for a, b in pairs if a == b) / n
    m1 = Counter(a for a, _ in pairs)
    m2 = Counter(b for _, b in pairs)
    pe = sum(m1[c] * m2[c] for c in cats) / (n * n)
    if pe == 1.0:
        return None  # both raters constant — kappa undefined
    return (po - pe) / (1 - pe)


def fmt_kappa(pairs) -> str:
    if not pairs:
        return "—"
    k = cohen_kappa(pairs)
    po = sum(1 for a, b in pairs if a == b) / len(pairs)
    ks = "undef" if k is None else f"{k:.3f}"
    return f"agree {po*100:.1f}%  kappa {ks}  (n={len(pairs)})"


def normalize_label(s: str) -> str:
    s = (s or "").strip().upper()
    if s in ("A", "B", "C", "U"):
        return s
    if "TYPE_A" in s or "BYPASS" in s:
        return "A"
    if "TYPE_B" in s or ("SELF" in s and "CORR" in s):
        return "B"
    if "TYPE_C" in s or "PROPAGAT" in s:
        return "C"
    if "UNCLEAR" in s:
        return "U"
    return ""


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------- pipeline ----------

def join_all(a1_path, a2_path, meta_path):
    meta = {r["annotation_id"]: r for r in load_csv(meta_path)}
    a1 = {r["annotation_id"]: r for r in load_csv(a1_path)}
    a2 = {r["annotation_id"]: r for r in load_csv(a2_path)}
    joined = []
    for aid, m in meta.items():
        r1, r2 = a1.get(aid), a2.get(aid)
        if r1 is None or r2 is None:
            print(f"  warn: annotation_id={aid} missing from an annotator file",
                  file=sys.stderr)
            continue
        joined.append({
            **m,
            "h1": normalize_label(r1.get("human_label", "")),
            "h2": normalize_label(r2.get("human_label", "")),
            "conf1": (r1.get("human_confidence") or "").strip().lower(),
            "conf2": (r2.get("human_confidence") or "").strip().lower(),
            "notes1": r1.get("human_notes", ""),
            "notes2": r2.get("human_notes", ""),
            "_row1": r1,
        })
    return joined


def apply_adjudication(joined, adj_path):
    adj = {r["annotation_id"]: normalize_label(r.get("consensus_label", ""))
           for r in load_csv(adj_path)}
    n_used = 0
    for r in joined:
        lbl = adj.get(r["annotation_id"], "")
        if lbl:
            r["adjudicated"] = lbl
            n_used += 1
    print(f"Adjudication: {n_used} labels applied from {adj_path.name}")


def consensus_of(r) -> str:
    """Consensus = agreed label, else adjudicated label, else ''. """
    if r["h1"] and r["h1"] == r["h2"]:
        return r["h1"]
    return r.get("adjudicated", "")


def export_disagreements(joined, out_path: Path):
    rows = [r for r in joined
            if r["h1"] and r["h2"] and r["h1"] != r["h2"]
            and not r.get("adjudicated")]
    if not rows:
        return 0
    cols = ["annotation_id", "source", "stratum", "annotator1_label",
            "annotator2_label", "annotator1_notes", "annotator2_notes",
            "consensus_label"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in sorted(rows, key=lambda x: int(x["annotation_id"])):
            w.writerow({
                "annotation_id": r["annotation_id"],
                "source": r["source"],
                "stratum": r["stratum"],
                "annotator1_label": r["h1"],
                "annotator2_label": r["h2"],
                "annotator1_notes": r["notes1"],
                "annotator2_notes": r["notes2"],
                "consensus_label": "",
            })
    return len(rows)


def report(joined) -> dict:
    out: dict = {}
    n_total = len(joined)
    print("=" * 70)
    print(f"n=500 TWO-ANNOTATOR AGREEMENT REPORT  (rows joined: {n_total})")
    print("=" * 70)

    # ---- 1. Coverage ----
    for tag in ("h1", "h2"):
        n_lab = sum(1 for r in joined if r[tag])
        dist = Counter(r[tag] for r in joined if r[tag])
        print(f"\nAnnotator {tag[-1]}: {n_lab}/{n_total} labeled "
              f"({n_lab/n_total*100:.1f}%)  {dict(sorted(dist.items()))}")
        out[f"coverage_annotator{tag[-1]}"] = {
            "labeled": n_lab, "total": n_total, "dist": dict(dist)}

    both = [r for r in joined if r["h1"] and r["h2"]]

    # ---- 2. Inter-annotator reliability ----
    print(f"\n{'─' * 70}")
    print("2. INTER-ANNOTATOR RELIABILITY (Cohen's kappa) — the 2.6 headline")
    print(f"{'─' * 70}")
    p4 = [(r["h1"], r["h2"]) for r in both]
    p3 = [(r["h1"], r["h2"]) for r in both
          if r["h1"] in "ABC" and r["h2"] in "ABC"]
    pc = [("C" if r["h1"] == "C" else "X", "C" if r["h2"] == "C" else "X")
          for r in both]
    print(f"  4-class (A/B/C/U):        {fmt_kappa(p4)}")
    print(f"  3-class (A/B/C only):     {fmt_kappa(p3)}")
    print(f"  binary  (C vs non-C):     {fmt_kappa(pc)}")
    out["kappa"] = {
        "four_class": cohen_kappa(p4), "n_four": len(p4),
        "three_class": cohen_kappa(p3), "n_three": len(p3),
        "binary_c": cohen_kappa(pc), "n_binary": len(pc),
        "raw_agree_four": (sum(1 for a, b in p4 if a == b) / len(p4)
                           if p4 else None),
    }
    print("\n  Per-source (4-class):")
    out["kappa_by_source"] = {}
    for src in ("gsm8k", "mmlu", "bbh"):
        ps = [(r["h1"], r["h2"]) for r in both if r["source"] == src]
        print(f"    {src:6s} {fmt_kappa(ps)}")
        out["kappa_by_source"][src] = {
            "kappa": cohen_kappa(ps), "n": len(ps),
            "raw_agree": (sum(1 for a, b in ps if a == b) / len(ps)
                          if ps else None)}
    # inter-annotator confusion matrix
    conf = Counter((a, b) for a, b in p4)
    print("\n  Confusion (rows=annotator1, cols=annotator2):")
    cats = ["A", "B", "C", "U"]
    print("        " + "".join(f"{c:>6s}" for c in cats))
    for a in cats:
        print(f"     {a}  " + "".join(f"{conf.get((a, b), 0):6d}"
                                      for b in cats))
    out["confusion_a1_a2"] = {f"{a}|{b}": v for (a, b), v in conf.items()}

    # ---- 3. Consensus ----
    print(f"\n{'─' * 70}")
    print("3. CONSENSUS LABELS")
    print(f"{'─' * 70}")
    n_agree = sum(1 for r in both if r["h1"] == r["h2"])
    n_adj = sum(1 for r in both if r["h1"] != r["h2"]
                and r.get("adjudicated"))
    n_open = sum(1 for r in both if r["h1"] != r["h2"]
                 and not r.get("adjudicated"))
    print(f"  agreed: {n_agree}   adjudicated: {n_adj}   unresolved: {n_open}")
    out["consensus"] = {"agreed": n_agree, "adjudicated": n_adj,
                        "unresolved": n_open}
    withc = [r for r in both if consensus_of(r)]

    # ---- 4. Consensus vs paper labels ----
    print(f"\n{'─' * 70}")
    print("4. HUMAN (consensus) vs PAPER LABELS  (Wilson 95% CIs)")
    print(f"{'─' * 70}")

    def vs_paper(rows, field):
        sub = [r for r in rows if normalize_label(r.get(field, ""))]
        k = sum(1 for r in sub
                if consensus_of(r) == normalize_label(r[field]))
        return k, len(sub)

    k, n = vs_paper(withc, "final_label")
    print(f"  consensus ↔ paper final_label:  {fmt_pct_ci(k, n)}")
    out["consensus_vs_final"] = {"agree": k, "n": n}
    kj, nj = vs_paper([r for r in withc if r["judge_label"]], "judge_label")
    print(f"  consensus ↔ LLM judge (subset): {fmt_pct_ci(kj, nj)}")
    out["consensus_vs_judge"] = {"agree": kj, "n": nj}

    for tag in ("h1", "h2"):
        sub = [r for r in both if r[tag]]
        ka = sum(1 for r in sub
                 if r[tag] == normalize_label(r["final_label"]))
        print(f"  annotator{tag[-1]} ↔ paper final_label: "
              f"{fmt_pct_ci(ka, len(sub))}")
        out[f"annotator{tag[-1]}_vs_final"] = {"agree": ka, "n": len(sub)}

    print("\n  Per-source (consensus ↔ final_label):")
    out["by_source"] = {}
    for src in ("gsm8k", "mmlu", "bbh"):
        k, n = vs_paper([r for r in withc if r["source"] == src],
                        "final_label")
        print(f"    {src:6s} {fmt_pct_ci(k, n)}")
        out["by_source"][src] = {"agree": k, "n": n}

    print("\n  Per-stratum (consensus ↔ final_label):")
    out["by_stratum"] = {}
    by_str = defaultdict(list)
    for r in withc:
        by_str[r["stratum"]].append(r)
    for s in sorted(by_str):
        k, n = vs_paper(by_str[s], "final_label")
        print(f"    {s:18s} {fmt_pct_ci(k, n)}")
        out["by_stratum"][s] = {"agree": k, "n": n}

    print("\n  Per variant-agreement tier (consensus ↔ final_label):")
    out["by_variant_tier"] = {}
    by_tier = defaultdict(list)
    for r in withc:
        by_tier[r["variant_agreement"]].append(r)
    for t in sorted(by_tier):
        k, n = vs_paper(by_tier[t], "final_label")
        print(f"    {t:14s} {fmt_pct_ci(k, n)}")
        out["by_variant_tier"][t] = {"agree": k, "n": n}

    # ---- 5. Calibration on anchor strata ----
    print(f"\n{'─' * 70}")
    print("5. CALIBRATION on anchor strata (agree-tier + type_c)")
    print(f"{'─' * 70}")
    out["calibration"] = {}
    anchors = [s for s in by_str
               if s.endswith("_agree") or s.endswith("_type_c")]
    for s in sorted(anchors):
        k, n = vs_paper(by_str[s], "final_label")
        line = f"  {s:18s} {fmt_pct_ci(k, n)}"
        if n and k / n < 0.9:
            line += ("   ⚠️  below 90%"
                     + (" (known rule bug — expected)"
                        if s == "BBH_type_c" else " — protocol drift?"))
        print(line)
        out["calibration"][s] = {"agree": k, "n": n}

    # ---- 6. Confidence ----
    print(f"\n{'─' * 70}")
    print("6. CONFIDENCE DISTRIBUTIONS")
    print(f"{'─' * 70}")
    out["confidence"] = {}
    for tag in ("conf1", "conf2"):
        dist = Counter(r[tag] for r in both if r[tag])
        tot = sum(dist.values()) or 1
        pretty = {k: f"{v} ({v/tot*100:.0f}%)"
                  for k, v in sorted(dist.items())}
        print(f"  annotator{tag[-1]}: {pretty}")
        out["confidence"][f"annotator{tag[-1]}"] = dict(dist)

    return out


def main():
    ap = argparse.ArgumentParser(
        description="n=500 two-annotator agreement analysis (item 2.6)")
    ap.add_argument("--annotator1", type=Path, required=True,
                    help="Annotator 1's filled CSV")
    ap.add_argument("--annotator2", type=Path, required=True,
                    help="Annotator 2's filled CSV")
    ap.add_argument("--adjudicated", type=Path, default=None,
                    help="Filled disagreements_for_adjudication.csv "
                         "(annotation_id, consensus_label)")
    ap.add_argument("--metadata", type=Path,
                    default=HERE / "_analysis_metadata.csv")
    ap.add_argument("--out", type=Path, default=HERE / "results_n500.json")
    args = ap.parse_args()

    joined = join_all(args.annotator1, args.annotator2, args.metadata)
    print(f"Joined {len(joined)} rows.\n")
    if args.adjudicated:
        apply_adjudication(joined, args.adjudicated)

    results = report(joined)

    adj_out = HERE / "disagreements_for_adjudication.csv"
    n_dis = export_disagreements(joined, adj_out)
    if n_dis:
        print(f"\n{n_dis} unresolved disagreements written to {adj_out.name}.")
        print("Have the annotators settle them together, fill the "
              "consensus_label column, then re-run with --adjudicated.")

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
