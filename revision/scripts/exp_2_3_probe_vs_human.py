"""
exp_2_3_probe_vs_human.py — probe-vs-human agreement on the n=200
annotation subset (item 2.3, gated on item 2.1 human-label completion).

What this script does:
  1. Loads the N=2,000 feature subset (extracted_features_42layers.npz)
     and the corresponding subclassified_pairs.json labels.
  2. Trains a bypass probe (A vs non-A) at layer 10 — the best-performing
     binary-bypass layer in the main paper (Table~3).
  3. Joins the human-annotation CSV against subclassified_pairs.json by
     composite key (id, perturbation_strategy, perturbation_point).
  4. For the subset of annotation rows that are also in the N=2,000
     training set, retrieves the *held-out* probe prediction from a
     grouped 5-fold CV (so the prediction is genuinely held-out).
  5. For annotation rows NOT in the N=2,000 training set, attempts to
     retrieve features from extracted_features_42layers.npz; if not
     present (the default), the row is reported as "feature_missing"
     and an explicit caveat is printed at the end.
  6. Computes probe-vs-human agreement (binary: bypass vs non-bypass)
     with Wilson 95% CI, broken out by source and overall.

This script is safe to run repeatedly. It does NOT require GPU.

Usage:
  python revision/scripts/exp_2_3_probe_vs_human.py
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
ANNOT = ROOT / "revision" / "human_annotation" / "data_to_label.csv"
META = ROOT / "revision" / "human_annotation" / "_analysis_metadata.csv"
OUTDIR = ROOT / "revision" / "revision_exp_results" / "exp_2_3_probe_vs_human"
OUTDIR.mkdir(parents=True, exist_ok=True)
FEATURES = DATA / "extracted_features_42layers.npz"
ANNOT_FEATURES = DATA / "extracted_features_annot200.npz"
LAYER = 10  # best bypass layer per paper Table 3

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def label_of(d: dict) -> str:
    jl = d.get("judge_label")
    if jl in {"A", "B", "C"}:
        return jl
    lc = d.get("label_3class", "")
    if lc == "error_propagation":
        return "C"
    if lc == "silent_bypass":
        return "A"
    if lc == "self_correction":
        return "B"
    return "unclear"


def composite(rec: dict) -> tuple:
    return (
        rec.get("id"),
        rec.get("perturbation_strategy"),
        rec.get("perturbation_point"),
    )


# ---------------------------------------------------------------------------
# Step 1: Load features + labels, train held-out probe via grouped CV
# ---------------------------------------------------------------------------

def train_holdout_probe(layer: int = LAYER) -> dict:
    """Return per-record held-out predictions for the N=2,000 training set."""
    print(f"\n[1] Loading features at layer {layer} and training bypass probe...")
    feats_npz = np.load(FEATURES, allow_pickle=True)
    X = feats_npz[f"layer_{layer}"]   # shape (2000, 10752)
    pair_ids = feats_npz["pair_ids"]
    pert_points = feats_npz["perturbation_points"]
    print(f"   features: shape {X.shape}, dtype {X.dtype}")

    # Build lookup from full subclassified_pairs.json keyed by (id, perturbation_point).
    # The features-file rows are NOT in the same order as subclassified_pairs.json —
    # they were sampled separately at extraction time. Looking up by composite key is
    # how scripts/10b_retrain_probes.py (the paper's training script) does it.
    all_pairs = json.load(open(DATA / "subclassified_pairs.json"))
    lookup = {(p["id"], p.get("perturbation_point", "")): p for p in all_pairs}

    # Per-feature-row records aligned to features index
    sub = []
    for i in range(X.shape[0]):
        key = (str(pair_ids[i]), str(pert_points[i]))
        sub.append(lookup.get(key, {}))
    print(f"   subclassified rows aligned to features: {sum(1 for s in sub if s)}/{len(sub)}")

    # Labels: binary bypass (A vs non-A). A = silent_bypass.
    labels = []
    groups = []
    keep_idx = []
    for i, r in enumerate(sub):
        lab = label_of(r)
        if lab not in {"A", "B", "C"}:
            continue
        labels.append(1 if lab == "A" else 0)
        groups.append(r.get("id"))
        keep_idx.append(i)

    X_kept = X[keep_idx]
    y = np.array(labels, dtype=int)
    g = np.array(groups)
    print(f"   kept (A/non-A clear): {len(y)}")
    print(f"   class balance: bypass={y.sum()}, non-bypass={(y == 0).sum()}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_kept)

    # Grouped 5-fold CV; collect held-out predictions
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pred_label = np.zeros(len(y), dtype=int)
    pred_prob = np.zeros(len(y))
    for fold, (tr, te) in enumerate(sgkf.split(X_scaled, y, groups=g), 1):
        clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
        clf.fit(X_scaled[tr], y[tr])
        pred_label[te] = clf.predict(X_scaled[te])
        pred_prob[te] = clf.predict_proba(X_scaled[te])[:, 1]
        acc = (pred_label[te] == y[te]).mean()
        print(f"   fold {fold}: held-out acc = {acc:.3f} (n={len(te)})")
    overall = (pred_label == y).mean()
    print(f"   overall held-out acc: {overall:.3f}")

    # Train final probe on all data for use on out-of-training features
    final_clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)
    final_clf.fit(X_scaled, y)

    # Map composite key → held-out prediction (for training-set rows)
    composite_to_pred: dict[tuple, dict] = {}
    for j, i in enumerate(keep_idx):
        ck = composite(sub[i])
        composite_to_pred[ck] = {
            "in_training": True,
            "probe_label": "A" if pred_label[j] == 1 else "non-A",
            "probe_prob_A": float(pred_prob[j]),
            "ground_truth": "A" if y[j] == 1 else "non-A",
        }

    return {
        "composite_to_pred": composite_to_pred,
        "scaler": scaler,
        "final_clf": final_clf,
        "overall_acc": float(overall),
        "n_training": int(len(y)),
    }


def predict_on_annot_features(probe: dict, layer: int = LAYER) -> dict:
    """Apply the final probe (fit on all N=2,000) to the n=200 annot features.
    Returns composite_key → prediction dict for rows NOT already in the
    training set (overlap rows keep their grouped-CV held-out predictions).
    """
    if not ANNOT_FEATURES.exists():
        print(f"   (no {ANNOT_FEATURES.name} — non-overlap predictions skipped; "
              "run revision/scripts/exp_2_3_extract_annot_features.py on a GPU pod)")
        return {}
    af = np.load(ANNOT_FEATURES, allow_pickle=True)
    X = af[f"layer_{layer}"]
    pair_ids = af["pair_ids"]
    strategies = af["perturbation_strategies"]
    points = af["perturbation_points"]

    X_scaled = probe["scaler"].transform(X)
    pred_lab = probe["final_clf"].predict(X_scaled)
    pred_prob = probe["final_clf"].predict_proba(X_scaled)[:, 1]

    composite_to_pred: dict[tuple, dict] = {}
    for i in range(len(pair_ids)):
        ck = (str(pair_ids[i]), str(strategies[i]), str(points[i]))
        # Don't overwrite in-training predictions — those are grouped-CV held-out
        if ck in probe["composite_to_pred"]:
            continue
        composite_to_pred[ck] = {
            "in_training": False,
            "probe_label": "A" if pred_lab[i] == 1 else "non-A",
            "probe_prob_A": float(pred_prob[i]),
        }
    print(f"   non-overlap rows predicted via final probe: {len(composite_to_pred)}")
    return composite_to_pred


# ---------------------------------------------------------------------------
# Step 2: Load annotations + join to predictions
# ---------------------------------------------------------------------------

def load_annotations() -> tuple[list[dict], list[dict]]:
    with open(ANNOT) as f:
        annot = list(csv.DictReader(f))
    with open(META) as f:
        meta = list(csv.DictReader(f))
    by_id = {m["annotation_id"]: m for m in meta}
    for r in annot:
        m = by_id.get(r["annotation_id"], {})
        r["_meta"] = m
        r["_composite"] = (
            m.get("example_id"),
            m.get("strategy"),
            m.get("perturbation_point"),
        )
    return annot, meta


def human_to_bypass(label: str) -> str | None:
    """Map human label {A, B, C, U} to binary {A, non-A}. Returns None if unfilled or U."""
    if not label or label == "U":
        return None
    if label == "A":
        return "A"
    if label in {"B", "C"}:
        return "non-A"
    return None


# ---------------------------------------------------------------------------
# Step 3: Evaluate
# ---------------------------------------------------------------------------

def evaluate(probe: dict, annot: list[dict]) -> dict:
    # Merge in-training (grouped-CV held-out) predictions with non-overlap
    # (final-probe-on-annot200-features) predictions
    composite_to_pred = dict(probe["composite_to_pred"])
    non_overlap = predict_on_annot_features(probe, layer=LAYER)
    composite_to_pred.update(non_overlap)

    rows = []
    for r in annot:
        ck = r["_composite"]
        hum = human_to_bypass(r.get("human_label", "").strip())
        pred_info = composite_to_pred.get(ck)
        rows.append({
            "annotation_id": r["annotation_id"],
            "source": r.get("source"),
            "stratum": r["_meta"].get("stratum"),
            "composite": ck,
            "human_label": r.get("human_label", "").strip() or None,
            "human_binary": hum,
            "in_training": pred_info["in_training"] if pred_info else None,
            "probe_label": pred_info["probe_label"] if pred_info else None,
            "probe_prob_A": pred_info["probe_prob_A"] if pred_info else None,
        })

    # Summary
    n_total = len(rows)
    n_annotated = sum(1 for r in rows if r["human_binary"] is not None)
    n_in_training = sum(1 for r in rows if r["in_training"] is True)
    n_with_pred = sum(1 for r in rows if r["probe_label"] is not None)
    n_eval = sum(1 for r in rows if r["probe_label"] is not None and r["human_binary"] is not None)

    print(f"\n[3] Coverage:")
    print(f"   total annotation rows: {n_total}")
    print(f"   with human label filled (and not U): {n_annotated}")
    print(f"   in N=2,000 training set (held-out CV pred): {n_in_training}")
    print(f"   non-overlap (final-probe pred on annot200 features): {n_with_pred - n_in_training}")
    print(f"   evaluable (probe-pred AND human-label): {n_eval}")

    # Agreement on evaluable rows (all annot rows with both pred + human label)
    agree_rows = [r for r in rows if r["probe_label"] is not None and r["human_binary"] is not None]
    if not agree_rows:
        print("\n[!] No annotation rows are evaluable yet — fill human_label or extract annot features.")
    else:
        n = len(agree_rows)
        agree = sum(1 for r in agree_rows if r["probe_label"] == r["human_binary"])
        lo, hi = wilson(agree, n)
        print(f"\n[4] Probe-vs-human agreement (binary bypass)")
        print(f"   n = {n}; agreement = {agree}/{n} = {agree/n*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]")

        # trivial-A baseline on the same rows
        n_human_A = sum(1 for r in agree_rows if r["human_binary"] == "A")
        print(f"   trivial-always-A baseline on same n: {n_human_A}/{n} = {n_human_A/n*100:.1f}%")

        by_src = defaultdict(list)
        for r in agree_rows:
            by_src[r["source"]].append((r["probe_label"] == r["human_binary"], r["human_binary"] == "A"))
        print("   by source:")
        for s, pairs in sorted(by_src.items()):
            nn = len(pairs)
            kk = sum(1 for ok, _ in pairs if ok)
            triv = sum(1 for _, isA in pairs if isA)
            lo_s, hi_s = wilson(kk, nn)
            print(f"     {s}: probe={kk}/{nn}={kk/nn*100:.1f}% [{lo_s*100:.1f}, {hi_s*100:.1f}]  "
                  f"trivial-A={triv}/{nn}={triv/nn*100:.1f}%")

        # in-training vs non-overlap breakdown
        in_tr = [r for r in agree_rows if r["in_training"] is True]
        out_tr = [r for r in agree_rows if r["in_training"] is False]
        for label, sub in [("in-training (held-out)", in_tr), ("non-overlap (final-probe)", out_tr)]:
            if not sub:
                continue
            nn = len(sub)
            kk = sum(1 for r in sub if r["probe_label"] == r["human_binary"])
            lo_s, hi_s = wilson(kk, nn)
            print(f"   {label}: {kk}/{nn} = {kk/nn*100:.1f}% [{lo_s*100:.1f}, {hi_s*100:.1f}]")

    # Persist outputs
    out = {
        "n_annotation_rows": n_total,
        "n_annotated": n_annotated,
        "n_in_training": n_in_training,
        "n_eval": n_eval,
        "probe_layer": LAYER,
        "probe_holdout_accuracy_on_training": probe["overall_acc"],
        "rows": rows,
    }
    if agree_rows:
        n = len(agree_rows)
        agree = sum(1 for r in agree_rows if r["probe_label"] == r["human_binary"])
        lo, hi = wilson(agree, n)
        out["agreement_overall"] = {
            "n": n,
            "agree": agree,
            "rate": agree / n,
            "wilson_ci95": [lo, hi],
        }

    (OUTDIR / "results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUTDIR / 'results.json'}")

    # Diagnostic CSV
    out_csv = OUTDIR / "per_row.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["annotation_id", "source", "stratum", "in_training",
                    "human_label", "human_binary", "probe_label", "probe_prob_A"])
        for r in rows:
            w.writerow([r["annotation_id"], r["source"], r["stratum"],
                        r["in_training"], r["human_label"], r["human_binary"],
                        r["probe_label"],
                        f"{r['probe_prob_A']:.3f}" if r["probe_prob_A"] is not None else ""])
    print(f"Wrote {out_csv}")

    return out


def main() -> None:
    global ANNOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", type=Path, default=ANNOT,
                    help="Path to the annotator's filled CSV (default: data_to_label.csv)")
    args = ap.parse_args()
    ANNOT = args.annotated
    print(f"Using annotated file: {ANNOT}")

    probe = train_holdout_probe(layer=LAYER)
    annot, meta = load_annotations()
    print(f"\n[2] Loaded {len(annot)} annotation rows and {len(meta)} metadata rows")
    evaluate(probe, annot)



if __name__ == "__main__":
    main()
