"""
Script 10b: Re-train probes locally using saved feature arrays.

No GPU needed — loads pre-extracted features from extracted_features_42layers.npz
and re-reads labels from subclassified_pairs.json (which may have been updated
by the LLM judge).

Usage: python3 experiments/probes/retrain_probes.py
"""

import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def train_eval_probe(X, y, groups=None, probe_type="linear", n_folds=5, use_pca=128):
    if len(X) < 20 or len(np.unique(y)) < 2:
        return {"accuracy": 0, "accuracy_std": 0, "f1_macro": 0, "n": len(X)}

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    if use_pca and X_s.shape[1] > use_pca:
        X_s = PCA(n_components=min(use_pca, X_s.shape[0] - 1)).fit_transform(X_s)

    min_class = min(Counter(y).values())
    folds = min(n_folds, min_class)
    if folds < 2:
        folds = 2

    if probe_type == "linear":
        model = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight="balanced")
    elif probe_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(256,), max_iter=1000,
                              random_state=42, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=10)
    else:
        raise ValueError(f"Unknown: {probe_type}")

    if groups is not None:
        n_groups = len(np.unique(groups))
        folds = min(folds, n_groups)
        if folds < 2:
            folds = 2
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
        acc = cross_val_score(model, X_s, y, cv=cv, groups=groups, scoring="accuracy")
        f1 = cross_val_score(model, X_s, y, cv=cv, groups=groups, scoring="f1_macro")
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        acc = cross_val_score(model, X_s, y, cv=skf, scoring="accuracy")
        f1 = cross_val_score(model, X_s, y, cv=skf, scoring="f1_macro")

    return {"accuracy": acc.mean(), "accuracy_std": acc.std(), "f1_macro": f1.mean(), "n": len(X)}


def main():
    parser = argparse.ArgumentParser(description="Re-train probes from saved features")
    parser.add_argument("--pca-dims", type=int, default=128)
    parser.add_argument("--probe-types", type=str, default="linear,mlp")
    args = parser.parse_args()

    probe_types = args.probe_types.split(",")
    use_pca = args.pca_dims if args.pca_dims > 0 else None

    # Load saved features
    features_path = PROCESSED_DIR / "extracted_features_42layers.npz"
    print(f"Loading features from {features_path}...")
    data = np.load(features_path, allow_pickle=True)

    pair_ids = data["pair_ids"]
    perturbation_points = data["perturbation_points"]
    n_layers = sum(1 for k in data.files if k.startswith("layer_"))
    # question_ids for Grouped-CV (same base question → same group)
    if "question_ids" in data.files:
        all_question_ids = data["question_ids"]
    else:
        # Fallback: pair_ids are already question-level IDs (e.g., gsm8k_49)
        all_question_ids = pair_ids
    print(f"Loaded: {len(pair_ids)} examples, {n_layers} layers")

    # Load current labels (may have been updated by judge)
    with open(PROCESSED_DIR / "subclassified_pairs.json") as f:
        all_pairs = json.load(f)

    # Build lookup: (id, perturbation_point) -> pair
    pair_lookup = {}
    for p in all_pairs:
        key = (p["id"], p.get("perturbation_point", ""))
        pair_lookup[key] = p

    # Match features to current labels
    labels_3class = []
    labels_bypass = []
    labels_faithful = []
    valid_mask = []

    for i in range(len(pair_ids)):
        key = (str(pair_ids[i]), str(perturbation_points[i]))
        pair = pair_lookup.get(key)

        if pair is None or pair.get("label_3class") not in ("silent_bypass", "self_correction", "error_propagation"):
            valid_mask.append(False)
            labels_3class.append("unknown")
            labels_bypass.append("unknown")
            labels_faithful.append("unknown")
            continue

        valid_mask.append(True)
        labels_3class.append(pair["label_3class"])
        subtype = pair.get("subtype", "")
        labels_bypass.append("bypass" if subtype == "TYPE_A" else "non_bypass")
        labels_faithful.append("error_prop" if subtype == "TYPE_C" else "non_error_prop")

    valid_mask = np.array(valid_mask)
    question_ids = all_question_ids[valid_mask]
    print(f"Valid examples: {valid_mask.sum()} / {len(valid_mask)}")
    print(f"Unique questions for Grouped-CV: {len(np.unique(question_ids))}")

    le_3class = LabelEncoder()
    y_3class = le_3class.fit_transform([l for l, v in zip(labels_3class, valid_mask) if v])
    le_bypass = LabelEncoder()
    y_bypass = le_bypass.fit_transform([l for l, v in zip(labels_bypass, valid_mask) if v])
    le_faithful = LabelEncoder()
    y_faithful = le_faithful.fit_transform([l for l, v in zip(labels_faithful, valid_mask) if v])

    tasks = [
        ("3-class (A/B/C)", y_3class, le_3class),
        ("Binary bypass", y_bypass, le_bypass),
        ("Binary faithful", y_faithful, le_faithful),
    ]

    print(f"\nClass distributions:")
    for name, y, le in tasks:
        dist = {le.classes_[i]: int(c) for i, c in enumerate(np.bincount(y))}
        print(f"  {name}: {dist}")

    # Train probes
    print(f"\nTraining probes ({n_layers} layers × {len(probe_types)} types × {len(tasks)} tasks)...")
    results = []

    for layer_idx in tqdm(range(n_layers), desc="Layers"):
        X_all = data[f"layer_{layer_idx}"]
        X = X_all[valid_mask]

        for task_name, y, le in tasks:
            for probe_type in probe_types:
                r = train_eval_probe(X, y, groups=question_ids, probe_type=probe_type, use_pca=use_pca)
                results.append({
                    "layer": layer_idx, "task": task_name, "probe": probe_type,
                    "accuracy": r["accuracy"], "accuracy_std": r["accuracy_std"],
                    "f1_macro": r["f1_macro"], "n": r["n"],
                })

    df = pd.DataFrame(results)

    # Summary
    print(f"\n{'='*70}")
    print("BEST RESULTS PER TASK × PROBE TYPE")
    print(f"{'='*70}")
    for task_name in df["task"].unique():
        print(f"\n{task_name}:")
        for probe_type in probe_types:
            sub = df[(df["task"] == task_name) & (df["probe"] == probe_type)]
            if sub.empty:
                continue
            best = sub.loc[sub["accuracy"].idxmax()]
            print(f"  {probe_type:10s}: layer {int(best['layer']):2d}, "
                  f"acc={best['accuracy']:.3f}±{best['accuracy_std']:.3f}, "
                  f"F1={best['f1_macro']:.3f}")

    # Figures
    sns.set_theme(style="whitegrid", font_scale=1.1)
    colors = {"linear": "#2196F3", "mlp": "#F44336"}
    styles = {"linear": "-", "mlp": "--"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, task_name in zip(axes, df["task"].unique()):
        task_df = df[df["task"] == task_name]
        for probe_type in probe_types:
            sub = task_df[task_df["probe"] == probe_type].sort_values("layer")
            ax.plot(sub["layer"], sub["accuracy"],
                    color=colors.get(probe_type, "gray"),
                    linestyle=styles.get(probe_type, "-"),
                    linewidth=2, label=probe_type, alpha=0.85)
        n_classes = 3 if "3-class" in task_name else 2
        ax.axhline(y=1 / n_classes, color="gray", linestyle=":", alpha=0.5, label="Chance")
        ax.set_xlabel("Layer")
        ax.set_ylabel("CV Accuracy")
        ax.set_title(task_name)
        ax.legend(fontsize=9)
        ax.set_ylim(0.0, 1.05)

    plt.suptitle("Layer-wise Probe Accuracy (re-trained with judge labels)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "layer_sweep_judged.png", dpi=300, bbox_inches="tight")
    print(f"\nSaved layer_sweep_judged.png")

    df.to_csv(FIGURES_DIR / "table6_probe_judged.csv", index=False)
    print(f"Saved table6_probe_judged.csv")


if __name__ == "__main__":
    main()
