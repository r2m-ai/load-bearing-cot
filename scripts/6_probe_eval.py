"""
Script 6: Probe evaluation and ablation studies.

Runs additional analyses:
1. Early detection: how much CoT do we need to detect unfaithfulness?
2. Per-source breakdown: GSM8K vs MMLU performance
3. Feature importance: which timepoints matter most?
4. Cross-task generalization: train on GSM8K, test on MMLU (and vice versa)

Output:
- figures/figure3_early_detection.png
- figures/table2_generalization.csv
"""

import json
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HIDDEN_DIR = PROCESSED_DIR / "hidden_states"
FIGURES_DIR = Path(__file__).parent.parent / "figures"

TIMEPOINTS = [0.0, 0.25, 0.50, 0.75, 1.0]


def extract_features_at_timepoints(
    hidden_states: torch.Tensor,
    prompt_length: int,
    layer: int,
    timepoints: list[float],
    pooling: str = "mean",
) -> np.ndarray:
    """Extract features at specified timepoints only."""
    h = hidden_states[layer]
    cot_h = h[prompt_length:]
    cot_len = cot_h.shape[0]

    if cot_len == 0:
        return np.zeros(len(timepoints) * h.shape[-1])

    features = []
    window_size = max(1, cot_len // 10)

    for tp in timepoints:
        center = int(tp * (cot_len - 1))
        start = max(0, center - window_size // 2)
        end = min(cot_len, start + window_size)
        window = cot_h[start:end].float()

        if pooling == "mean":
            feat = window.mean(dim=0).numpy()
        else:
            feat = window.max(dim=0).values.numpy()

        features.append(feat)

    return np.concatenate(features)


def build_dataset_with_source(
    metadata: list[dict],
    pairs_by_id: dict,
    layer: int,
    timepoints: list[float],
    pooling: str = "mean",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build dataset with source labels for cross-task analysis."""
    X_list, y_list, sources, qids = [], [], [], []

    for meta in metadata:
        pair = pairs_by_id[meta["id"]]

        # Faithful
        data = torch.load(HIDDEN_DIR / meta["faithful_file"], weights_only=True)
        feat = extract_features_at_timepoints(
            data["hidden_states"], data["prompt_length"],
            layer, timepoints, pooling,
        )
        X_list.append(feat)
        y_list.append(0)
        sources.append(pair["source"])
        qids.append(meta["id"])

        # Unfaithful
        data = torch.load(HIDDEN_DIR / meta["unfaithful_file"], weights_only=True)
        feat = extract_features_at_timepoints(
            data["hidden_states"], data["prompt_length"],
            layer, timepoints, pooling,
        )
        X_list.append(feat)
        y_list.append(1)
        sources.append(pair["source"])
        qids.append(meta["id"])

    return np.array(X_list), np.array(y_list), sources, np.array(qids)


def eval_early_detection(metadata, pairs_by_id, layer, pooling):
    """
    Test detection accuracy using only early portions of CoT.
    How much of the CoT do we need to see?
    """
    # Different "observation windows": see only first X% of CoT
    windows = [
        ([0.0], "0% only"),
        ([0.0, 0.25], "0-25%"),
        ([0.0, 0.25, 0.50], "0-50%"),
        ([0.0, 0.25, 0.50, 0.75], "0-75%"),
        ([0.0, 0.25, 0.50, 0.75, 1.0], "0-100% (full)"),
    ]

    results = []
    for timepoints, label in windows:
        print(f"\n  Window: {label}")
        X, y, _, question_ids = build_dataset_with_source(
            metadata, pairs_by_id, layer, timepoints, pooling
        )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if X_scaled.shape[1] > 128:
            pca = PCA(n_components=128)
            X_scaled = pca.fit_transform(X_scaled)

        model = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=cv, groups=question_ids, scoring="accuracy")

        results.append({
            "window": label,
            "timepoints": timepoints,
            "accuracy_mean": scores.mean(),
            "accuracy_std": scores.std(),
        })
        print(f"  Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

    return results


def eval_cross_task(metadata, pairs_by_id, layer, pooling):
    """Train on one task, test on another."""
    X, y, sources, _ = build_dataset_with_source(
        metadata, pairs_by_id, layer, TIMEPOINTS, pooling
    )

    sources = np.array(sources)
    results = []

    for train_source, test_source in [("gsm8k", "mmlu"), ("mmlu", "gsm8k")]:
        train_mask = sources == train_source
        test_mask = sources == test_source

        if train_mask.sum() < 10 or test_mask.sum() < 10:
            print(f"  Skipping {train_source}→{test_source}: insufficient data")
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if X_train_s.shape[1] > 128:
            pca = PCA(n_components=128)
            X_train_s = pca.fit_transform(X_train_s)
            X_test_s = pca.transform(X_test_s)

        model = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "train": train_source,
            "test": test_source,
            "accuracy": acc,
            "f1": f1,
            "n_train": train_mask.sum(),
            "n_test": test_mask.sum(),
        })
        print(f"  {train_source}→{test_source}: acc={acc:.3f}, f1={f1:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Probe evaluation and ablations")
    parser.add_argument("--pooling", type=str, default="mean")
    args = parser.parse_args()

    # Load metadata and pairs
    with open(HIDDEN_DIR / "metadata.json") as f:
        metadata = json.load(f)
    with open(PROCESSED_DIR / "faithful_unfaithful_pairs.json") as f:
        pairs = json.load(f)
    pairs_by_id = {p["id"]: p for p in pairs}

    # Load best probe info
    probe_path = PROCESSED_DIR / "probe_model.pkl"
    with open(probe_path, "rb") as f:
        probe_info = pickle.load(f)
    best_layer = probe_info["layer"]
    print(f"Using best layer: {best_layer}")

    # ---- 1. Early Detection ----
    print(f"\n{'='*50}")
    print("EARLY DETECTION ANALYSIS")
    print(f"{'='*50}")
    early_results = eval_early_detection(
        metadata, pairs_by_id, best_layer, args.pooling
    )

    # Plot Figure 3: Early detection
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 5))

    windows = [r["window"] for r in early_results]
    accs = [r["accuracy_mean"] for r in early_results]
    stds = [r["accuracy_std"] for r in early_results]

    bars = ax.bar(range(len(windows)), accs, yerr=stds, capsize=5,
                  color=["#BBDEFB", "#90CAF9", "#64B5F6", "#42A5F5", "#2196F3"],
                  edgecolor="white", linewidth=1.5)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows, rotation=15, ha="right")
    ax.set_ylabel("CV Accuracy")
    ax.set_title("Early Detection: How Much CoT Do We Need?")
    ax.set_ylim(0.3, 1.05)
    ax.legend()

    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{acc:.2f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "figure3_early_detection.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure3_early_detection.pdf", bbox_inches="tight")
    print(f"\nSaved Figure 3 to {fig_path}")

    # ---- 2. Cross-Task Generalization ----
    print(f"\n{'='*50}")
    print("CROSS-TASK GENERALIZATION")
    print(f"{'='*50}")
    cross_results = eval_cross_task(
        metadata, pairs_by_id, best_layer, args.pooling
    )

    # ---- 3. Per-Source Breakdown ----
    print(f"\n{'='*50}")
    print("PER-SOURCE BREAKDOWN")
    print(f"{'='*50}")
    X, y, sources, question_ids = build_dataset_with_source(
        metadata, pairs_by_id, best_layer, TIMEPOINTS, args.pooling
    )
    sources = np.array(sources)

    for src in ["gsm8k", "mmlu"]:
        mask = sources == src
        if mask.sum() < 10:
            continue
        X_src, y_src = X[mask], y[mask]
        qids_src = question_ids[mask]

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_src)
        if X_s.shape[1] > 128:
            pca = PCA(n_components=128)
            X_s = pca.fit_transform(X_s)

        model = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_s, y_src, cv=cv, groups=qids_src, scoring="accuracy")
        print(f"  {src}: {scores.mean():.3f} ± {scores.std():.3f} (n={mask.sum()})")

    # ---- Save Table 2 ----
    table2_rows = []

    # Early detection rows
    for r in early_results:
        table2_rows.append({
            "Experiment": "Early Detection",
            "Setting": r["window"],
            "Accuracy": f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}",
        })

    # Cross-task rows
    for r in cross_results:
        table2_rows.append({
            "Experiment": "Cross-Task",
            "Setting": f"Train: {r['train']} → Test: {r['test']}",
            "Accuracy": f"{r['accuracy']:.3f} (F1: {r['f1']:.3f})",
        })

    df = pd.DataFrame(table2_rows)
    table_path = FIGURES_DIR / "table2_generalization.csv"
    df.to_csv(table_path, index=False)
    print(f"\nSaved Table 2 to {table_path}")

    print(f"\n{'='*50}")
    print("EVALUATION COMPLETE")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
