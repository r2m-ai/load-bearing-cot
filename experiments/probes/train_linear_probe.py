"""
Script 5: Train linear probe to classify faithful vs. unfaithful CoT.

Extracts features from hidden states at 5 key timepoints in the CoT,
then trains a logistic regression classifier.

Output:
- figures/figure2_probe_accuracy.png
- figures/table1_results.csv
- data/processed/probe_model.pkl
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HIDDEN_DIR = PROCESSED_DIR / "hidden_states"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Timepoints to sample (as fraction of CoT length)
TIMEPOINTS = [0.0, 0.25, 0.50, 0.75, 1.0]
TIMEPOINT_LABELS = ["0%", "25%", "50%", "75%", "100%"]


def extract_features(
    hidden_states: torch.Tensor,
    prompt_length: int,
    layer: int,
    pooling: str = "mean",
) -> np.ndarray:
    """
    Extract features from hidden states at key timepoints.

    Args:
        hidden_states: (n_layers, seq_len, hidden_dim)
        prompt_length: number of prompt tokens
        layer: which layer to extract from
        pooling: "mean" or "max" pooling over a window of tokens

    Returns:
        Feature vector of shape (n_timepoints * hidden_dim,)
    """
    # Get the specified layer's hidden states for the CoT portion
    h = hidden_states[layer]  # (seq_len, hidden_dim)
    cot_h = h[prompt_length:]  # (cot_len, hidden_dim)
    cot_len = cot_h.shape[0]

    if cot_len == 0:
        return np.zeros(len(TIMEPOINTS) * h.shape[-1])

    features = []
    window_size = max(1, cot_len // 10)  # Pool over ~10% of CoT

    for tp in TIMEPOINTS:
        center = int(tp * (cot_len - 1))
        start = max(0, center - window_size // 2)
        end = min(cot_len, start + window_size)

        window = cot_h[start:end].float()

        if pooling == "mean":
            feat = window.mean(dim=0).numpy()
        elif pooling == "max":
            feat = window.max(dim=0).values.numpy()
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        features.append(feat)

    return np.concatenate(features)


def build_dataset(
    metadata: list[dict],
    layer: int,
    pooling: str = "mean",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build feature matrix and labels from hidden states.

    Returns:
        X: (n_samples, n_features)
        y: (n_samples,) — 0=faithful, 1=unfaithful
        ids: list of example IDs
    """
    X_list = []
    y_list = []
    ids = []
    qids = []

    for meta in tqdm(metadata, desc=f"Features (layer {layer})"):
        # Faithful version → label 0
        faithful_data = torch.load(
            HIDDEN_DIR / meta["faithful_file"], weights_only=True
        )
        feat_f = extract_features(
            faithful_data["hidden_states"],
            faithful_data["prompt_length"],
            layer, pooling,
        )
        X_list.append(feat_f)
        y_list.append(0)
        ids.append(f"{meta['id']}_faithful")
        qids.append(meta["id"])

        # Unfaithful version → label 1
        unfaithful_data = torch.load(
            HIDDEN_DIR / meta["unfaithful_file"], weights_only=True
        )
        feat_u = extract_features(
            unfaithful_data["hidden_states"],
            unfaithful_data["prompt_length"],
            layer, pooling,
        )
        X_list.append(feat_u)
        y_list.append(1)
        ids.append(f"{meta['id']}_unfaithful")
        qids.append(meta["id"])

    X = np.array(X_list)
    y = np.array(y_list)
    question_ids = np.array(qids)
    return X, y, ids, question_ids


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_folds: int = 5,
    use_pca: int | None = 128,
) -> dict:
    """
    Train logistic regression with k-fold cross-validation.
    Uses Grouped-CV (StratifiedGroupKFold) when groups are provided.

    Returns dict with accuracy, precision, recall, F1, and per-fold scores.
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA for dimensionality reduction
    pca = None
    if use_pca and X_scaled.shape[1] > use_pca:
        pca = PCA(n_components=use_pca)
        X_scaled = pca.fit_transform(X_scaled)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"  PCA: {X.shape[1]} → {use_pca} dims ({explained_var:.1%} variance explained)")

    # K-fold cross-validation
    model = LogisticRegression(max_iter=2000, C=1.0, random_state=42)

    if groups is not None:
        n_groups = len(np.unique(groups))
        actual_folds = min(n_folds, n_groups)
        if actual_folds < 2:
            actual_folds = 2
        cv = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        fold_scores = cross_val_score(model, X_scaled, y, cv=cv, groups=groups, scoring="accuracy")
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="accuracy")

    # Train final model on all data for inspection
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    results = {
        "accuracy_mean": fold_scores.mean(),
        "accuracy_std": fold_scores.std(),
        "fold_scores": fold_scores.tolist(),
        "train_accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }

    return results, model, scaler, pca


def main():
    parser = argparse.ArgumentParser(description="Train linear probe")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"])
    parser.add_argument("--pca-dims", type=int, default=128, help="PCA dimensions (0 to disable)")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    use_pca = args.pca_dims if args.pca_dims > 0 else None

    # Load metadata
    with open(HIDDEN_DIR / "metadata.json") as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} pairs ({len(metadata)*2} examples)")

    # Determine number of layers from a sample file
    sample_data = torch.load(
        HIDDEN_DIR / metadata[0]["faithful_file"], weights_only=True
    )
    n_layers = sample_data["hidden_states"].shape[0]
    hidden_dim = sample_data["hidden_states"].shape[2]
    print(f"Model: {n_layers} layers, {hidden_dim} hidden dim")

    # Train probe at each layer
    layer_results = {}

    print(f"\nTraining probes across {n_layers} layers...")
    for layer_idx in range(n_layers):
        print(f"\nLayer {layer_idx}/{n_layers-1}:")
        X, y, ids, question_ids = build_dataset(metadata, layer_idx, args.pooling)
        print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")

        results, model, scaler, pca = train_and_evaluate(
            X, y, groups=question_ids, n_folds=args.n_folds, use_pca=use_pca
        )
        layer_results[layer_idx] = results

        print(f"  CV Accuracy: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f}")
        print(f"  F1: {results['f1']:.3f}")

    # Find best layer
    best_layer = max(layer_results, key=lambda l: layer_results[l]["accuracy_mean"])
    best_acc = layer_results[best_layer]["accuracy_mean"]
    print(f"\n{'='*50}")
    print(f"Best layer: {best_layer} (accuracy: {best_acc:.3f})")

    # Re-train best layer probe and save
    print(f"\nRe-training best probe (layer {best_layer})...")
    X_best, y_best, ids_best, qids_best = build_dataset(metadata, best_layer, args.pooling)
    best_results, best_model, best_scaler, best_pca = train_and_evaluate(
        X_best, y_best, groups=qids_best, n_folds=args.n_folds, use_pca=use_pca
    )

    # Save the best probe
    probe_path = PROCESSED_DIR / "probe_model.pkl"
    with open(probe_path, "wb") as f:
        pickle.dump({
            "model": best_model,
            "scaler": best_scaler,
            "pca": best_pca,
            "layer": best_layer,
            "pooling": args.pooling,
            "results": best_results,
        }, f)
    print(f"Saved best probe to {probe_path}")

    # ---- Figure 2: Probe accuracy by layer ----
    layers = sorted(layer_results.keys())
    accs = [layer_results[l]["accuracy_mean"] for l in layers]
    stds = [layer_results[l]["accuracy_std"] for l in layers]
    f1s = [layer_results[l]["f1"] for l in layers]

    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, accs, color="#2196F3", linewidth=2.5, marker="o",
            markersize=4, label="CV Accuracy")
    ax.fill_between(
        layers,
        [a - s for a, s in zip(accs, stds)],
        [a + s for a, s in zip(accs, stds)],
        alpha=0.2, color="#2196F3",
    )
    ax.plot(layers, f1s, color="#4CAF50", linewidth=2, linestyle="--",
            marker="s", markersize=3, label="F1 Score")

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")
    ax.axvline(x=best_layer, color="#F44336", linestyle="--", alpha=0.5,
               label=f"Best layer ({best_layer})")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Score")
    ax.set_title("Linear Probe: Faithful vs. Unfaithful CoT Detection by Layer")
    ax.legend(loc="lower right")
    ax.set_ylim(0.3, 1.05)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "figure2_probe_accuracy.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure2_probe_accuracy.pdf", bbox_inches="tight")
    print(f"\nSaved Figure 2 to {fig_path}")

    # ---- Table 1: Classification metrics ----
    table_data = []
    for l in layers:
        r = layer_results[l]
        table_data.append({
            "Layer": l,
            "CV Accuracy": f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}",
            "Precision": f"{r['precision']:.3f}",
            "Recall": f"{r['recall']:.3f}",
            "F1": f"{r['f1']:.3f}",
        })

    df = pd.DataFrame(table_data)
    table_path = FIGURES_DIR / "table1_results.csv"
    df.to_csv(table_path, index=False)
    print(f"Saved Table 1 to {table_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"PROBE TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Best layer: {best_layer}")
    print(f"Best CV accuracy: {best_acc:.3f} ± {layer_results[best_layer]['accuracy_std']:.3f}")
    print(f"Best F1: {layer_results[best_layer]['f1']:.3f}")
    print(f"\nTop 5 layers:")
    sorted_layers = sorted(layers, key=lambda l: layer_results[l]["accuracy_mean"], reverse=True)
    for l in sorted_layers[:5]:
        r = layer_results[l]
        print(f"  Layer {l:2d}: {r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f}")


if __name__ == "__main__":
    main()
