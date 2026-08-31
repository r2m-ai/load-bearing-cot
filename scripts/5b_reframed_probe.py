"""
Script 5b (v3): Reframed probe — classify behavior WITHIN perturbed examples.

Instead of original-vs-perturbed (trivial surface difference), this probe asks:
Given hidden states from the PERTURBED prefix, can we predict whether the model
will (A) silently bypass, (B) self-correct, or (C) propagate the error?

Same input context, same perturbation — different downstream behavior.
If this works, it means the representation at the perturbation point already
encodes what the model will do next.

Supports both 3-class (A/B/C) and binary (bypass vs non-bypass) modes.

Input: data/processed/subclassified_pairs.json + hidden states
Output: figures/figure4_reframed_probe.png, figures/table3_reframed.csv
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HIDDEN_DIR = PROCESSED_DIR / "hidden_states"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TIMEPOINTS = [0.0, 0.25, 0.50, 0.75, 1.0]


def extract_features(
    hidden_states: torch.Tensor,
    prompt_length: int,
    layer: int,
    pooling: str = "mean",
) -> np.ndarray:
    """Extract features at key timepoints."""
    h = hidden_states[layer]
    cot_h = h[prompt_length:]
    cot_len = cot_h.shape[0]

    if cot_len == 0:
        return np.zeros(len(TIMEPOINTS) * h.shape[-1])

    features = []
    window_size = max(1, cot_len // 10)

    for tp in TIMEPOINTS:
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


def extract_perturbation_point_features(
    hidden_states: torch.Tensor,
    prompt_length: int,
    layer: int,
    perturbation_fraction: float = 0.5,
) -> np.ndarray:
    """
    Extract features specifically at the perturbation point.
    This is where the model "decides" to self-correct or follow the error.
    """
    h = hidden_states[layer]
    cot_h = h[prompt_length:]
    cot_len = cot_h.shape[0]

    if cot_len == 0:
        return np.zeros(h.shape[-1] * 3)

    # Get features at: just before perturbation, at perturbation, just after
    perturb_idx = int(perturbation_fraction * (cot_len - 1))
    window = max(1, cot_len // 20)

    before_start = max(0, perturb_idx - window * 2)
    before_end = max(1, perturb_idx - window)
    at_start = max(0, perturb_idx - window // 2)
    at_end = min(cot_len, perturb_idx + window // 2 + 1)
    after_start = min(cot_len - 1, perturb_idx + window)
    after_end = min(cot_len, perturb_idx + window * 2 + 1)

    before = cot_h[before_start:before_end].float().mean(dim=0).numpy()
    at = cot_h[at_start:at_end].float().mean(dim=0).numpy()
    after = cot_h[after_start:after_end].float().mean(dim=0).numpy()

    return np.concatenate([before, at, after])


def build_reframed_dataset(
    pairs: list[dict],
    metadata_by_id: dict,
    layer: int,
    feature_mode: str = "perturbation_point",
    label_mode: str = "3class",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build dataset using ONLY perturbed examples' hidden states.
    Labels come from behavior sub-classification.
    """
    X_list = []
    y_list = []
    ids = []
    qids = []

    for pair in tqdm(pairs, desc=f"Features (layer {layer})", leave=False):
        pid = pair["id"]
        if pid not in metadata_by_id:
            continue

        meta = metadata_by_id[pid]

        # Load the PERTURBED version's hidden states
        uf_path = HIDDEN_DIR / meta["unfaithful_file"]
        if not uf_path.exists():
            continue

        data = torch.load(uf_path, weights_only=True)

        # Get perturbation fraction
        target_idx = pair.get("target_step_idx", 3)
        total_steps = pair.get("total_steps", 6)
        perturb_frac = target_idx / total_steps

        if feature_mode == "perturbation_point":
            feat = extract_perturbation_point_features(
                data["hidden_states"], data["prompt_length"],
                layer, perturbation_fraction=perturb_frac,
            )
        elif feature_mode == "full":
            feat = extract_features(
                data["hidden_states"], data["prompt_length"], layer,
            )
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        # Assign label
        if label_mode == "3class":
            label = pair.get("label_3class", "unclear")
            if label == "unclear":
                continue
        elif label_mode == "binary_bypass":
            subtype = pair.get("subtype", "")
            if subtype == "TYPE_A":
                label = "silent_bypass"
            elif subtype in ("TYPE_B", "TYPE_C"):
                label = "non_bypass"
            else:
                continue
        elif label_mode == "binary_faithful":
            label = pair.get("label", "unclear")
            if label == "unclear":
                continue
        else:
            raise ValueError(f"Unknown label_mode: {label_mode}")

        X_list.append(feat)
        y_list.append(label)
        ids.append(f"{pid}_{pair.get('perturbation_point', '')}")
        qids.append(pid)

    X = np.array(X_list) if X_list else np.zeros((0, 0))
    le = LabelEncoder()
    y = le.fit_transform(y_list) if y_list else np.array([])
    question_ids = np.array(qids) if qids else np.array([])

    return X, y, ids, le, question_ids


def train_and_evaluate(X, y, groups=None, n_folds=5, use_pca=128):
    """Train logistic regression with k-fold CV. Uses Grouped-CV when groups provided."""
    if len(X) < 20 or len(np.unique(y)) < 2:
        return {"accuracy_mean": 0, "accuracy_std": 0, "f1_macro": 0,
                "fold_scores": [], "n_samples": len(X), "n_classes": len(np.unique(y))}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if use_pca and X_scaled.shape[1] > use_pca:
        pca = PCA(n_components=min(use_pca, X_scaled.shape[0] - 1))
        X_scaled = pca.fit_transform(X_scaled)

    # Adjust folds if class sizes are small
    min_class_size = min(np.bincount(y))
    actual_folds = min(n_folds, min_class_size)
    if actual_folds < 2:
        actual_folds = 2

    model = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                               class_weight="balanced")

    if groups is not None:
        n_groups = len(np.unique(groups))
        actual_folds = min(actual_folds, n_groups)
        if actual_folds < 2:
            actual_folds = 2
        cv = StratifiedGroupKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        acc_scores = cross_val_score(model, X_scaled, y, cv=cv, groups=groups, scoring="accuracy")
        f1_scores = cross_val_score(model, X_scaled, y, cv=cv, groups=groups, scoring="f1_macro")
    else:
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        acc_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="accuracy")
        f1_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="f1_macro")

    # Train on all for classification report
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    return {
        "accuracy_mean": acc_scores.mean(),
        "accuracy_std": acc_scores.std(),
        "f1_macro": f1_scores.mean(),
        "fold_scores": acc_scores.tolist(),
        "n_samples": len(X),
        "n_classes": len(np.unique(y)),
        "n_folds": actual_folds,
        "train_report": classification_report(y, y_pred, output_dict=True),
    }


def main():
    parser = argparse.ArgumentParser(description="Reframed probe (v3)")
    parser.add_argument("--feature-mode", type=str, default="perturbation_point",
                        choices=["perturbation_point", "full"])
    parser.add_argument("--pca-dims", type=int, default=128)
    args = parser.parse_args()

    use_pca = args.pca_dims if args.pca_dims > 0 else None

    # Load subclassified pairs
    sub_path = PROCESSED_DIR / "subclassified_pairs.json"
    if sub_path.exists():
        with open(sub_path) as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} subclassified pairs")
    else:
        # Fallback to regular pairs
        with open(PROCESSED_DIR / "faithful_unfaithful_pairs.json") as f:
            pairs = json.load(f)
        print(f"Loaded {len(pairs)} pairs (no sub-classification)")

    # Load metadata
    with open(HIDDEN_DIR / "metadata.json") as f:
        metadata = json.load(f)
    metadata_by_id = {m["id"]: m for m in metadata}

    # Get layer info
    sample_data = torch.load(
        HIDDEN_DIR / metadata[0]["unfaithful_file"], weights_only=True
    )
    n_layers = sample_data["hidden_states"].shape[0]
    print(f"Layers available: {n_layers}")

    # ---- Experiment 1: 3-class probe (Type A vs B vs C) ----
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: 3-class probe (silent_bypass / self_correction / error_propagation)")
    print(f"{'='*60}")

    results_3class = {}
    for layer_idx in range(n_layers):
        X, y, ids, le, question_ids = build_reframed_dataset(
            pairs, metadata_by_id, layer_idx,
            feature_mode=args.feature_mode, label_mode="3class",
        )
        if len(X) < 20:
            print(f"  Layer {layer_idx}: insufficient data ({len(X)} samples)")
            continue

        result = train_and_evaluate(X, y, groups=question_ids, use_pca=use_pca)
        results_3class[layer_idx] = result
        classes = le.classes_
        class_dist = {classes[i]: int(c) for i, c in enumerate(np.bincount(y))}
        print(f"  Layer {layer_idx}: acc={result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f}, "
              f"F1={result['f1_macro']:.3f}, n={result['n_samples']}, dist={class_dist}")

    # ---- Experiment 2: Binary probe (silent_bypass vs non-bypass) ----
    print(f"\n{'='*60}")
    print("EXPERIMENT 2: Binary probe (silent_bypass vs non-bypass)")
    print(f"{'='*60}")

    results_binary = {}
    for layer_idx in range(n_layers):
        X, y, ids, le, question_ids = build_reframed_dataset(
            pairs, metadata_by_id, layer_idx,
            feature_mode=args.feature_mode, label_mode="binary_bypass",
        )
        if len(X) < 20:
            continue

        result = train_and_evaluate(X, y, groups=question_ids, use_pca=use_pca)
        results_binary[layer_idx] = result
        print(f"  Layer {layer_idx}: acc={result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f}, "
              f"F1={result['f1_macro']:.3f}, n={result['n_samples']}")

    # ---- Experiment 3: Original binary (faithful vs unfaithful) with perturbation point features ----
    print(f"\n{'='*60}")
    print("EXPERIMENT 3: Binary faithful/unfaithful with perturbation-point features")
    print(f"{'='*60}")

    results_orig = {}
    for layer_idx in range(n_layers):
        X, y, ids, le, question_ids = build_reframed_dataset(
            pairs, metadata_by_id, layer_idx,
            feature_mode=args.feature_mode, label_mode="binary_faithful",
        )
        if len(X) < 20:
            continue

        result = train_and_evaluate(X, y, groups=question_ids, use_pca=use_pca)
        results_orig[layer_idx] = result
        print(f"  Layer {layer_idx}: acc={result['accuracy_mean']:.3f}±{result['accuracy_std']:.3f}, "
              f"F1={result['f1_macro']:.3f}, n={result['n_samples']}")

    # ---- Plot Figure 4: All probe results ----
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, results, title in [
        (axes[0], results_3class, "3-class\n(bypass/self-correct/propagate)"),
        (axes[1], results_binary, "Binary\n(bypass vs non-bypass)"),
        (axes[2], results_orig, "Binary\n(faithful vs unfaithful)"),
    ]:
        if not results:
            ax.set_title(title + "\n(no data)")
            continue

        layers = sorted(results.keys())
        accs = [results[l]["accuracy_mean"] for l in layers]
        stds = [results[l]["accuracy_std"] for l in layers]
        f1s = [results[l]["f1_macro"] for l in layers]

        ax.plot(layers, accs, "o-", color="#2196F3", linewidth=2, label="Accuracy")
        ax.fill_between(layers, [a-s for a,s in zip(accs,stds)],
                        [a+s for a,s in zip(accs,stds)], alpha=0.2, color="#2196F3")
        ax.plot(layers, f1s, "s--", color="#4CAF50", linewidth=1.5, label="F1 (macro)")

        n_classes = results[layers[0]].get("n_classes", 2)
        chance = 1.0 / n_classes
        ax.axhline(y=chance, color="gray", linestyle=":", alpha=0.5, label=f"Chance ({chance:.2f})")

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_ylim(0.0, 1.05)

    plt.suptitle(f"Reframed Probes: Classifying Model Behavior from Perturbed Representations\n"
                 f"(feature mode: {args.feature_mode})", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure4_reframed_probe.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure4_reframed_probe.pdf", bbox_inches="tight")
    print(f"\nSaved Figure 4")

    # ---- Table 3 ----
    rows = []
    for label_mode, results, name in [
        ("3class", results_3class, "3-class"),
        ("binary_bypass", results_binary, "Binary (bypass)"),
        ("binary_faithful", results_orig, "Binary (faithful)"),
    ]:
        for l in sorted(results.keys()):
            r = results[l]
            rows.append({
                "Experiment": name,
                "Layer": l,
                "Accuracy": f"{r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f}",
                "F1_macro": f"{r['f1_macro']:.3f}",
                "N_samples": r["n_samples"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(FIGURES_DIR / "table3_reframed.csv", index=False)
    print(f"Saved Table 3")

    # Summary
    print(f"\n{'='*60}")
    print("REFRAMED PROBE SUMMARY")
    print(f"{'='*60}")
    for name, results in [("3-class", results_3class), ("Binary bypass", results_binary),
                          ("Binary faithful", results_orig)]:
        if results:
            best_layer = max(results, key=lambda l: results[l]["accuracy_mean"])
            best = results[best_layer]
            print(f"  {name:20s}: best layer={best_layer}, "
                  f"acc={best['accuracy_mean']:.3f}±{best['accuracy_std']:.3f}, "
                  f"F1={best['f1_macro']:.3f}")


if __name__ == "__main__":
    main()
