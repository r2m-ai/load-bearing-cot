"""
Script 10 (v6): Fused hidden state extraction + MLP/linear probe sweep.

OPTIMIZED: Instead of saving 240GB of .pt files to disk then loading them back,
this script:
1. Loads the model once
2. For each example, runs ONE forward pass
3. Extracts perturbation-point features for ALL 42 layers in-memory
4. Stores only the feature vectors (~10KB per example, not ~240MB)
5. Trains all probes from RAM

This reduces the pipeline from ~3.5 hrs to ~30 min total.

Input: data/processed/subclassified_pairs.json (with judge labels)
Output: figures/layer_sweep.png, mlp_gain.png, table6_probe_comparison.csv
"""

import json
import argparse
import time
import os
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "google/gemma-2-9b-it"


def extract_perturbation_point_features_all_layers(
    hidden_states_tuple, prompt_length, n_layers, perturb_frac
):
    """
    Extract before/at/after features at perturbation point for ALL layers.
    hidden_states_tuple: tuple of (n_layers+1) tensors from model output
    Returns dict: layer_idx -> numpy feature vector
    """
    results = {}
    for l in range(n_layers):
        # hidden_states_tuple[0] is embedding, [1..n] are transformer layers
        h = hidden_states_tuple[l + 1].squeeze(0)  # (seq_len, hidden_dim)
        cot_h = h[prompt_length:]
        cot_len = cot_h.shape[0]

        if cot_len == 0:
            results[l] = np.zeros(h.shape[-1] * 3)
            continue

        idx = int(perturb_frac * (cot_len - 1))
        w = max(1, cot_len // 20)

        before = cot_h[max(0, idx - w * 2):max(1, idx - w)].float().mean(dim=0).cpu().numpy()
        at = cot_h[max(0, idx - w // 2):min(cot_len, idx + w // 2 + 1)].float().mean(dim=0).cpu().numpy()
        after = cot_h[min(cot_len - 1, idx + w):min(cot_len, idx + w * 2 + 1)].float().mean(dim=0).cpu().numpy()

        results[l] = np.concatenate([before, at, after])

    return results


def run_forward_pass(model, tokenizer, prompt, cot_text):
    """Run a single forward pass and return all hidden states."""
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": cot_text},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)

    # Get prompt length
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    return outputs.hidden_states, prompt_length


def train_eval_probe(X, y, groups=None, probe_type="linear", n_folds=5, use_pca=128):
    """Train and evaluate a probe. Uses GroupKFold when groups are provided."""
    if len(X) < 20 or len(np.unique(y)) < 2:
        return {"accuracy": 0, "accuracy_std": 0, "f1_macro": 0, "n": len(X)}

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    if use_pca and X_s.shape[1] > use_pca:
        n_comp = min(use_pca, X_s.shape[0] - 1)
        X_s = PCA(n_components=n_comp).fit_transform(X_s)

    min_class = min(Counter(y).values())
    folds = min(n_folds, min_class)
    if folds < 2:
        folds = 2

    if probe_type == "linear":
        model = LogisticRegression(max_iter=2000, C=1.0, random_state=42,
                                   class_weight="balanced")
    elif probe_type == "mlp":
        model = MLPClassifier(hidden_layer_sizes=(256,), max_iter=1000,
                              random_state=42, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=10)
    elif probe_type == "mlp_deep":
        model = MLPClassifier(hidden_layer_sizes=(512, 256), max_iter=1000,
                              random_state=42, early_stopping=True,
                              validation_fraction=0.15, n_iter_no_change=10)
    else:
        raise ValueError(f"Unknown probe_type: {probe_type}")

    if groups is not None:
        n_groups = len(np.unique(groups))
        folds = min(folds, n_groups)
        if folds < 2:
            folds = 2
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
        acc_scores = cross_val_score(model, X_s, y, cv=cv, groups=groups, scoring="accuracy")
        f1_scores = cross_val_score(model, X_s, y, cv=cv, groups=groups, scoring="f1_macro")
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        acc_scores = cross_val_score(model, X_s, y, cv=skf, scoring="accuracy")
        f1_scores = cross_val_score(model, X_s, y, cv=skf, scoring="f1_macro")

    return {
        "accuracy": acc_scores.mean(),
        "accuracy_std": acc_scores.std(),
        "f1_macro": f1_scores.mean(),
        "n": len(X),
    }


def main():
    parser = argparse.ArgumentParser(description="Fused MLP probe + layer sweep")
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--pca-dims", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--probe-types", type=str, default="linear,mlp,mlp_deep")
    args = parser.parse_args()

    probe_types = args.probe_types.split(",")
    use_pca = args.pca_dims if args.pca_dims > 0 else None
    device = args.device if torch.cuda.is_available() else "cpu"

    # Enable TF32 for H100
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load subclassified pairs
    with open(PROCESSED_DIR / "subclassified_pairs.json") as f:
        all_pairs = json.load(f)

    clear_pairs = [p for p in all_pairs if p.get("label_3class") in
                   ("silent_bypass", "self_correction", "error_propagation")]
    print(f"Clear pairs: {len(clear_pairs)}")

    if args.max_examples:
        clear_pairs = clear_pairs[:args.max_examples]
        print(f"Using first {len(clear_pairs)} examples")

    # Load model
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map=device, attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map=device, attn_implementation="sdpa",
        )
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"Model loaded: {n_layers} layers")

    if torch.cuda.is_available():
        print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # ---- PHASE 1: Fused extraction (GPU) ----
    print(f"\nPhase 1: Extracting features for {n_layers} layers (fused, no disk I/O)...")
    t0 = time.time()

    all_features = {l: [] for l in range(n_layers)}
    all_labels_3class = []
    all_labels_bypass = []
    all_labels_faithful = []
    all_question_ids = []

    for pair in tqdm(clear_pairs, desc="Forward passes"):
        # Get the perturbed CoT (prefix + continuation)
        perturbed_cot = pair.get("perturbed_cot", "")
        if not perturbed_cot:
            continue

        target_idx = pair.get("target_step_idx", 3)
        total_steps = pair.get("total_steps", 6)
        perturb_frac = target_idx / total_steps

        try:
            hidden_states, prompt_length = run_forward_pass(
                model, tokenizer, pair["prompt"], perturbed_cot
            )

            feats = extract_perturbation_point_features_all_layers(
                hidden_states, prompt_length, n_layers, perturb_frac
            )

            for l in range(n_layers):
                all_features[l].append(feats[l])

            all_labels_3class.append(pair["label_3class"])
            subtype = pair.get("subtype", "")
            all_labels_bypass.append("bypass" if subtype == "TYPE_A" else "non_bypass")
            all_labels_faithful.append("error_prop" if subtype == "TYPE_C" else "non_error_prop")
            all_question_ids.append(pair["id"])

            # Free GPU memory
            del hidden_states

        except Exception as e:
            print(f"  Error on {pair['id']}: {e}")
            continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    extraction_time = time.time() - t0
    print(f"Extraction done: {len(all_labels_3class)} examples in {extraction_time:.0f}s "
          f"({extraction_time / len(all_labels_3class):.1f}s/example)")

    # Free GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert to numpy
    for l in range(n_layers):
        all_features[l] = np.array(all_features[l])

    # Save extracted features so probes can be re-run locally without GPU
    # (e.g., after LLM judge corrects labels)
    features_path = PROCESSED_DIR / "extracted_features_42layers.npz"
    save_dict = {f"layer_{l}": all_features[l] for l in range(n_layers)}
    save_dict["pair_ids"] = np.array([p["id"] for p in clear_pairs[:len(all_labels_3class)]])
    save_dict["perturbation_points"] = np.array([
        p.get("perturbation_point", "") for p in clear_pairs[:len(all_labels_3class)]
    ])
    save_dict["question_ids"] = np.array(all_question_ids)
    np.savez_compressed(features_path, **save_dict)
    feat_size = features_path.stat().st_size / (1024 * 1024)
    print(f"Saved extracted features to {features_path} ({feat_size:.0f} MB)")
    print(f"  → Re-run probes locally: python3 experiments/probes/retrain_probes.py")

    question_ids = np.array(all_question_ids)

    le_3class = LabelEncoder()
    y_3class = le_3class.fit_transform(all_labels_3class)
    le_bypass = LabelEncoder()
    y_bypass = le_bypass.fit_transform(all_labels_bypass)
    le_faithful = LabelEncoder()
    y_faithful = le_faithful.fit_transform(all_labels_faithful)

    tasks = [
        ("3-class (A/B/C)", y_3class, le_3class),
        ("Binary bypass", y_bypass, le_bypass),
        ("Binary faithful", y_faithful, le_faithful),
    ]

    print(f"\nClass distributions:")
    for name, y, le in tasks:
        dist = {le.classes_[i]: int(c) for i, c in enumerate(np.bincount(y))}
        print(f"  {name}: {dist}")

    # ---- PHASE 2: Train probes (CPU) ----
    print(f"\nPhase 2: Training probes ({n_layers} layers × {len(probe_types)} types × {len(tasks)} tasks)...")
    t0 = time.time()
    results = []

    print(f"\nUsing Grouped-CV (StratifiedGroupKFold) with {len(np.unique(question_ids))} unique questions")

    for layer_idx in tqdm(range(n_layers), desc="Probes"):
        X = all_features[layer_idx]
        for task_name, y, le in tasks:
            for probe_type in probe_types:
                r = train_eval_probe(X, y, groups=question_ids, probe_type=probe_type, use_pca=use_pca)
                results.append({
                    "layer": layer_idx,
                    "task": task_name,
                    "probe": probe_type,
                    "accuracy": r["accuracy"],
                    "accuracy_std": r["accuracy_std"],
                    "f1_macro": r["f1_macro"],
                    "n": r["n"],
                })

    probe_time = time.time() - t0
    print(f"Probes done in {probe_time:.0f}s")

    df = pd.DataFrame(results)

    # ---- Print summary ----
    print(f"\n{'='*70}")
    print("BEST RESULTS PER TASK × PROBE TYPE")
    print(f"{'='*70}")
    for task_name in df["task"].unique():
        print(f"\n{task_name}:")
        for probe_type in probe_types:
            sub = df[(df["task"] == task_name) & (df["probe"] == probe_type)]
            if sub.empty:
                continue
            best_idx = sub["accuracy"].idxmax()
            best = sub.loc[best_idx]
            print(f"  {probe_type:10s}: layer {int(best['layer']):2d}, "
                  f"acc={best['accuracy']:.3f}±{best['accuracy_std']:.3f}, "
                  f"F1={best['f1_macro']:.3f}")

    # ---- Figure 7: Layer-wise accuracy ----
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors = {"linear": "#2196F3", "mlp": "#F44336", "mlp_deep": "#4CAF50"}
    styles = {"linear": "-", "mlp": "--", "mlp_deep": ":"}

    for ax, task_name in zip(axes, df["task"].unique()):
        task_df = df[df["task"] == task_name]
        for probe_type in probe_types:
            sub = task_df[task_df["probe"] == probe_type].sort_values("layer")
            ax.plot(sub["layer"], sub["accuracy"],
                    color=colors.get(probe_type, "gray"),
                    linestyle=styles.get(probe_type, "-"),
                    linewidth=2, label=probe_type, alpha=0.85)
            ax.fill_between(sub["layer"],
                            sub["accuracy"] - sub["accuracy_std"],
                            sub["accuracy"] + sub["accuracy_std"],
                            alpha=0.1, color=colors.get(probe_type, "gray"))

        n_classes = {"3-class (A/B/C)": 3}.get(task_name, 2)
        ax.axhline(y=1 / n_classes, color="gray", linestyle=":", alpha=0.5, label="Chance")
        ax.set_xlabel("Layer")
        ax.set_ylabel("CV Accuracy")
        ax.set_title(task_name)
        ax.legend(fontsize=9)
        ax.set_ylim(0.0, 1.05)

    plt.suptitle("Layer-wise Probe Accuracy: Linear vs MLP", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "layer_sweep.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "layer_sweep.pdf", bbox_inches="tight")
    print(f"\nSaved Figure 7")

    # ---- Figure 8: MLP gain ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    for ax, task_name in zip(axes, df["task"].unique()):
        task_df = df[df["task"] == task_name]
        linear_acc = task_df[task_df["probe"] == "linear"].sort_values("layer")["accuracy"].values

        for probe_type in ["mlp", "mlp_deep"]:
            probe_acc = task_df[task_df["probe"] == probe_type].sort_values("layer")["accuracy"].values
            if len(probe_acc) == 0:
                continue
            gain = probe_acc - linear_acc
            layers = task_df[task_df["probe"] == probe_type].sort_values("layer")["layer"].values
            ax.plot(layers, gain, color=colors.get(probe_type, "gray"),
                    linewidth=2, label=f"{probe_type} - linear")
            ax.fill_between(layers, 0, gain, alpha=0.15,
                            color=colors.get(probe_type, "gray"))

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Accuracy Gain")
        ax.set_title(task_name)
        ax.legend(fontsize=9)

    plt.suptitle("MLP Gain over Linear Probe by Layer", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mlp_gain.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "mlp_gain.pdf", bbox_inches="tight")
    print(f"Saved Figure 8")

    # ---- Table 6 ----
    df.to_csv(FIGURES_DIR / "table6_probe_comparison.csv", index=False)
    print(f"Saved Table 6")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("SUMMARY: Does MLP unlock bypass detection?")
    print(f"{'='*70}")

    for task_name in df["task"].unique():
        task_df = df[df["task"] == task_name]
        best_linear = task_df[task_df["probe"] == "linear"]["accuracy"].max()
        best_mlp = max(
            task_df[task_df["probe"] == p]["accuracy"].max()
            for p in probe_types if p != "linear"
        )
        gain = best_mlp - best_linear

        print(f"\n{task_name}:")
        print(f"  Linear best: {best_linear:.3f}")
        print(f"  MLP best:    {best_mlp:.3f}")
        print(f"  Gain:        {gain:+.3f}")

        if gain > 0.10:
            print(f"  → SIGNIFICANT — non-linear signal exists!")
        elif gain > 0.03:
            print(f"  → Modest gain — weak non-linear signal")
        else:
            print(f"  → No gain — signal not present in these features")

    print(f"\nTotal time: extraction={extraction_time:.0f}s + probes={probe_time:.0f}s "
          f"= {extraction_time + probe_time:.0f}s ({(extraction_time + probe_time) / 60:.0f} min)")


if __name__ == "__main__":
    main()
