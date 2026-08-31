"""
Script 8 (v4): Feature leakage detection.

Tests whether the 99.9% probe accuracy reflects genuine mechanistic signal
or leakage from perturbation type / strategy.

If the probe learns "what the model will do" (real signal), it should
transfer across strategies, positions, and sources.
If it learns "which perturbation was applied" (leakage), it won't transfer.

Tests:
1. Cross-strategy: train on strategy X, test on strategy Y
2. Cross-source: train on GSM8K, test on MMLU
3. Cross-position: train on early, test on late
4. Within-strategy: train/test within same strategy (upper bound)

Output: figures/figure6_leakage_test.png, figures/table5_leakage.csv
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from itertools import combinations

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HIDDEN_DIR = PROCESSED_DIR / "hidden_states"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def extract_perturbation_point_all_layers(hidden_states, prompt_length, n_layers, perturb_frac):
    """Extract perturbation-point features for all layers."""
    results = {}
    for l in range(n_layers):
        h = hidden_states[l]
        cot_h = h[prompt_length:]
        cot_len = cot_h.shape[0]
        if cot_len == 0:
            results[l] = np.zeros(h.shape[-1] * 3)
            continue
        idx = int(perturb_frac * (cot_len - 1))
        w = max(1, cot_len // 20)
        before = cot_h[max(0, idx-w*2):max(1, idx-w)].float().mean(dim=0).numpy()
        at = cot_h[max(0, idx-w//2):min(cot_len, idx+w//2+1)].float().mean(dim=0).numpy()
        after = cot_h[min(cot_len-1, idx+w):min(cot_len, idx+w*2+1)].float().mean(dim=0).numpy()
        results[l] = np.concatenate([before, at, after])
    return results


def train_test_probe(X_train, y_train, X_test, y_test, use_pca=128):
    """Train on one split, test on another."""
    if len(X_train) < 10 or len(X_test) < 5:
        return None
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    if use_pca and X_tr.shape[1] > use_pca:
        pca = PCA(n_components=min(use_pca, X_tr.shape[0] - 1))
        X_tr = pca.fit_transform(X_tr)
        X_te = pca.transform(X_te)

    model = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight="balanced")
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_classes": dict(Counter(y_train)),
        "test_classes": dict(Counter(y_test)),
    }


def main():
    parser = argparse.ArgumentParser(description="Feature leakage test")
    parser.add_argument("--layer", type=int, default=2,
                        help="Layer index to test (default: 2, best from v3)")
    args = parser.parse_args()

    # Load subclassified pairs
    with open(PROCESSED_DIR / "subclassified_pairs.json") as f:
        all_pairs = json.load(f)

    # Filter to clear examples only
    pairs = [p for p in all_pairs if p.get("label_3class") in
             ("silent_bypass", "self_correction", "error_propagation")]
    print(f"Loaded {len(pairs)} clearly classified pairs")

    # Load metadata
    with open(HIDDEN_DIR / "metadata.json") as f:
        metadata = json.load(f)
    meta_by_id = {m["id"]: m for m in metadata}

    sample = torch.load(HIDDEN_DIR / metadata[0]["unfaithful_file"], weights_only=True)
    n_layers = sample["hidden_states"].shape[0]
    layer = args.layer
    print(f"Using layer index {layer} (of {n_layers})")

    # Single-pass feature extraction
    print("Extracting features (single pass)...")
    features = []
    labels = []
    meta_info = []  # (strategy, source, position)
    question_ids_list = []

    for pair in tqdm(pairs, desc="Loading"):
        pid = pair["id"]
        if pid not in meta_by_id:
            continue
        meta = meta_by_id[pid]
        uf_path = HIDDEN_DIR / meta["unfaithful_file"]
        if not uf_path.exists():
            continue

        data = torch.load(uf_path, weights_only=True)
        target_idx = pair.get("target_step_idx", 3)
        total_steps = pair.get("total_steps", 6)
        perturb_frac = target_idx / total_steps

        feats = extract_perturbation_point_all_layers(
            data["hidden_states"], data["prompt_length"], n_layers, perturb_frac
        )

        # Binary label: bypass vs non-bypass
        label = "bypass" if pair.get("subtype") == "TYPE_A" else "non_bypass"

        features.append(feats[layer])
        labels.append(label)
        question_ids_list.append(pid)
        meta_info.append({
            "strategy": pair.get("perturbation_strategy", "unknown"),
            "source": pair.get("source", "unknown"),
            "position": pair.get("perturbation_point", "unknown"),
            "label_3class": pair.get("label_3class", "unknown"),
        })
        del data

    X = np.array(features)
    y = np.array(labels)
    question_ids = np.array(question_ids_list)
    strategies = np.array([m["strategy"] for m in meta_info])
    sources = np.array([m["source"] for m in meta_info])
    positions = np.array([m["position"] for m in meta_info])

    print(f"Total: {len(X)} examples, {Counter(y)}")

    results = []

    # ---- Test 1: Within-strategy CV (upper bound) ----
    print(f"\n{'='*60}")
    print("TEST 1: Within-strategy cross-validation (upper bound)")
    print(f"{'='*60}")

    for strat in sorted(set(strategies)):
        mask = strategies == strat
        X_s, y_s = X[mask], y[mask]
        qids_s = question_ids[mask]
        if len(X_s) < 20 or len(set(y_s)) < 2:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_s)
        if X_scaled.shape[1] > 128:
            X_scaled = PCA(n_components=min(128, len(X_scaled)-1)).fit_transform(X_scaled)

        model = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight="balanced")
        min_class = min(Counter(y_s).values())
        folds = min(5, min_class)
        n_groups = len(np.unique(qids_s))
        folds = min(folds, n_groups)
        if folds < 2:
            folds = 2
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, y_s, cv=cv, groups=qids_s, scoring="accuracy")

        print(f"  {strat:25s}: {scores.mean():.3f}±{scores.std():.3f} (n={len(X_s)}, dist={dict(Counter(y_s))})")
        results.append({
            "Test": "Within-strategy CV",
            "Train": strat, "Test_on": strat,
            "Accuracy": f"{scores.mean():.3f}",
            "F1": "-",
            "N_train": len(X_s), "N_test": len(X_s),
        })

    # ---- Test 2: Cross-strategy transfer ----
    print(f"\n{'='*60}")
    print("TEST 2: Cross-strategy transfer (LEAKAGE TEST)")
    print(f"{'='*60}")

    valid_strategies = [s for s in sorted(set(strategies))
                        if sum(strategies == s) >= 20 and len(set(y[strategies == s])) >= 2]

    transfer_matrix = {}
    for train_strat in valid_strategies:
        transfer_matrix[train_strat] = {}
        for test_strat in valid_strategies:
            if train_strat == test_strat:
                continue
            train_mask = strategies == train_strat
            test_mask = strategies == test_strat

            r = train_test_probe(X[train_mask], y[train_mask], X[test_mask], y[test_mask])
            if r:
                transfer_matrix[train_strat][test_strat] = r["accuracy"]
                print(f"  {train_strat:25s} → {test_strat:25s}: acc={r['accuracy']:.3f}, f1={r['f1_macro']:.3f}")
                results.append({
                    "Test": "Cross-strategy",
                    "Train": train_strat, "Test_on": test_strat,
                    "Accuracy": f"{r['accuracy']:.3f}",
                    "F1": f"{r['f1_macro']:.3f}",
                    "N_train": r["n_train"], "N_test": r["n_test"],
                })

    # ---- Test 3: Cross-source transfer ----
    print(f"\n{'='*60}")
    print("TEST 3: Cross-source transfer (GSM8K ↔ MMLU)")
    print(f"{'='*60}")

    for train_src, test_src in [("gsm8k", "mmlu"), ("mmlu", "gsm8k")]:
        train_mask = sources == train_src
        test_mask = sources == test_src
        r = train_test_probe(X[train_mask], y[train_mask], X[test_mask], y[test_mask])
        if r:
            print(f"  {train_src} → {test_src}: acc={r['accuracy']:.3f}, f1={r['f1_macro']:.3f}, "
                  f"n_train={r['n_train']}, n_test={r['n_test']}")
            results.append({
                "Test": "Cross-source",
                "Train": train_src, "Test_on": test_src,
                "Accuracy": f"{r['accuracy']:.3f}",
                "F1": f"{r['f1_macro']:.3f}",
                "N_train": r["n_train"], "N_test": r["n_test"],
            })
        else:
            print(f"  {train_src} → {test_src}: SKIPPED (insufficient data)")

    # ---- Test 4: Cross-position transfer ----
    print(f"\n{'='*60}")
    print("TEST 4: Cross-position transfer")
    print(f"{'='*60}")

    for train_pos, test_pos in [("early", "late"), ("late", "early"),
                                 ("early", "middle"), ("middle", "late")]:
        train_mask = positions == train_pos
        test_mask = positions == test_pos
        r = train_test_probe(X[train_mask], y[train_mask], X[test_mask], y[test_mask])
        if r:
            print(f"  {train_pos} → {test_pos}: acc={r['accuracy']:.3f}, f1={r['f1_macro']:.3f}")
            results.append({
                "Test": "Cross-position",
                "Train": train_pos, "Test_on": test_pos,
                "Accuracy": f"{r['accuracy']:.3f}",
                "F1": f"{r['f1_macro']:.3f}",
                "N_train": r["n_train"], "N_test": r["n_test"],
            })

    # ---- Test 5: Leave-one-strategy-out ----
    print(f"\n{'='*60}")
    print("TEST 5: Leave-one-strategy-out")
    print(f"{'='*60}")

    for held_out in valid_strategies:
        train_mask = strategies != held_out
        test_mask = strategies == held_out

        # Need both classes in both splits
        if len(set(y[train_mask])) < 2 or len(set(y[test_mask])) < 2:
            continue

        r = train_test_probe(X[train_mask], y[train_mask], X[test_mask], y[test_mask])
        if r:
            print(f"  Hold out {held_out:25s}: acc={r['accuracy']:.3f}, f1={r['f1_macro']:.3f}, n_test={r['n_test']}")
            results.append({
                "Test": "Leave-one-strategy-out",
                "Train": f"all except {held_out}", "Test_on": held_out,
                "Accuracy": f"{r['accuracy']:.3f}",
                "F1": f"{r['f1_macro']:.3f}",
                "N_train": r["n_train"], "N_test": r["n_test"],
            })

    # ---- Figure 6: Transfer matrix heatmap ----
    sns.set_theme(style="whitegrid", font_scale=1.0)

    if transfer_matrix:
        strats = sorted(transfer_matrix.keys())
        matrix = np.zeros((len(strats), len(strats)))
        for i, train_s in enumerate(strats):
            for j, test_s in enumerate(strats):
                if train_s == test_s:
                    matrix[i][j] = np.nan  # diagonal = within-strategy
                elif test_s in transfer_matrix.get(train_s, {}):
                    matrix[i][j] = transfer_matrix[train_s][test_s]

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.isnan(matrix)
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                    xticklabels=[s.replace("_", "\n") for s in strats],
                    yticklabels=[s.replace("_", "\n") for s in strats],
                    mask=mask, vmin=0.3, vmax=1.0, ax=ax,
                    cbar_kws={"label": "Transfer Accuracy"})
        ax.set_xlabel("Test Strategy")
        ax.set_ylabel("Train Strategy")
        ax.set_title("Cross-Strategy Transfer Matrix\n(high transfer = real signal, low = leakage)")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "figure6_leakage_test.png", dpi=300, bbox_inches="tight")
        plt.savefig(FIGURES_DIR / "figure6_leakage_test.pdf", bbox_inches="tight")
        print(f"\nSaved Figure 6 (transfer matrix)")

    # Table 5
    df = pd.DataFrame(results)
    df.to_csv(FIGURES_DIR / "table5_leakage.csv", index=False)
    print(f"Saved Table 5")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("LEAKAGE TEST SUMMARY")
    print(f"{'='*60}")

    # Compute average transfer accuracy
    cross_strat = [r for r in results if r["Test"] == "Cross-strategy"]
    if cross_strat:
        avg_transfer = np.mean([float(r["Accuracy"]) for r in cross_strat])
        print(f"\nCross-strategy transfer: avg={avg_transfer:.3f} ({len(cross_strat)} pairs)")
        if avg_transfer > 0.80:
            print("→ HIGH TRANSFER: Probe captures genuine mechanistic signal, not strategy artifacts")
        elif avg_transfer > 0.60:
            print("→ PARTIAL TRANSFER: Some real signal, some strategy-specific features")
        else:
            print("→ LOW TRANSFER: Probe may be detecting perturbation type, not model behavior")

    loso = [r for r in results if r["Test"] == "Leave-one-strategy-out"]
    if loso:
        avg_loso = np.mean([float(r["Accuracy"]) for r in loso])
        print(f"\nLeave-one-strategy-out: avg={avg_loso:.3f}")

    cross_src = [r for r in results if r["Test"] == "Cross-source"]
    if cross_src:
        for r in cross_src:
            print(f"\nCross-source {r['Train']}→{r['Test_on']}: {r['Accuracy']}")

    cross_pos = [r for r in results if r["Test"] == "Cross-position"]
    if cross_pos:
        avg_pos = np.mean([float(r["Accuracy"]) for r in cross_pos])
        print(f"\nCross-position transfer: avg={avg_pos:.3f}")


if __name__ == "__main__":
    main()
