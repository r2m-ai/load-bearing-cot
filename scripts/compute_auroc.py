"""
Compute AUROC for probe results at best layers.
Uses same features/labels as 10b_retrain_probes.py but adds roc_auc_score.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
FIGURES_DIR = Path(__file__).parent.parent / "figures"


def compute_auroc_cv(X, y, groups=None, probe_type="linear", n_folds=5, use_pca=128):
    """Train probe with CV and compute AUROC from held-out predictions."""
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    if use_pca and X_s.shape[1] > use_pca:
        pca = PCA(n_components=min(use_pca, X_s.shape[0] - 1))
        X_s = pca.fit_transform(X_s)

    min_class = min(Counter(y).values())
    folds = min(n_folds, min_class)
    if folds < 2:
        folds = 2

    if probe_type == "linear":
        make_model = lambda: LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight="balanced")
    elif probe_type == "mlp":
        make_model = lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=1000,
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
        split_iter = cv.split(X_s, y, groups=groups)
    else:
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        split_iter = skf.split(X_s, y)
    n_classes = len(np.unique(y))

    fold_accs = []
    fold_f1s = []
    fold_aurocs = []

    for train_idx, test_idx in split_iter:
        X_train, X_test = X_s[train_idx], X_s[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = make_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc = np.mean(y_pred == y_test)
        fold_accs.append(acc)

        # F1 macro
        from sklearn.metrics import f1_score
        fold_f1s.append(f1_score(y_test, y_pred, average="macro"))

        # AUROC
        if n_classes == 2:
            # Binary: use probability of positive class
            auroc = roc_auc_score(y_test, y_proba[:, 1])
        else:
            # Multiclass: one-vs-rest
            auroc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
        fold_aurocs.append(auroc)

    return {
        "accuracy": np.mean(fold_accs),
        "accuracy_std": np.std(fold_accs),
        "f1_macro": np.mean(fold_f1s),
        "f1_std": np.std(fold_f1s),
        "auroc": np.mean(fold_aurocs),
        "auroc_std": np.std(fold_aurocs),
        "per_fold_acc": fold_accs,
        "per_fold_auroc": fold_aurocs,
        "n": len(X),
    }


def main():
    # Load features
    features_path = DATA_DIR / "extracted_features_42layers.npz"
    print(f"Loading features from {features_path}...")
    data = np.load(features_path, allow_pickle=True)

    pair_ids = data["pair_ids"]
    perturbation_points = data["perturbation_points"]
    n_layers = sum(1 for k in data.files if k.startswith("layer_"))
    if "question_ids" in data.files:
        all_question_ids = data["question_ids"]
    else:
        all_question_ids = pair_ids
    print(f"Loaded: {len(pair_ids)} examples, {n_layers} layers")

    # Load labels
    with open(DATA_DIR / "subclassified_pairs.json") as f:
        all_pairs = json.load(f)

    pair_lookup = {}
    for p in all_pairs:
        key = (p["id"], p.get("perturbation_point", ""))
        pair_lookup[key] = p

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

    # Best layers from table5_7_probe_with_ci.csv
    best_layers = {
        "3-class (A/B/C)": {"linear": 23, "mlp": 22},
        "Binary bypass": {"linear": 10, "mlp": 10},
        "Binary faithful": {"linear": 22, "mlp": 24},
    }

    tasks = [
        ("3-class (A/B/C)", y_3class, le_3class),
        ("Binary bypass", y_bypass, le_bypass),
        ("Binary faithful", y_faithful, le_faithful),
    ]

    results = []
    for task_name, y, le in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        dist = {le.classes_[i]: int(c) for i, c in enumerate(np.bincount(y))}
        print(f"  Class distribution: {dist}")

        for probe_type in ["linear", "mlp"]:
            layer = best_layers[task_name][probe_type]
            X_all = data[f"layer_{layer}"]
            X = X_all[valid_mask]

            print(f"\n  {probe_type} @ layer {layer}:")
            r = compute_auroc_cv(X, y, groups=question_ids, probe_type=probe_type)

            print(f"    Accuracy: {r['accuracy']:.3f} ± {r['accuracy_std']:.3f}")
            print(f"    F1-macro: {r['f1_macro']:.3f} ± {r['f1_std']:.3f}")
            print(f"    AUROC:    {r['auroc']:.3f} ± {r['auroc_std']:.3f}")
            print(f"    Per-fold acc:   {[f'{a:.3f}' for a in r['per_fold_acc']]}")
            print(f"    Per-fold auroc: {[f'{a:.3f}' for a in r['per_fold_auroc']]}")

            results.append({
                "task": task_name, "layer": layer, "probe": probe_type,
                "acc_mean": round(r["accuracy"], 4),
                "acc_std": round(r["accuracy_std"], 4),
                "f1_mean": round(r["f1_macro"], 4),
                "f1_std": round(r["f1_std"], 4),
                "auroc_mean": round(r["auroc"], 4),
                "auroc_std": round(r["auroc_std"], 4),
                "per_fold_acc": ",".join(f"{a:.3f}" for a in r["per_fold_acc"]),
                "per_fold_auroc": ",".join(f"{a:.3f}" for a in r["per_fold_auroc"]),
                "n": r["n"],
            })

    df = pd.DataFrame(results)
    out_path = FIGURES_DIR / "table_auroc.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
