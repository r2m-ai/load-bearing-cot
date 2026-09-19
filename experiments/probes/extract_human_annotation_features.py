"""
exp_2_3_extract_annot_features.py — extract Gemma-2-9B-IT hidden-state
features for the n=200 human-annotation subset (item 2.3, GPU step).

Why this exists: the existing `data/processed/extracted_features_42layers.npz`
was extracted from a separately-sampled N=2,000 probe-training set; only
15 of the 200 annotation rows overlap that set. To run probe-vs-human on
the full n=200 we need features for the other 185 rows. This script
extracts features for ALL 200 (including the 15 overlap rows, which act
as a sanity check against the canonical features file).

Reuses the exact extraction logic from `experiments/probes/layer_sweep.py`:
    forward pass via chat-template → output_hidden_states → window-averaged
    before/at/after features at perturbation point (3 * hidden_dim per layer).

Usage:
  # Main extraction (run on H100/H200):
  python experiments/probes/extract_human_annotation_features.py

  # Sanity check after extraction (compares the 15 overlap rows against
  # the canonical features file at layer 10). Run on the same machine:
  python experiments/probes/extract_human_annotation_features.py --sanity-only

Output: data/processed/extracted_features_annot200.npz
  - layer_0 .. layer_41: (200, 10752) float32
  - pair_ids, perturbation_strategies, perturbation_points: (200,) str
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
META_CSV = ROOT / "data" / "annotations" / "pilot_n200" / "_analysis_metadata.csv"
EXP_PAIRS = DATA / "expanded_pairs.json"
CANONICAL_FEATS = DATA / "extracted_features_42layers.npz"
OUT_FEATS = DATA / "extracted_features_annot200.npz"

MODEL_ID = "google/gemma-2-9b-it"


# ---------------------------------------------------------------------------
# Reused verbatim from experiments/probes/layer_sweep.py for bit-exact parity
# ---------------------------------------------------------------------------

def extract_perturbation_point_features_all_layers(
    hidden_states_tuple, prompt_length, n_layers, perturb_frac
):
    results = {}
    for l in range(n_layers):
        h = hidden_states_tuple[l + 1].squeeze(0)
        cot_h = h[prompt_length:]
        cot_len = cot_h.shape[0]

        if cot_len == 0:
            results[l] = np.zeros(h.shape[-1] * 3, dtype=np.float32)
            continue

        idx = int(perturb_frac * (cot_len - 1))
        w = max(1, cot_len // 20)

        before = cot_h[max(0, idx - w * 2):max(1, idx - w)].float().mean(dim=0).cpu().numpy()
        at = cot_h[max(0, idx - w // 2):min(cot_len, idx + w // 2 + 1)].float().mean(dim=0).cpu().numpy()
        after = cot_h[min(cot_len - 1, idx + w):min(cot_len, idx + w * 2 + 1)].float().mean(dim=0).cpu().numpy()

        results[l] = np.concatenate([before, at, after]).astype(np.float32)

    return results


def run_forward_pass(model, tokenizer, prompt, cot_text):
    import torch
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

    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(
            **inputs, output_hidden_states=True, use_cache=False,
        )
    return outputs.hidden_states, prompt_length


# ---------------------------------------------------------------------------
# Annotation-set loading
# ---------------------------------------------------------------------------

def load_annotation_records() -> list[dict]:
    """Return the 200 expanded_pairs.json records (in annotation_id order)
    that correspond to the human-annotation subset."""
    with open(META_CSV) as f:
        meta = list(csv.DictReader(f))
    composite_keys = [
        (m["example_id"], m["strategy"], m["perturbation_point"]) for m in meta
    ]

    pairs = json.load(open(EXP_PAIRS))
    lookup = {
        (p["id"], p.get("perturbation_strategy", ""), p.get("perturbation_point", "")): p
        for p in pairs
    }
    records = []
    missing = []
    for ann_id, ck in enumerate(composite_keys, start=1):
        p = lookup.get(ck)
        if p is None:
            missing.append((ann_id, ck))
            continue
        records.append({**p, "_annotation_id": ann_id, "_composite": ck})

    if missing:
        raise RuntimeError(f"{len(missing)} annotation rows have no matching record in expanded_pairs.json: {missing[:5]}...")

    print(f"Loaded {len(records)} annotation records (all 200 matched expanded_pairs.json).")
    return records


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_all(records: list[dict], device: str = "cuda") -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

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

    all_features = {l: [] for l in range(n_layers)}
    pair_ids, strategies, points = [], [], []

    t0 = time.time()
    for r in tqdm(records, desc="Forward passes"):
        target_idx = r.get("target_step_idx", 3)
        total_steps = max(r.get("total_steps", 6), 1)
        perturb_frac = target_idx / total_steps

        hidden_states, prompt_length = run_forward_pass(
            model, tokenizer, r["prompt"], r["perturbed_cot"]
        )
        feats = extract_perturbation_point_features_all_layers(
            hidden_states, prompt_length, n_layers, perturb_frac
        )
        for l in range(n_layers):
            all_features[l].append(feats[l])
        pair_ids.append(r["id"])
        strategies.append(r.get("perturbation_strategy", ""))
        points.append(r.get("perturbation_point", ""))

    elapsed = time.time() - t0
    print(f"Extracted {len(pair_ids)} examples in {elapsed/60:.1f} min")

    save_dict = {f"layer_{l}": np.stack(all_features[l]).astype(np.float32) for l in range(n_layers)}
    save_dict["pair_ids"] = np.array(pair_ids)
    save_dict["perturbation_strategies"] = np.array(strategies)
    save_dict["perturbation_points"] = np.array(points)

    np.savez_compressed(OUT_FEATS, **save_dict)
    sz_mb = OUT_FEATS.stat().st_size / (1024 * 1024)
    print(f"Wrote {OUT_FEATS} ({sz_mb:.1f} MB)")
    return save_dict


# ---------------------------------------------------------------------------
# Sanity check: re-extracted overlap rows must match the canonical features
# ---------------------------------------------------------------------------

def sanity_check(layer: int = 10) -> None:
    """Compare layer-`layer` features for the overlap rows between the new
    annot200 extraction and the canonical N=2,000 features file. Same
    model, same input, deterministic forward pass → vectors should match
    within float32 noise (a few ULPs / <1e-4 cosine distance).
    """
    if not OUT_FEATS.exists():
        raise RuntimeError(f"{OUT_FEATS} not found — run extraction first.")
    if not CANONICAL_FEATS.exists():
        raise RuntimeError(f"{CANONICAL_FEATS} not found — nothing to compare against.")

    new = np.load(OUT_FEATS, allow_pickle=True)
    canon = np.load(CANONICAL_FEATS, allow_pickle=True)

    # Canonical features are keyed by (pair_id, perturbation_point); new
    # features add strategy as a third component. To match canonical (which
    # has no strategy column) we accept any annot row whose (id, point) is
    # present in canon and verify identity-style — there can be duplicates
    # in canon for the same (id, point) under different strategies that
    # both were sampled into N=2,000; we match on first-found-canonical.
    new_X = new[f"layer_{layer}"]
    new_ids = new["pair_ids"]
    new_points = new["perturbation_points"]

    canon_X = canon[f"layer_{layer}"]
    canon_ids = canon["pair_ids"]
    canon_points = canon["perturbation_points"]

    canon_index: dict[tuple, int] = {}
    for i in range(len(canon_ids)):
        key = (str(canon_ids[i]), str(canon_points[i]))
        canon_index.setdefault(key, i)

    overlaps = []
    for j in range(len(new_ids)):
        key = (str(new_ids[j]), str(new_points[j]))
        if key in canon_index:
            overlaps.append((j, canon_index[key]))

    print(f"Sanity overlap: {len(overlaps)} rows (annot vs canonical) at layer {layer}")
    if not overlaps:
        print("No overlap rows found — cannot sanity-check. (This is unexpected; investigate composite-key matching.)")
        return

    dists, cosines = [], []
    for j, i in overlaps:
        v_new = new_X[j].astype(np.float64)
        v_can = canon_X[i].astype(np.float64)
        denom = (np.linalg.norm(v_new) * np.linalg.norm(v_can)) or 1.0
        cos = float(np.dot(v_new, v_can) / denom)
        l2 = float(np.linalg.norm(v_new - v_can))
        dists.append(l2)
        cosines.append(cos)

    cos_arr = np.array(cosines)
    l2_arr = np.array(dists)
    print(f"  cosine sim:  mean={cos_arr.mean():.6f}  min={cos_arr.min():.6f}  "
          f"frac<0.999={(cos_arr < 0.999).mean()*100:.1f}%")
    print(f"  L2 distance: mean={l2_arr.mean():.4f}  max={l2_arr.max():.4f}")
    if cos_arr.min() < 0.99:
        print("⚠️  some overlap pairs disagree substantially — extraction logic may have drifted.")
    else:
        print("✅ overlap features agree (cosine ≥ 0.99 on every row).")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sanity-only", action="store_true",
                    help="Skip extraction; just compare existing annot200 features to canonical.")
    ap.add_argument("--sanity-layer", type=int, default=10)
    args = ap.parse_args()

    if not args.sanity_only:
        records = load_annotation_records()
        extract_all(records, device=args.device)

    sanity_check(layer=args.sanity_layer)


if __name__ == "__main__":
    main()
