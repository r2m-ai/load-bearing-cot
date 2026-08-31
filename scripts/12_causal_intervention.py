"""
Script 12 (v7): Causal intervention via activation steering.

Takes the error propagation direction from the linear probe,
injects it into the model's hidden states during generation,
and measures whether error propagation rate changes.

If steering TOWARD the direction increases errors -> causal evidence.
If steering AWAY decreases errors -> bidirectional control.

Input: data/processed/subclassified_pairs.json, model
Output: figures/figure9_dose_response.png, figure10_layer_intervention.png

Optimized for H100 GPU throughput via batched generation and batched
direction extraction.
"""

import json
import re
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from collections import Counter
from functools import partial

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"12_intervention_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "google/gemma-2-9b-it"


# ---- Answer extraction (same as other scripts) ----

def extract_answer(response: str, source: str) -> str | None:
    if source == "gsm8k":
        match = re.search(r"Answer:\s*\$?([\d,]+\.?\d*)", response)
        if match:
            return match.group(1).replace(",", "")
        numbers = re.findall(r"[\d,]+\.?\d*", response)
        if numbers:
            return numbers[-1].replace(",", "")
    elif source == "bbh":
        match = re.search(r"Answer:\s*(.+?)(?:\n|$)", response)
        if match:
            return match.group(1).strip()
    else:  # mmlu
        match = re.search(r"Answer:\s*\(?([A-Da-d])\)?", response)
        if match:
            return match.group(1).upper()
        match = re.search(r"\b([A-Da-d])\b", response)
        if match:
            return match.group(1).upper()
    return None


def check_correct(extracted: str | None, pair: dict) -> bool:
    if extracted is None:
        return False
    source = pair.get("source", "")
    ref = pair.get("reference_answer", "")
    if source == "gsm8k":
        try:
            return float(extracted) == float(ref.replace(",", ""))
        except ValueError:
            return False
    elif source == "mmlu":
        return extracted.upper() == pair.get("correct_letter", "").upper()
    else:  # bbh
        return extracted.lower().strip() == ref.lower().strip()


# ---- Batched direction extraction ----

def _adaptive_batch_size(max_input_length: int, base_limit: int = 80000) -> int:
    """Compute batch size based on longest sequence length to avoid OOM."""
    return min(16, max(1, base_limit // max(max_input_length, 1)))


def extract_direction(model, tokenizer, pairs: list[dict], target_layer: int) -> tuple[np.ndarray, StandardScaler]:
    """
    Extract the error propagation direction by training a logistic regression
    on RAW hidden states (no PCA) at the perturbation point.

    Batched version: processes multiple examples per forward pass.

    Returns: (unit direction vector in hidden_dim space, scaler).
    """
    logging.info(f"Extracting error propagation direction from layer {target_layer}...")

    # Prepare all inputs
    all_input_texts = []
    all_prompt_lengths = []
    valid_pairs = []

    for pair in pairs:
        perturbed_cot = pair.get("perturbed_cot", "")
        if not perturbed_cot:
            continue

        messages = [
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": perturbed_cot},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        prompt_messages = [{"role": "user", "content": pair["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

        all_input_texts.append(input_text)
        all_prompt_lengths.append(prompt_length)
        valid_pairs.append(pair)

    features = []
    labels = []

    # Process in batches
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Sort by length for more efficient batching
    token_lengths = [len(tokenizer.encode(t, truncation=True, max_length=2048)) for t in all_input_texts]
    sorted_indices = sorted(range(len(all_input_texts)), key=lambda i: token_lengths[i])

    pbar = tqdm(total=len(sorted_indices), desc="Extracting direction (batched)")
    idx = 0
    while idx < len(sorted_indices):
        # Determine batch size based on longest sequence in upcoming examples
        max_len = token_lengths[sorted_indices[min(idx + 15, len(sorted_indices) - 1)]]
        batch_size = _adaptive_batch_size(max_len)
        batch_indices = sorted_indices[idx:idx + batch_size]
        idx += len(batch_indices)

        batch_texts = [all_input_texts[i] for i in batch_indices]
        batch_prompt_lengths = [all_prompt_lengths[i] for i in batch_indices]
        batch_pairs = [valid_pairs[i] for i in batch_indices]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=2048, padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        h_all = outputs.hidden_states[target_layer + 1]  # (batch, seq, hidden_dim), +1 for embedding

        # For left-padded inputs, compute actual start positions
        attention_mask = inputs["attention_mask"]  # (batch, seq)
        seq_len = attention_mask.shape[1]

        for b in range(len(batch_indices)):
            # Number of pad tokens = seq_len - actual_length
            actual_length = attention_mask[b].sum().item()
            pad_offset = seq_len - actual_length
            prompt_length = batch_prompt_lengths[b]
            # In the padded tensor, the prompt starts at pad_offset
            # CoT starts at pad_offset + prompt_length
            cot_start = pad_offset + prompt_length
            cot_end = seq_len  # end of actual tokens

            if cot_start >= cot_end:
                continue

            cot_h = h_all[b, cot_start:cot_end]
            cot_len = cot_h.shape[0]

            # Get perturbation point feature (mean of middle third)
            mid = cot_len // 2
            w = max(1, cot_len // 6)
            feat = cot_h[max(0, mid - w):min(cot_len, mid + w)].float().mean(dim=0).cpu().numpy()

            features.append(feat)
            labels.append(1 if batch_pairs[b].get("subtype") == "TYPE_C" else 0)

        del outputs, h_all
        torch.cuda.empty_cache()
        pbar.update(len(batch_indices))

    pbar.close()
    tokenizer.padding_side = old_padding_side

    X = np.array(features)
    y = np.array(labels)
    logging.info(f"Direction data: {len(X)} examples, {Counter(y)}")

    # Train logistic regression WITHOUT PCA to get raw direction
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight="balanced")
    clf.fit(X_s, y)

    # The direction in original space
    direction = clf.coef_[0] / scaler.scale_
    direction = direction / np.linalg.norm(direction)

    accuracy = clf.score(X_s, y)
    logging.info(f"Direction probe accuracy: {accuracy:.3f}")
    logging.info(f"Direction norm (pre-normalize): {np.linalg.norm(clf.coef_[0]):.3f}")

    return direction, scaler


def extract_directions_all_layers(
    model, tokenizer, pairs: list[dict], layers: list[int]
) -> dict[int, tuple[np.ndarray, StandardScaler]]:
    """
    Extract directions for ALL layers in a single pass using output_hidden_states=True.

    Instead of running N_layers * N_examples forward passes, runs just N_examples passes
    and extracts hidden states from all requested layers simultaneously.

    Returns: dict mapping layer_idx -> (direction, scaler).
    """
    logging.info(f"Extracting directions for layers {layers} in a single pass...")

    # Prepare inputs (same as extract_direction)
    all_input_texts = []
    all_prompt_lengths = []
    valid_pairs = []

    for pair in pairs:
        perturbed_cot = pair.get("perturbed_cot", "")
        if not perturbed_cot:
            continue

        messages = [
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": perturbed_cot},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        prompt_messages = [{"role": "user", "content": pair["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

        all_input_texts.append(input_text)
        all_prompt_lengths.append(prompt_length)
        valid_pairs.append(pair)

    # Per-layer feature collectors
    layer_features = {l: [] for l in layers}
    all_labels = []

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    token_lengths = [len(tokenizer.encode(t, truncation=True, max_length=2048)) for t in all_input_texts]
    sorted_indices = sorted(range(len(all_input_texts)), key=lambda i: token_lengths[i])

    pbar = tqdm(total=len(sorted_indices), desc="Extracting all-layer directions (batched)")
    idx = 0
    while idx < len(sorted_indices):
        max_len = token_lengths[sorted_indices[min(idx + 15, len(sorted_indices) - 1)]]
        batch_size = _adaptive_batch_size(max_len)
        batch_indices = sorted_indices[idx:idx + batch_size]
        idx += len(batch_indices)

        batch_texts = [all_input_texts[i] for i in batch_indices]
        batch_prompt_lengths = [all_prompt_lengths[i] for i in batch_indices]
        batch_pairs = [valid_pairs[i] for i in batch_indices]

        inputs = tokenizer(
            batch_texts, return_tensors="pt", truncation=True,
            max_length=2048, padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        attention_mask = inputs["attention_mask"]
        seq_len = attention_mask.shape[1]

        for b in range(len(batch_indices)):
            actual_length = attention_mask[b].sum().item()
            pad_offset = seq_len - actual_length
            prompt_length = batch_prompt_lengths[b]
            cot_start = pad_offset + prompt_length
            cot_end = seq_len

            if cot_start >= cot_end:
                continue

            # Only record label once per example (not per layer)
            if len(all_labels) <= len(layer_features[layers[0]]):
                pass  # labels accumulated below

            for layer_idx in layers:
                h = outputs.hidden_states[layer_idx + 1]  # +1 for embedding
                cot_h = h[b, cot_start:cot_end]
                cot_len = cot_h.shape[0]

                mid = cot_len // 2
                w = max(1, cot_len // 6)
                feat = cot_h[max(0, mid - w):min(cot_len, mid + w)].float().mean(dim=0).cpu().numpy()
                layer_features[layer_idx].append(feat)

            all_labels.append(1 if batch_pairs[b].get("subtype") == "TYPE_C" else 0)

        del outputs
        torch.cuda.empty_cache()
        pbar.update(len(batch_indices))

    pbar.close()
    tokenizer.padding_side = old_padding_side

    y = np.array(all_labels)
    logging.info(f"All-layer direction data: {len(y)} examples, {Counter(y)}")

    # Train a logistic regression per layer
    results = {}
    for layer_idx in layers:
        X = np.array(layer_features[layer_idx])
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight="balanced")
        clf.fit(X_s, y)

        direction = clf.coef_[0] / scaler.scale_
        direction = direction / np.linalg.norm(direction)

        accuracy = clf.score(X_s, y)
        logging.info(f"  Layer {layer_idx}: probe accuracy={accuracy:.3f}")

        results[layer_idx] = (direction, scaler)

    return results


# ---- Batched steering hook ----

class BatchSteeringHook:
    """Hook that adds a direction to hidden states during batched generation."""

    def __init__(self, direction: torch.Tensor, alpha: float, prompt_lengths: list[int]):
        self.direction = direction  # (hidden_dim,)
        self.alpha = alpha
        self.prompt_lengths = prompt_lengths  # list of ints, one per batch element
        self.handle = None

    def __call__(self, module, input, output):
        hidden = output[0]  # (batch, seq_len, hidden_dim)
        batch, seq_len, dim = hidden.shape

        if seq_len == 1:
            # During autoregressive generation with KV cache, each step has seq_len=1.
            # These are always generation tokens (past the prompt), so always steer.
            hidden = hidden + self.alpha * self.direction
        else:
            # Initial forward pass processes the full padded sequence.
            # Steer only CoT tokens (after prompt_length) for each example.
            mask = torch.zeros(batch, seq_len, 1, device=hidden.device, dtype=hidden.dtype)
            for i, pl in enumerate(self.prompt_lengths):
                if i < batch and pl < seq_len:
                    mask[i, pl:, 0] = 1.0
            hidden = hidden + self.alpha * self.direction.unsqueeze(0).unsqueeze(0) * mask

        return (hidden,) + output[1:]

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle:
            self.handle.remove()


# ---- Steering experiment (batched) ----

def preprocess_pairs(pairs: list[dict], tokenizer) -> list[dict]:
    """Pre-tokenize all pairs once. Avoids re-tokenizing per alpha/direction."""
    processed = []
    for pair in pairs:
        prompt_messages = [{"role": "user", "content": pair["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_length = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

        prefix = pair.get("prefix", pair.get("perturbed_cot", ""))
        messages = [
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": prefix},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        if input_text.endswith("<end_of_turn>\n"):
            input_text = input_text[:-len("<end_of_turn>\n")]
        elif input_text.endswith("<end_of_turn>"):
            input_text = input_text[:-len("<end_of_turn>")]

        # Pre-compute token length for adaptive batching
        token_length = len(tokenizer.encode(input_text, truncation=True, max_length=2048))

        processed.append({
            **pair,
            "_input_text": input_text,
            "_prompt_length": prompt_length,
            "_token_length": token_length,
        })
    return processed


def _generate_batch(
    model, tokenizer, batch_pairs: list[dict], direction_tensor: torch.Tensor,
    alpha: float, target_layer: int, max_new_tokens: int,
) -> list[dict]:
    """Run batched generation with steering for a group of examples."""
    batch_texts = [p["_input_text"] for p in batch_pairs]

    # Left-pad for batched generation
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        batch_texts, return_tensors="pt", truncation=True,
        max_length=2048, padding=True,
    ).to(model.device)
    tokenizer.padding_side = old_padding_side

    # Compute prompt_lengths adjusted for left-padding
    # For generation with left-padding, the prompt_length in the padded tensor
    # is: pad_offset + original_prompt_length
    attention_mask = inputs["attention_mask"]
    seq_len = attention_mask.shape[1]
    adjusted_prompt_lengths = []
    for i, pair in enumerate(batch_pairs):
        actual_length = attention_mask[i].sum().item()
        pad_offset = seq_len - actual_length
        adjusted_prompt_lengths.append(pad_offset + pair["_prompt_length"])

    hook = BatchSteeringHook(direction_tensor, alpha, adjusted_prompt_lengths)
    hook.register(model.model.layers[target_layer])

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=False, use_cache=True,
            )

        input_len = inputs["input_ids"].shape[1]
        results = []
        for i, pair in enumerate(batch_pairs):
            response = tokenizer.decode(
                outputs[i][input_len:], skip_special_tokens=True
            ).strip()

            extracted = extract_answer(response, pair.get("source", "gsm8k"))
            is_correct = check_correct(extracted, pair)

            results.append({
                "id": pair["id"],
                "source": pair.get("source", ""),
                "alpha": alpha,
                "layer": target_layer,
                "extracted_answer": extracted,
                "reference_answer": pair.get("reference_answer", ""),
                "is_correct": is_correct,
                "baseline_behavior": pair.get("behavior", ""),
                "baseline_subtype": pair.get("subtype", ""),
                "perturbation_point": pair.get("perturbation_point", ""),
                "response_length": len(response),
                "response_preview": response[:200],
            })
    finally:
        hook.remove()

    del outputs
    return results


def run_steering_experiment(
    model, tokenizer, pairs: list[dict], direction: np.ndarray,
    target_layer: int, alphas: list[float], max_new_tokens: int = 256,
) -> list[dict]:
    """
    Batched steering experiment.

    Optimizations over v6:
    1. Pre-tokenize all pairs once (not per alpha)
    2. Run alpha=0 baseline once for error_prop direction (reuse for random)
    3. Batched generation: process multiple examples per model.generate() call
    4. Adaptive batch size based on sequence length
    """
    direction_tensor = torch.tensor(direction, dtype=model.dtype).to(model.device)

    random_dir = np.random.randn(len(direction))
    random_dir = random_dir / np.linalg.norm(random_dir)
    random_tensor = torch.tensor(random_dir, dtype=model.dtype).to(model.device)

    # Pre-tokenize
    if not pairs or "_input_text" not in pairs[0]:
        pairs = preprocess_pairs(pairs, tokenizer)

    results = []
    baseline_cache = {}  # pair_id -> baseline result (alpha=0)

    # Sort alphas so 0 is first -- cache baseline
    sorted_alphas = sorted(alphas, key=lambda a: (a != 0, abs(a)))

    directions = [("error_prop", direction_tensor), ("random", random_tensor)]

    total_runs = len(pairs) * len(sorted_alphas) * len(directions)
    pbar = tqdm(total=total_runs, desc="Steering (batched)")

    for dir_name, dir_tensor in directions:
        for alpha in sorted_alphas:
            # For alpha=0 on random direction, reuse baseline from error_prop
            if alpha == 0 and dir_name == "random":
                for pair in pairs:
                    pid = pair["id"] + "_" + pair.get("perturbation_point", "")
                    if pid in baseline_cache:
                        r = baseline_cache[pid].copy()
                        r["direction"] = "random"
                        results.append(r)
                    pbar.update(1)
                continue

            # Sort pairs by token length for efficient batching
            indexed_pairs = list(enumerate(range(len(pairs))))
            sorted_pair_indices = sorted(range(len(pairs)), key=lambda i: pairs[i].get("_token_length", 0))

            batch_start = 0
            while batch_start < len(sorted_pair_indices):
                # Adaptive batch size based on longest sequence in this batch neighborhood
                lookahead = min(batch_start + 15, len(sorted_pair_indices) - 1)
                max_len = pairs[sorted_pair_indices[lookahead]].get("_token_length", 512)
                batch_size = _adaptive_batch_size(max_len)
                batch_end = min(batch_start + batch_size, len(sorted_pair_indices))

                batch_pair_indices = sorted_pair_indices[batch_start:batch_end]
                batch_pairs = [pairs[i] for i in batch_pair_indices]

                batch_results = _generate_batch(
                    model, tokenizer, batch_pairs, dir_tensor,
                    alpha, target_layer, max_new_tokens,
                )

                for r in batch_results:
                    r["direction"] = dir_name
                    results.append(r)

                    # Cache baseline
                    if alpha == 0 and dir_name == "error_prop":
                        pid = r["id"] + "_" + r.get("perturbation_point", "")
                        baseline_cache[pid] = r

                pbar.update(len(batch_pairs))
                batch_start = batch_end

    pbar.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Causal intervention via activation steering")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["dose_response", "layer_specificity", "cross_dataset", "all"],
                        help="Which experiment to run")
    parser.add_argument("--max-examples", type=int, default=200,
                        help="Max examples per type to steer (default: 200)")
    parser.add_argument("--target-layer", type=int, default=21,
                        help="Primary layer to intervene on (default: 21)")
    parser.add_argument("--alphas", type=str, default="-10,-5,-2,-1,-0.5,0,0.5,1,2,5,10",
                        help="Alpha values to test")
    parser.add_argument("--layer-sweep", type=str, default="0,5,10,15,21,25,30,35,41",
                        help="Layers for specificity experiment")
    parser.add_argument("--n-random-controls", type=int, default=5,
                        help="Number of random direction controls")
    parser.add_argument("--direction-examples", type=int, default=500,
                        help="Examples to use for direction extraction")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    layer_sweep = [int(l) for l in args.layer_sweep.split(",")]
    experiments = [args.experiment] if args.experiment != "all" else ["dose_response", "layer_specificity", "cross_dataset"]

    # Load data
    with open(PROCESSED_DIR / "subclassified_pairs.json") as f:
        all_pairs = json.load(f)

    clear_pairs = [p for p in all_pairs if p.get("label_3class") in
                   ("silent_bypass", "self_correction", "error_propagation")]
    logging.info(f"Clear pairs: {len(clear_pairs)}")

    # Load model
    logging.info(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map=args.device, attn_implementation="flash_attention_2",
        )
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map=args.device, attn_implementation="sdpa",
        )
    model.eval()

    # Global timer for total elapsed
    t_global = time.time()

    # Step 1: Extract direction for primary layer
    direction_pairs = clear_pairs[:args.direction_examples]
    direction, scaler = extract_direction(
        model, tokenizer, direction_pairs, args.target_layer
    )
    logging.info(f"Direction extracted from layer {args.target_layer}")

    # Save direction for reproducibility
    np.save(PROCESSED_DIR / "error_prop_direction.npy", direction)

    # Step 2: Select examples for steering
    # Use a MIX of bypass, self-correction, and error propagation
    steering_pairs = []
    for subtype in ["TYPE_A", "TYPE_B", "TYPE_C"]:
        subset = [p for p in clear_pairs[args.direction_examples:] if p.get("subtype") == subtype]
        n = min(args.max_examples // 3, len(subset))
        steering_pairs.extend(subset[:n])

    logging.info(f"Steering on {len(steering_pairs)} examples "
                 f"(A={sum(1 for p in steering_pairs if p['subtype']=='TYPE_A')}, "
                 f"B={sum(1 for p in steering_pairs if p['subtype']=='TYPE_B')}, "
                 f"C={sum(1 for p in steering_pairs if p['subtype']=='TYPE_C')})")

    # Pre-tokenize steering pairs once for reuse across experiments
    steering_pairs = preprocess_pairs(steering_pairs, tokenizer)

    # Step 3: Run experiments
    all_results = []

    # ---- Experiment A: Dose-Response ----
    if "dose_response" in experiments:
        logging.info("=== Experiment A: Dose-Response ===")
        t0 = time.time()
        results_a = run_steering_experiment(
            model, tokenizer, steering_pairs, direction,
            args.target_layer, alphas,
        )
        all_results.extend(results_a)
        logging.info(f"Dose-response: {len(results_a)} results in {time.time()-t0:.0f}s")

        # Multiple random controls
        for rc in range(1, args.n_random_controls):
            logging.info(f"Random control {rc+1}/{args.n_random_controls}...")
            rand_dir = np.random.randn(len(direction))
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            rand_results = run_steering_experiment(
                model, tokenizer, steering_pairs[:50], rand_dir,
                args.target_layer, [min(alphas), 0, max(alphas)],
            )
            # Mark as additional random controls
            for r in rand_results:
                r["direction"] = f"random_{rc}"
            all_results.extend(rand_results)

    # ---- Experiment B: Layer Specificity ----
    if "layer_specificity" in experiments:
        logging.info("=== Experiment B: Layer Specificity ===")
        layer_subset = steering_pairs[:100]
        layer_alphas = [-5, 0, 5]

        # Extract directions for ALL layers in a single pass (huge speedup)
        layer_directions = extract_directions_all_layers(
            model, tokenizer, direction_pairs[:200], layer_sweep
        )

        for layer_idx in layer_sweep:
            logging.info(f"Layer {layer_idx}...")
            layer_dir, _ = layer_directions[layer_idx]
            layer_results = run_steering_experiment(
                model, tokenizer, layer_subset, layer_dir,
                layer_idx, layer_alphas,
            )
            for r in layer_results:
                r["experiment"] = "layer_specificity"
                r["layer"] = layer_idx
            all_results.extend(layer_results)

    # ---- Experiment E: Cross-Dataset ----
    if "cross_dataset" in experiments:
        logging.info("=== Experiment E: Cross-Dataset Transfer ===")
        # Direction was extracted from mixed data. Test on each source separately.
        cross_alphas = [-5, 0, 5]
        for source in ["gsm8k", "mmlu", "bbh"]:
            source_pairs = [p for p in steering_pairs if p.get("source") == source][:50]
            if len(source_pairs) < 10:
                logging.info(f"  {source}: too few examples ({len(source_pairs)}), skipping")
                continue
            cross_results = run_steering_experiment(
                model, tokenizer, source_pairs, direction,
                args.target_layer, cross_alphas,
            )
            for r in cross_results:
                r["experiment"] = "cross_dataset"
            all_results.extend(cross_results)

    results = all_results
    elapsed_total = time.time() - t_global

    # Save raw results
    results_path = PROCESSED_DIR / "intervention_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"All experiments done: {len(results)} total results in {elapsed_total:.0f}s")

    # ---- Analysis ----
    df = pd.DataFrame(results)

    # Error rate by alpha and direction
    print(f"\n{'='*60}")
    print("DOSE-RESPONSE: Error rate by alpha")
    print(f"{'='*60}")

    for dir_name in ["error_prop", "random"]:
        print(f"\n{dir_name} direction:")
        for alpha in alphas:
            sub = df[(df["alpha"] == alpha) & (df["direction"] == dir_name)]
            if sub.empty:
                continue
            error_rate = 1 - sub["is_correct"].mean()
            n = len(sub)
            print(f"  alpha={alpha:+6.1f}: error_rate={error_rate:.3f} (n={n})")

    # By baseline type
    print(f"\n{'='*60}")
    print("ERROR RATE BY BASELINE TYPE x ALPHA")
    print(f"{'='*60}")

    for subtype in ["TYPE_A", "TYPE_B", "TYPE_C"]:
        sub = df[(df["baseline_subtype"] == subtype) & (df["direction"] == "error_prop")]
        if sub.empty:
            continue
        print(f"\n{subtype}:")
        for alpha in alphas:
            alpha_sub = sub[sub["alpha"] == alpha]
            if alpha_sub.empty:
                continue
            error_rate = 1 - alpha_sub["is_correct"].mean()
            print(f"  alpha={alpha:+6.1f}: error_rate={error_rate:.3f}")

    # ---- Figure 9: Dose-Response ----
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Overall dose-response
    ax = axes[0]
    for dir_name, color, label in [("error_prop", "#F44336", "Error propagation direction"),
                                    ("random", "#9E9E9E", "Random direction")]:
        rates = []
        for alpha in alphas:
            sub = df[(df["alpha"] == alpha) & (df["direction"] == dir_name)]
            rates.append(1 - sub["is_correct"].mean() if len(sub) > 0 else 0)
        ax.plot(alphas, rates, "o-", color=color, linewidth=2.5, markersize=8, label=label)

    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Steering Strength (α)")
    ax.set_ylabel("Error Rate")
    ax.set_title("A) Dose-Response Curve")
    ax.legend()

    # Panel B: By baseline type
    ax = axes[1]
    type_colors = {"TYPE_A": "#F44336", "TYPE_B": "#FF9800", "TYPE_C": "#2196F3"}
    type_names = {"TYPE_A": "Silent Bypass", "TYPE_B": "Self-Correction", "TYPE_C": "Error Propagation"}

    for subtype in ["TYPE_A", "TYPE_B", "TYPE_C"]:
        rates = []
        for alpha in alphas:
            sub = df[(df["alpha"] == alpha) & (df["direction"] == "error_prop") &
                     (df["baseline_subtype"] == subtype)]
            rates.append(1 - sub["is_correct"].mean() if len(sub) > 0 else 0)
        ax.plot(alphas, rates, "o-", color=type_colors[subtype], linewidth=2,
                markersize=6, label=type_names[subtype])

    ax.axvline(x=0, color="gray", linestyle=":", alpha=0.3)
    ax.set_xlabel("Steering Strength (α)")
    ax.set_ylabel("Error Rate")
    ax.set_title("B) By Baseline Behavior Type")
    ax.legend()

    plt.suptitle("Causal Intervention: Activation Steering with Error Propagation Direction", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "figure9_dose_response.png", dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure9_dose_response.pdf", bbox_inches="tight")
    print(f"\nSaved Figure 9")

    # ---- Figure 10: Layer Specificity ----
    layer_df = df[df.get("experiment", "") == "layer_specificity"] if "experiment" in df.columns else pd.DataFrame()
    if not layer_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        layers_tested = sorted(layer_df["layer"].unique())
        for alpha in [-5, 5]:
            rates = []
            for layer in layers_tested:
                sub = layer_df[(layer_df["layer"] == layer) & (layer_df["alpha"] == alpha) &
                               (layer_df["direction"] == "error_prop")]
                rates.append(1 - sub["is_correct"].mean() if len(sub) > 0 else 0)
            color = "#F44336" if alpha > 0 else "#2196F3"
            label = f"α={alpha:+.0f}"
            ax.plot(layers_tested, rates, "o-", color=color, linewidth=2, markersize=8, label=label)

        baseline_sub = layer_df[layer_df["alpha"] == 0]
        if not baseline_sub.empty:
            baseline_rate = 1 - baseline_sub["is_correct"].mean()
            ax.axhline(y=baseline_rate, color="gray", linestyle=":", label=f"Baseline (α=0)")

        ax.set_xlabel("Layer")
        ax.set_ylabel("Error Rate")
        ax.set_title("Layer Specificity: Which Layers Are Causally Involved?")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "figure10_layer_specificity.png", dpi=300, bbox_inches="tight")
        plt.savefig(FIGURES_DIR / "figure10_layer_specificity.pdf", bbox_inches="tight")
        print(f"Saved Figure 10")

    # ---- Table 7 ----
    table_rows = []
    for dir_name in ["error_prop", "random"]:
        for alpha in alphas:
            sub = df[(df["alpha"] == alpha) & (df["direction"] == dir_name)]
            if sub.empty:
                continue
            table_rows.append({
                "Direction": dir_name,
                "Alpha": alpha,
                "Error_Rate": f"{1 - sub['is_correct'].mean():.3f}",
                "N": len(sub),
                "Correct": sub["is_correct"].sum(),
                "Avg_Response_Length": f"{sub['response_length'].mean():.0f}",
            })
    pd.DataFrame(table_rows).to_csv(FIGURES_DIR / "table7_intervention.csv", index=False)
    print(f"Saved Table 7")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    baseline_error = 1 - df[(df["alpha"] == 0) & (df["direction"] == "error_prop")]["is_correct"].mean()
    max_alpha_error = 1 - df[(df["alpha"] == max(alphas)) & (df["direction"] == "error_prop")]["is_correct"].mean()
    min_alpha_error = 1 - df[(df["alpha"] == min(alphas)) & (df["direction"] == "error_prop")]["is_correct"].mean()
    random_max = 1 - df[(df["alpha"] == max(alphas)) & (df["direction"] == "random")]["is_correct"].mean()

    print(f"Baseline error rate (α=0):        {baseline_error:.3f}")
    print(f"Max positive steering (α={max(alphas)}):   {max_alpha_error:.3f}")
    print(f"Max negative steering (α={min(alphas)}):  {min_alpha_error:.3f}")
    print(f"Random direction (α={max(alphas)}):       {random_max:.3f}")
    print(f"Total wall time:                  {elapsed_total:.0f}s")

    induction = max_alpha_error - baseline_error
    suppression = baseline_error - min_alpha_error

    if induction > 0.05:
        print(f"\n→ POSITIVE STEERING INDUCES ERRORS (+{induction:.3f})")
    else:
        print(f"\n→ Positive steering has weak effect ({induction:+.3f})")

    if suppression > 0.05:
        print(f"→ NEGATIVE STEERING SUPPRESSES ERRORS (-{suppression:.3f})")
    else:
        print(f"→ Negative steering has weak effect ({suppression:+.3f})")

    if abs(random_max - baseline_error) < 0.03:
        print(f"→ Random direction has NO effect (control passed)")
    else:
        print(f"→ WARNING: Random direction also affects errors ({random_max - baseline_error:+.3f})")


if __name__ == "__main__":
    main()
