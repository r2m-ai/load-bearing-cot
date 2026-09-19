"""
Script 13: Prompt-stage activation steering.

Tests whether the faithfulness decision can be overridden by steering
during PROMPT ENCODING (before any CoT generation begins).

Key difference from script 12: the steering hook acts on tokens
[0:prompt_length] (the question/prompt tokens) rather than
[prompt_length:] (the CoT tokens). During autoregressive generation
steps (seq_len=1), the hook does NOT steer, since those tokens are
post-prompt.

Input: data/processed/subclassified_pairs.json, data/processed/error_prop_direction.npy
Output: data/processed/prompt_steering_results.json, logs/13_prompt_steering_*.log
"""

import json
import re
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from collections import Counter

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"13_prompt_steering_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "google/gemma-2-9b-it"


# ---- Answer extraction (same as script 12) ----

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


# ---- Prompt steering hook ----

class PromptSteeringHook:
    """Hook that adds a direction to hidden states during PROMPT encoding only.

    Unlike BatchSteeringHook (script 12) which steers CoT tokens [prompt_length:],
    this hook steers PROMPT tokens [0:prompt_length] and does NOT steer during
    autoregressive generation (seq_len=1).
    """

    def __init__(self, direction: torch.Tensor, alpha: float, prompt_lengths: list[int]):
        self.direction = direction  # (hidden_dim,)
        self.alpha = alpha
        self.prompt_lengths = prompt_lengths  # list of ints, one per batch element
        self.handle = None

    def __call__(self, module, input, output):
        # output can be a tuple (hidden, ...) or just a tensor depending on model config
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        # Handle both 2D (seq, dim) and 3D (batch, seq, dim)
        was_2d = hidden.dim() == 2
        if was_2d:
            hidden = hidden.unsqueeze(0)
        batch, seq_len, dim = hidden.shape

        if seq_len == 1:
            # During autoregressive generation, don't steer (we only steer prompt)
            return output
        else:
            # Initial forward pass -- steer ONLY prompt tokens
            mask = torch.zeros(batch, seq_len, 1, device=hidden.device, dtype=hidden.dtype)
            for i, pl in enumerate(self.prompt_lengths):
                if i < batch:
                    mask[i, :pl, 0] = 1.0  # steer UP TO prompt_length
            hidden = hidden + self.alpha * self.direction.unsqueeze(0).unsqueeze(0) * mask

        if was_2d:
            hidden = hidden.squeeze(0)

        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        else:
            return hidden

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle:
            self.handle.remove()


# ---- Preprocessing ----

def _adaptive_batch_size(max_input_length: int, base_limit: int = 80000) -> int:
    """Compute batch size based on longest sequence length to avoid OOM."""
    return min(16, max(1, base_limit // max(max_input_length, 1)))


def preprocess_pairs(pairs: list[dict], tokenizer) -> list[dict]:
    """Pre-tokenize all pairs once. Builds input = prompt + perturbed CoT prefix for continuation."""
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
        # Strip trailing end-of-turn so the model continues generating
        if input_text.endswith("<end_of_turn>\n"):
            input_text = input_text[:-len("<end_of_turn>\n")]
        elif input_text.endswith("<end_of_turn>"):
            input_text = input_text[:-len("<end_of_turn>")]

        token_length = len(tokenizer.encode(input_text, truncation=True, max_length=2048))

        processed.append({
            **pair,
            "_input_text": input_text,
            "_prompt_length": prompt_length,
            "_token_length": token_length,
        })
    return processed


# ---- Batched generation with prompt steering ----

def _generate_batch(
    model, tokenizer, batch_pairs: list[dict], direction_tensor: torch.Tensor,
    alpha: float, target_layer: int, max_new_tokens: int,
) -> list[dict]:
    """Run batched generation with prompt-stage steering."""
    batch_texts = [p["_input_text"] for p in batch_pairs]

    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        batch_texts, return_tensors="pt", truncation=True,
        max_length=2048, padding=True,
    ).to(model.device)
    tokenizer.padding_side = old_padding_side

    # Compute prompt_lengths adjusted for left-padding
    attention_mask = inputs["attention_mask"]
    seq_len = attention_mask.shape[1]
    adjusted_prompt_lengths = []
    for i, pair in enumerate(batch_pairs):
        actual_length = attention_mask[i].sum().item()
        pad_offset = seq_len - actual_length
        # Prompt tokens run from pad_offset to pad_offset + prompt_length
        # We want to steer [0 : pad_offset + prompt_length] which includes padding
        # but the padding positions have zero attention so steering them is harmless
        adjusted_prompt_lengths.append(pad_offset + pair["_prompt_length"])

    hook = PromptSteeringHook(direction_tensor, alpha, adjusted_prompt_lengths)
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
                "baseline_subtype": pair.get("subtype", ""),
                "perturbation_point": pair.get("perturbation_point", ""),
                "response_length": len(response),
                "response_preview": response[:200],
            })
    finally:
        hook.remove()

    del outputs
    return results


def run_prompt_steering(
    model, tokenizer, pairs: list[dict], direction: np.ndarray,
    target_layer: int, alphas: list[float], max_new_tokens: int = 256,
) -> list[dict]:
    """Run prompt-stage steering across all alphas."""
    direction_tensor = torch.tensor(direction, dtype=model.dtype).to(model.device)

    if not pairs or "_input_text" not in pairs[0]:
        pairs = preprocess_pairs(pairs, tokenizer)

    results = []

    # Sort by token length for efficient batching
    sorted_indices = sorted(range(len(pairs)), key=lambda i: pairs[i].get("_token_length", 0))

    total_runs = len(pairs) * len(alphas)
    pbar = tqdm(total=total_runs, desc="Prompt steering (batched)")

    baseline_cache = {}  # pair_id -> result for alpha=0

    # Process alpha=0 first so we can cache
    sorted_alphas = sorted(alphas, key=lambda a: (a != 0, abs(a)))

    for alpha in sorted_alphas:
        batch_start = 0
        while batch_start < len(sorted_indices):
            lookahead = min(batch_start + 15, len(sorted_indices) - 1)
            max_len = pairs[sorted_indices[lookahead]].get("_token_length", 512)
            batch_size = _adaptive_batch_size(max_len)
            batch_end = min(batch_start + batch_size, len(sorted_indices))

            batch_pair_indices = sorted_indices[batch_start:batch_end]
            batch_pairs = [pairs[i] for i in batch_pair_indices]

            batch_results = _generate_batch(
                model, tokenizer, batch_pairs, direction_tensor,
                alpha, target_layer, max_new_tokens,
            )

            for r in batch_results:
                results.append(r)
                if alpha == 0:
                    pid = r["id"] + "_" + r.get("perturbation_point", "")
                    baseline_cache[pid] = r

            pbar.update(len(batch_pairs))
            batch_start = batch_end

    pbar.close()
    return results


# ---- Main ----

def main():
    parser = argparse.ArgumentParser(description="Prompt-stage activation steering")
    parser.add_argument("--max-examples", type=int, default=200,
                        help="Total examples to steer (balanced across types)")
    parser.add_argument("--target-layer", type=int, default=21,
                        help="Layer to intervene on (default: 21)")
    parser.add_argument("--alphas", type=str, default="-10,-5,0,5,10",
                        help="Alpha values to test")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]

    # Load data
    with open(PROCESSED_DIR / "subclassified_pairs.json") as f:
        all_pairs = json.load(f)

    clear_pairs = [p for p in all_pairs if p.get("label_3class") in
                   ("silent_bypass", "self_correction", "error_propagation")]
    logging.info(f"Clear pairs: {len(clear_pairs)}")

    # Load or extract direction
    direction_path = PROCESSED_DIR / "error_prop_direction.npy"
    direction = None
    if direction_path.exists():
        direction = np.load(direction_path)
        logging.info(f"Loaded direction from {direction_path} (shape={direction.shape})")
    else:
        logging.info("Direction file not found — will extract after model loads")

    # Select balanced examples: ~67 TYPE_A, ~67 TYPE_B, ~67 TYPE_C
    per_type = args.max_examples // 3
    steering_pairs = []
    for subtype in ["TYPE_A", "TYPE_B", "TYPE_C"]:
        subset = [p for p in clear_pairs if p.get("subtype") == subtype]
        n = min(per_type, len(subset))
        steering_pairs.extend(subset[:n])

    type_counts = Counter(p.get("subtype") for p in steering_pairs)
    logging.info(f"Steering on {len(steering_pairs)} examples "
                 f"(A={type_counts.get('TYPE_A', 0)}, "
                 f"B={type_counts.get('TYPE_B', 0)}, "
                 f"C={type_counts.get('TYPE_C', 0)})")

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

    # Extract direction if not loaded
    if direction is None:
        logging.info("Extracting error propagation direction from model...")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        dir_pairs = clear_pairs[:500]
        features, labels = [], []
        target_layer = args.target_layer
        tokenizer.padding_side = "left"
        for p in tqdm(dir_pairs, desc="Extracting direction"):
            cot = p.get("perturbed_cot", "")
            if not cot:
                continue
            messages = [
                {"role": "user", "content": p["prompt"]},
                {"role": "assistant", "content": cot},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
            prompt_msgs = [{"role": "user", "content": p["prompt"]}]
            prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
            prompt_len = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[target_layer + 1].squeeze(0)
            cot_h = h[prompt_len:]
            if cot_h.shape[0] == 0:
                continue
            mid = cot_h.shape[0] // 2
            w = max(1, cot_h.shape[0] // 6)
            feat = cot_h[max(0, mid-w):min(cot_h.shape[0], mid+w)].float().mean(dim=0).cpu().numpy()
            features.append(feat)
            labels.append(1 if p.get("subtype") == "TYPE_C" else 0)
            del out
        tokenizer.padding_side = "right"

        X = np.array(features)
        y = np.array(labels)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=42, class_weight="balanced")
        clf.fit(X_s, y)
        direction = clf.coef_[0] / scaler.scale_
        direction = direction / np.linalg.norm(direction)
        np.save(direction_path, direction)
        logging.info(f"Direction extracted and saved (probe acc={clf.score(X_s, y):.3f})")

    # Pre-tokenize
    steering_pairs = preprocess_pairs(steering_pairs, tokenizer)

    # Run experiment
    t0 = time.time()
    results = run_prompt_steering(
        model, tokenizer, steering_pairs, direction,
        args.target_layer, alphas,
    )
    elapsed = time.time() - t0
    logging.info(f"Prompt steering done: {len(results)} results in {elapsed:.0f}s")

    # Save results
    results_path = PROCESSED_DIR / "prompt_steering_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Saved results to {results_path}")

    # ---- Analysis ----
    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print("PROMPT-STAGE STEERING: Error rate by alpha")
    print(f"{'='*60}")

    for alpha in sorted(alphas):
        sub = df[df["alpha"] == alpha]
        if sub.empty:
            continue
        error_rate = 1 - sub["is_correct"].mean()
        n = len(sub)
        print(f"  alpha={alpha:+6.1f}: error_rate={error_rate:.3f} (n={n})")

    print(f"\n{'='*60}")
    print("ERROR RATE BY TYPE x ALPHA")
    print(f"{'='*60}")

    for subtype in ["TYPE_A", "TYPE_B", "TYPE_C"]:
        sub = df[df["baseline_subtype"] == subtype]
        if sub.empty:
            continue
        type_label = {"TYPE_A": "Silent Bypass", "TYPE_B": "Self-Correction",
                      "TYPE_C": "Error Propagation"}.get(subtype, subtype)
        print(f"\n{subtype} ({type_label}):")
        for alpha in sorted(alphas):
            alpha_sub = sub[sub["alpha"] == alpha]
            if alpha_sub.empty:
                continue
            error_rate = 1 - alpha_sub["is_correct"].mean()
            print(f"  alpha={alpha:+6.1f}: error_rate={error_rate:.3f} (n={len(alpha_sub)})")

    # ---- Comparison with mid-generation steering (v7 null hypothesis) ----
    print(f"\n{'='*60}")
    print("COMPARISON: Prompt steering vs mid-generation steering")
    print(f"{'='*60}")

    # Load v7 results if available
    v7_path = PROCESSED_DIR / "intervention_results.json"
    if v7_path.exists():
        with open(v7_path) as f:
            v7_results = json.load(f)
        v7_df = pd.DataFrame(v7_results)
        v7_ep = v7_df[v7_df["direction"] == "error_prop"] if "direction" in v7_df.columns else v7_df

        print(f"\n{'Alpha':>8}  {'Prompt Steer':>14}  {'CoT Steer (v7)':>14}  {'Delta':>8}")
        print("-" * 50)
        for alpha in sorted(alphas):
            prompt_sub = df[df["alpha"] == alpha]
            v7_sub = v7_ep[v7_ep["alpha"] == alpha] if not v7_ep.empty else pd.DataFrame()

            p_err = 1 - prompt_sub["is_correct"].mean() if not prompt_sub.empty else float("nan")
            v7_err = 1 - v7_sub["is_correct"].mean() if not v7_sub.empty else float("nan")
            delta = p_err - v7_err if not (pd.isna(p_err) or pd.isna(v7_err)) else float("nan")

            print(f"  {alpha:+6.1f}  {p_err:>14.3f}  {v7_err:>14.3f}  {delta:>+8.3f}")
    else:
        logging.info(f"v7 results not found at {v7_path}, skipping comparison")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    baseline_sub = df[df["alpha"] == 0]
    baseline_error = 1 - baseline_sub["is_correct"].mean() if not baseline_sub.empty else float("nan")
    max_alpha = max(alphas)
    min_alpha = min(alphas)
    max_sub = df[df["alpha"] == max_alpha]
    min_sub = df[df["alpha"] == min_alpha]
    max_error = 1 - max_sub["is_correct"].mean() if not max_sub.empty else float("nan")
    min_error = 1 - min_sub["is_correct"].mean() if not min_sub.empty else float("nan")

    print(f"Baseline error rate (alpha=0):         {baseline_error:.3f}")
    print(f"Max positive steering (alpha={max_alpha:+.0f}):    {max_error:.3f}")
    print(f"Max negative steering (alpha={min_alpha:+.0f}):   {min_error:.3f}")
    print(f"Total wall time:                       {elapsed:.0f}s")

    if not pd.isna(baseline_error):
        induction = max_error - baseline_error
        suppression = baseline_error - min_error

        if induction > 0.05:
            print(f"\n-> Positive prompt steering INDUCES errors (+{induction:.3f})")
        else:
            print(f"\n-> Positive prompt steering has weak/no effect ({induction:+.3f})")

        if suppression > 0.05:
            print(f"-> Negative prompt steering SUPPRESSES errors (-{suppression:.3f})")
        else:
            print(f"-> Negative prompt steering has weak/no effect ({suppression:+.3f})")

        if abs(induction) < 0.03 and abs(suppression) < 0.03:
            print(f"-> CONCLUSION: Prompt-stage steering does NOT override faithfulness decisions")
            print(f"   (consistent with the decision being made DURING CoT, not at prompt encoding)")
        elif induction > 0.05 or suppression > 0.05:
            print(f"-> CONCLUSION: Prompt-stage steering CAN influence faithfulness")
            print(f"   (suggests the decision is partially set during prompt encoding)")


if __name__ == "__main__":
    main()
