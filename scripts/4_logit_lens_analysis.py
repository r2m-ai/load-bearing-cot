"""
Script 4: Logit Lens Analysis — Figure 1.

For each example, projects hidden states through the LM head at each
token position and layer to see when the model "decides" the answer.

Optimizations vs original:
- Only compute logit for the ANSWER TOKEN, not full vocab (256K → 1 dot product)
- Batch multiple examples on GPU
- Keep lm_head row vector on GPU, stream hidden states in batches

Output: figures/figure1_logit_lens.png
"""

import json
import argparse
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HIDDEN_DIR = PROCESSED_DIR / "hidden_states"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_lm_head(device: str = "cpu"):
    """Load LM head weights and optional final layer norm."""
    lm_head = torch.load(HIDDEN_DIR / "lm_head_weight.pt", weights_only=True).float().to(device)

    norm_path = HIDDEN_DIR / "final_norm_weight.pt"
    final_norm = None
    if norm_path.exists():
        final_norm = torch.load(norm_path, weights_only=True).float().to(device)

    return lm_head, final_norm


def compute_answer_logit_curve_fast(
    hidden_states: torch.Tensor,
    answer_head_vector: torch.Tensor,
    final_norm: torch.Tensor | None,
    prompt_length: int,
    n_bins: int = 20,
    layer: int = -2,
) -> np.ndarray:
    """
    Compute the answer token's logit at evenly-spaced points through the CoT.

    Instead of projecting through the FULL lm_head (256K vocab), we only
    dot-product with the single answer token's embedding vector.
    This is ~256K times less compute per position.
    """
    # Get the specified layer, CoT portion only
    h = hidden_states[layer]  # (seq_len, hidden_dim)
    cot_h = h[prompt_length:]  # (cot_len, hidden_dim)
    cot_len = cot_h.shape[0]

    if cot_len == 0:
        return np.zeros(n_bins)

    # Apply RMSNorm if available
    if final_norm is not None:
        rms = torch.sqrt(torch.mean(cot_h ** 2, dim=-1, keepdim=True) + 1e-6)
        cot_h = cot_h / rms * final_norm.unsqueeze(0)

    # Dot product with just the answer token's embedding: (cot_len, dim) @ (dim,) → (cot_len,)
    answer_logits = (cot_h @ answer_head_vector).cpu().numpy()

    # Bin into n_bins evenly spaced points
    bin_indices = np.linspace(0, cot_len - 1, n_bins).astype(int)
    return answer_logits[bin_indices]


def main():
    parser = argparse.ArgumentParser(description="Logit lens analysis")
    parser.add_argument("--layer", type=int, default=-2)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for computation (default: cuda)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load metadata
    with open(HIDDEN_DIR / "metadata.json") as f:
        metadata = json.load(f)
    print(f"Loaded metadata for {len(metadata)} pairs")

    # Load LM head on GPU — only need to do this once
    lm_head, final_norm = load_lm_head(device)
    print(f"LM head shape: {lm_head.shape} (on {device})")

    # Load paired data for answer info
    with open(PROCESSED_DIR / "faithful_unfaithful_pairs.json") as f:
        pairs = json.load(f)
    pairs_by_id = {p["id"]: p for p in pairs}

    # Load tokenizer for answer token lookup
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")

    # Pre-compute answer token IDs and their head vectors
    print("Pre-computing answer head vectors...")
    example_data = []  # (meta, answer_head_vector)
    for meta in metadata:
        pair = pairs_by_id.get(meta["id"])
        if pair is None:
            continue

        if pair["source"] == "gsm8k":
            answer_str = pair["reference_answer"]
        else:
            answer_str = pair.get("correct_letter", pair["reference_answer"])

        tokens = tokenizer.encode(str(answer_str), add_special_tokens=False)
        if not tokens:
            continue

        answer_token_id = tokens[0]
        # Extract just this token's row from lm_head: (hidden_dim,)
        answer_head_vector = lm_head[answer_token_id]
        example_data.append((meta, answer_head_vector))

    print(f"Processing {len(example_data)} pairs...")

    faithful_curves = []
    unfaithful_curves = []

    t0 = time.time()
    for meta, answer_head_vector in tqdm(example_data, desc="Logit lens"):
        # Process faithful version
        faithful_data = torch.load(
            HIDDEN_DIR / meta["faithful_file"], weights_only=True
        )
        # Move hidden states to GPU for the matmul
        h = faithful_data["hidden_states"].float().to(device)
        faithful_curve = compute_answer_logit_curve_fast(
            h, answer_head_vector, final_norm,
            faithful_data["prompt_length"],
            n_bins=args.n_bins, layer=args.layer,
        )
        faithful_curves.append(faithful_curve)
        del h

        # Process unfaithful version
        unfaithful_data = torch.load(
            HIDDEN_DIR / meta["unfaithful_file"], weights_only=True
        )
        h = unfaithful_data["hidden_states"].float().to(device)
        unfaithful_curve = compute_answer_logit_curve_fast(
            h, answer_head_vector, final_norm,
            unfaithful_data["prompt_length"],
            n_bins=args.n_bins, layer=args.layer,
        )
        unfaithful_curves.append(unfaithful_curve)
        del h

    elapsed = time.time() - t0
    print(f"Computed {len(example_data)} pairs in {elapsed:.1f}s ({elapsed/len(example_data):.3f}s/pair)")

    faithful_curves = np.array(faithful_curves)
    unfaithful_curves = np.array(unfaithful_curves)

    print(f"\nFaithful curves: {faithful_curves.shape}")
    print(f"Unfaithful curves: {unfaithful_curves.shape}")

    # Compute statistics
    faithful_mean = np.mean(faithful_curves, axis=0)
    faithful_std = np.std(faithful_curves, axis=0)
    unfaithful_mean = np.mean(unfaithful_curves, axis=0)
    unfaithful_std = np.std(unfaithful_curves, axis=0)

    x = np.linspace(0, 100, args.n_bins)

    # Plot Figure 1
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, faithful_mean, color="#2196F3", linewidth=2.5, label="Faithful CoT")
    ax.fill_between(
        x,
        faithful_mean - faithful_std,
        faithful_mean + faithful_std,
        alpha=0.2, color="#2196F3",
    )

    ax.plot(x, unfaithful_mean, color="#F44336", linewidth=2.5, label="Unfaithful CoT")
    ax.fill_between(
        x,
        unfaithful_mean - unfaithful_std,
        unfaithful_mean + unfaithful_std,
        alpha=0.2, color="#F44336",
    )

    ax.set_xlabel("% of CoT Tokens Generated")
    ax.set_ylabel("Answer Token Logit")
    ax.set_title("Logit Lens: When Does the Model Decide the Answer?")
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    fig_path = FIGURES_DIR / "figure1_logit_lens.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(FIGURES_DIR / "figure1_logit_lens.pdf", bbox_inches="tight")
    print(f"\nSaved Figure 1 to {fig_path}")

    # Save raw data
    np.savez(
        FIGURES_DIR / "logit_lens_data.npz",
        faithful_curves=faithful_curves,
        unfaithful_curves=unfaithful_curves,
        x_pct=x,
    )
    print(f"Saved raw curve data to {FIGURES_DIR / 'logit_lens_data.npz'}")

    # Summary
    print(f"\n{'='*50}")
    print(f"LOGIT LENS SUMMARY")
    print(f"{'='*50}")
    print(f"Faithful — early (0-25%) mean logit:  {np.mean(faithful_mean[:args.n_bins//4]):.3f}")
    print(f"Faithful — late (75-100%) mean logit:  {np.mean(faithful_mean[3*args.n_bins//4:]):.3f}")
    print(f"Unfaithful — early (0-25%) mean logit: {np.mean(unfaithful_mean[:args.n_bins//4]):.3f}")
    print(f"Unfaithful — late (75-100%) mean logit: {np.mean(unfaithful_mean[3*args.n_bins//4:]):.3f}")

    early_gap = np.mean(unfaithful_mean[:args.n_bins//4]) - np.mean(faithful_mean[:args.n_bins//4])
    print(f"\nEarly logit gap (unfaithful - faithful): {early_gap:.3f}")
    if early_gap > 0:
        print("→ Unfaithful CoT shows higher early answer logit (as hypothesized)")
    else:
        print("→ Unexpected: faithful CoT shows higher early answer logit")


if __name__ == "__main__":
    main()
