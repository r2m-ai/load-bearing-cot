"""
Script 3: Extract hidden states from all layers during CoT generation.

For each faithful/unfaithful pair, runs a forward pass through the model
and saves hidden states at all layers. These are used for:
- Logit lens analysis (Script 4)
- Linear probe training (Script 5)

H100 Optimizations:
- Flash Attention 2 (Hopper-optimized)
- TF32 matmul for any fp32 residual ops
- bfloat16 storage (saves disk, native H100 precision)
- Pinned memory for faster GPU→CPU transfers
- Async disk I/O with separate save thread
- use_cache=False (not needed for single forward pass, saves VRAM)

Output: data/processed/hidden_states/ (one .pt file per example)
"""

import json
import argparse
import logging
import time
import os
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
HIDDEN_DIR = PROCESSED_DIR / "hidden_states"
HIDDEN_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"3_hidden_states_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "google/gemma-2-9b-it"


def configure_h100():
    """Set H100-specific CUDA optimizations."""
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.info(f"GPU: {torch.cuda.get_device_name(0)} "
                 f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB)")


def load_model(device: str = "cuda"):
    """Load Gemma-2-9B-IT optimized for H100.

    Note: torch.compile is NOT used here because output_hidden_states=True
    produces dynamic output shapes that break CUDA graph capture.
    """
    logging.info(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    attn_impl = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map=device, attn_implementation=attn_impl,
        )
        logging.info("Using Flash Attention 2")
    except (ImportError, ValueError):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16,
            device_map=device, attn_implementation="sdpa",
        )
        logging.info("Using SDPA")

    model.eval()

    if torch.cuda.is_available():
        logging.info(f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    return model, tokenizer


def get_layer_indices(n_layers: int, mode: str = "all") -> list[int]:
    """
    Get which layer indices to save.
    - "all": every layer
    - "sampled": every 3rd layer + first/last
    - "key": only 5 key layers (early/mid/late)
    """
    if mode == "all":
        return list(range(n_layers))
    elif mode == "sampled":
        indices = set(range(0, n_layers, 3))
        indices.add(0)
        indices.add(n_layers - 1)
        return sorted(indices)
    elif mode == "key":
        return sorted(set([0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]))
    else:
        raise ValueError(f"Unknown layer mode: {mode}")


def save_tensor_async(data: dict, path: Path, executor: ThreadPoolExecutor):
    """Queue a torch.save call on a background thread to overlap with GPU compute."""
    def _save():
        torch.save(data, path)
    executor.submit(_save)


def extract_hidden_states(
    model, tokenizer, prompt: str, cot_response: str,
    layer_indices: list[int] | None = None,
) -> dict:
    """
    Run forward pass and extract hidden states. Uses non-blocking GPU→CPU
    transfer for better overlap with compute.
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": cot_response},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=2048,
    ).to(model.device)

    # Get prompt length
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt")
    prompt_length = prompt_tokens["input_ids"].shape[1]

    # Forward pass — no KV cache needed, saves VRAM
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
        )

    # Extract selected layers, convert to bfloat16, non-blocking CPU transfer
    if layer_indices is not None:
        hidden_states = torch.stack(
            [outputs.hidden_states[i + 1].squeeze(0).to("cpu", non_blocking=True)
             for i in layer_indices],
            dim=0,
        )
        saved_layers = layer_indices
    else:
        hidden_states = torch.stack(
            [h.squeeze(0).to("cpu", non_blocking=True) for h in outputs.hidden_states[1:]],
            dim=0,
        )
        saved_layers = list(range(hidden_states.shape[0]))

    token_ids = inputs["input_ids"].squeeze(0).cpu()

    # Free GPU memory
    del outputs
    torch.cuda.empty_cache()

    return {
        "hidden_states": hidden_states.bfloat16(),  # bfloat16 for H100 consistency
        "token_ids": token_ids,
        "prompt_length": prompt_length,
        "saved_layers": saved_layers,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--layer-mode", type=str, default="all",
                        choices=["all", "sampled", "key"],
                        help="Which layers to save (default: all)")
    parser.add_argument("--save-workers", type=int, default=4,
                        help="Async save threads (default: 4)")
    args = parser.parse_args()

    configure_h100()

    # Load paired data
    pairs_path = PROCESSED_DIR / "faithful_unfaithful_pairs.json"
    with open(pairs_path) as f:
        pairs = json.load(f)
    logging.info(f"Loaded {len(pairs)} pairs")

    if args.max_examples:
        pairs = pairs[:args.max_examples]

    model, tokenizer = load_model(args.device)

    # Determine layers
    n_layers = model.config.num_hidden_layers
    layer_indices = get_layer_indices(n_layers, args.layer_mode)
    logging.info(f"Model: {n_layers} layers, saving {len(layer_indices)} (mode={args.layer_mode})")

    # Save LM head weights for logit lens
    lm_head_weight = model.lm_head.weight.data.cpu().bfloat16()
    torch.save(lm_head_weight, HIDDEN_DIR / "lm_head_weight.pt")
    logging.info(f"Saved LM head weight: {lm_head_weight.shape}")

    if hasattr(model.model, "norm"):
        final_norm_weight = model.model.norm.weight.data.cpu().bfloat16()
        torch.save(final_norm_weight, HIDDEN_DIR / "final_norm_weight.pt")
        logging.info(f"Saved final layer norm weight: {final_norm_weight.shape}")

    # Check for existing progress
    metadata_path = HIDDEN_DIR / "metadata.json"
    metadata = []
    processed_ids = set()
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        processed_ids = {m["id"] for m in metadata}
        logging.info(f"Resuming: {len(processed_ids)} pairs already processed")

    remaining_pairs = [p for p in pairs if p["id"] not in processed_ids]
    logging.info(f"Processing {len(remaining_pairs)} remaining pairs...")

    # Async save executor — overlaps disk I/O with GPU compute
    save_executor = ThreadPoolExecutor(max_workers=args.save_workers)

    total_time = 0
    pbar = tqdm(remaining_pairs, desc="Extracting", initial=len(processed_ids), total=len(pairs))

    for pair in pbar:
        t0 = time.time()

        # Faithful version
        faithful_result = extract_hidden_states(
            model, tokenizer, pair["prompt"], pair["faithful_cot"],
            layer_indices=layer_indices,
        )
        faithful_path = HIDDEN_DIR / f"{pair['id']}_faithful.pt"
        save_tensor_async({
            "hidden_states": faithful_result["hidden_states"],
            "token_ids": faithful_result["token_ids"],
            "prompt_length": faithful_result["prompt_length"],
            "saved_layers": faithful_result["saved_layers"],
        }, faithful_path, save_executor)

        # Unfaithful version
        unfaithful_result = extract_hidden_states(
            model, tokenizer, pair["prompt"], pair["perturbed_cot"],
            layer_indices=layer_indices,
        )
        unfaithful_path = HIDDEN_DIR / f"{pair['id']}_unfaithful.pt"
        save_tensor_async({
            "hidden_states": unfaithful_result["hidden_states"],
            "token_ids": unfaithful_result["token_ids"],
            "prompt_length": unfaithful_result["prompt_length"],
            "saved_layers": unfaithful_result["saved_layers"],
        }, unfaithful_path, save_executor)

        elapsed = time.time() - t0
        total_time += elapsed

        metadata.append({
            "id": pair["id"],
            "source": pair["source"],
            "label": pair["label"],
            "faithful_file": faithful_path.name,
            "unfaithful_file": unfaithful_path.name,
            "faithful_cot_length": faithful_result["token_ids"].shape[0] - faithful_result["prompt_length"],
            "unfaithful_cot_length": unfaithful_result["token_ids"].shape[0] - unfaithful_result["prompt_length"],
            "extraction_time_s": round(elapsed, 2),
        })

        # Metadata checkpoint every 25 pairs
        if len(metadata) % 25 == 0:
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            new_count = len(metadata) - len(processed_ids)
            avg_time = total_time / new_count
            remaining_est = avg_time * (len(pairs) - len(metadata))
            pbar.set_postfix({
                "avg": f"{avg_time:.1f}s/pair",
                "eta": f"{remaining_est/60:.0f}m",
                "mem": f"{torch.cuda.memory_allocated()/1024**3:.0f}G",
            })

    pbar.close()

    # Wait for all async saves to complete
    logging.info("Waiting for async saves to complete...")
    save_executor.shutdown(wait=True)

    # Final metadata save
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save extraction config
    config = {
        "model_id": MODEL_ID,
        "n_model_layers": n_layers,
        "saved_layers": layer_indices,
        "layer_mode": args.layer_mode,
        "n_pairs": len(metadata),
        "dtype": "bfloat16",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(HIDDEN_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*50}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"Processed: {len(metadata)} pairs ({len(metadata)*2} forward passes)")
    print(f"Layers saved: {len(layer_indices)}/{n_layers} (mode={args.layer_mode})")
    print(f"Storage dtype: bfloat16")

    if metadata:
        # Wait a moment for last file to flush
        sample_file = HIDDEN_DIR / metadata[0]["faithful_file"]
        if sample_file.exists():
            sample_size = sample_file.stat().st_size / (1024 * 1024)
            total_est = sample_size * len(metadata) * 2
            print(f"Per file: ~{sample_size:.1f} MB")
            print(f"Total storage: ~{total_est:.0f} MB ({total_est/1024:.1f} GB)")

    if total_time > 0:
        new_count = len(metadata) - len(processed_ids)
        if new_count > 0:
            print(f"Avg time per pair: {total_time/new_count:.1f}s")
            print(f"Total extraction time: {total_time/60:.1f} min")

    if torch.cuda.is_available():
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.1f} GB")


if __name__ == "__main__":
    main()
