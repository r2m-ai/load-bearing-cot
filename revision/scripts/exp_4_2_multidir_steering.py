"""
exp_4_2_multidir_steering.py — multi-direction activation steering
(revision Theme 4, item 4.2 + 4.4).

Reviewer DzAa #4: single-direction linear steering is underpowered. The claim
"readable but not controllable" overgeneralizes from one intervention class.
This script tests whether steering along multiple directions in a *defensible*
k-dim subspace at layer 21 can flip behavioral type — the committed
multi-direction follow-up.

Two bases (per researcher sign-off 2026-05-19):

  M2 (primary)
    {class-mean-difference direction (Type-C vs non-C)}
    ∪ {top-(k−1) principal axes of the *within-Type-C* covariance, computed
       on residuals after projecting out the mean-difference component}.
    This is the M2 spec from the plan. NOT raw-activation PCA (which would
    return dominant nuisance variance — a strawman null).

  probe (robustness)
    Orthonormal basis from {weights of an L2-regularized 3-class
    multinomial logistic-regression probe} ∪ {top-(k−3) PCs of features
    after projecting out the probe directions} (when k > 3).

For each (basis, k ∈ {2, 4, 8}, α ∈ 11 values), we add  α · Σᵢ dᵢ  to the
residual-stream at layer 21 during forced continuation on the deterministic
198-set (66 examples per behavioral type, drawn from the *same* set as the
single-direction experiment to keep the comparison matched). Generations are
greedy-decoded, regex-scored (is_correct), and rule-based classified into
{TYPE_A bypass, TYPE_B self-correction, TYPE_C error-propagation,
NEEDS_JUDGE, UNCLEAR}. The judge stage re-scores ambiguous cases — item 4.4,
because regex answer-correctness is blind to A↔B flips (same lesson as §3.3).

Reference points (paper, single-direction at layer 21, α ∈ [−10, +10]):
  TYPE_A error rate: 0.0%  (pooled 0/726; exact-binomial UB 0.41%)
  TYPE_B error rate: 0.0%–3.0%  (uninformative; Type B steerability OPEN)
  TYPE_C error rate: 87.9%–97.0%  (at ceiling; Wilson CI [92.0%, 95.5%])

A meaningful multi-direction result would either (a) move Type A off 0% or
move Type C off ceiling for some k (positive — controllability claim must
rescope further), or (b) confirm the null at k ∈ {2, 4, 8} for both bases
(negative — strengthens "not controllable by tested steering"). Lead the
DzAa §5.4 update with whichever side lands.

Stages:
  extract: build the two bases at k_max=8 from a held-out set, save to disk.
  steer  : run 6 dose-response sweeps (3 k × 2 bases) × 11 α × 198 examples;
           rule-based classify continuations.
  judge  : LLM-judge re-score for NEEDS_JUDGE / UNCLEAR (item 4.4).

Usage:
  python revision/scripts/exp_4_2_multidir_steering.py --stage extract
  python revision/scripts/exp_4_2_multidir_steering.py --stage steer
  python revision/scripts/exp_4_2_multidir_steering.py --stage judge \\
         --api-key $ANTHROPIC_API_KEY

Outputs:
  data/processed/multidir_steering_bases.npz
  data/processed/multidir_steering_subclassified.json
  data/processed/multidir_steering_labeled.json
"""

import argparse
import importlib.util
import json
import logging
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "scripts" / "create_perturbations_v8_helpers.py").exists():
            return p
    raise RuntimeError("Could not locate repo root from " + str(start))


ROOT = _find_repo_root(Path(__file__).resolve())
SCRIPTS = ROOT / "scripts"
PROCESSED = ROOT / "data" / "processed"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_DIR / f"exp_4_2_multidir_steering_{datetime.now():%Y%m%d_%H%M%S}.log"),
        logging.StreamHandler(),
    ],
)

MODEL_ID = "google/gemma-2-9b-it"
CACHE_DIR = Path("/workspace/model_cache") if Path("/workspace").exists() else None

TARGET_LAYER = 21
ALPHAS = [-10.0, -5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
K_VALUES = [2, 4, 8]
K_MAX = 8
BASES = ["M2", "probe"]
N_PER_TYPE = 66
DIRECTION_EXAMPLES = 500  # held out for basis training; non-overlapping with the 198-set

IN_PAIRS = PROCESSED / "subclassified_pairs.json"
OUT_BASES = PROCESSED / "multidir_steering_bases.npz"
OUT_SUBCLASSIFIED = PROCESSED / "multidir_steering_subclassified.json"
OUT_LABELED = PROCESSED / "multidir_steering_labeled.json"


# ---- load digit-prefixed pipeline modules by path (matched labeling) ----

def _load_module(filename: str, modname: str):
    spec = importlib.util.spec_from_file_location(modname, SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- answer extraction (verbatim from 12_causal_intervention.py) ----

def extract_answer(response: str, source: str):
    if source == "gsm8k":
        m = re.search(r"Answer:\s*\$?([\d,]+\.?\d*)", response)
        if m:
            return m.group(1).replace(",", "")
        nums = re.findall(r"[\d,]+\.?\d*", response)
        if nums:
            return nums[-1].replace(",", "")
    elif source == "bbh":
        m = re.search(r"Answer:\s*(.+?)(?:\n|$)", response)
        if m:
            return m.group(1).strip()
    else:  # mmlu
        m = re.search(r"Answer:\s*\(?([A-Da-d])\)?", response)
        if m:
            return m.group(1).upper()
        m = re.search(r"\b([A-Da-d])\b", response)
        if m:
            return m.group(1).upper()
    return None


def check_correct(extracted, pair) -> bool:
    if extracted is None:
        return False
    src = pair.get("source", "")
    ref = pair.get("reference_answer", "")
    if src == "gsm8k":
        try:
            return float(extracted) == float(str(ref).replace(",", ""))
        except ValueError:
            return False
    if src == "mmlu":
        return extracted.upper() == pair.get("correct_letter", "").upper()
    return extracted.lower().strip() == str(ref).lower().strip()


# ---- model loading (force HF; we need forward-hook access to hidden states) ----

def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_kwargs = {"cache_dir": str(CACHE_DIR)} if CACHE_DIR else {}
    if CACHE_DIR:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **cache_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    for attn in ["flash_attention_2", "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
                attn_implementation=attn, **cache_kwargs)
            logging.info(f"HF model loaded with attention: {attn}")
            break
        except Exception as e:
            logging.info(f"  {attn} unavailable: {type(e).__name__}")
    else:
        raise RuntimeError("Could not load model")
    model.eval()
    return model, tokenizer


# ---- adaptive batching (verbatim from 12_causal_intervention.py) ----

def _adaptive_batch_size(max_input_length: int, base_limit: int = 240000,
                          cap: int = 48) -> int:
    """
    Heuristic batch size: roughly `base_limit / max_seq_len`, capped.

    Defaults tuned for H200 (141 GB VRAM) running a 9B model in bf16
    (~18 GB weights + ~10 GB scratch + KV cache + activations + steering
    hook overhead). Measured exp_1_1 at batch=16 uses 22 GB, exp_1_3 at
    batch=32 uses 22 GB → plenty of headroom for batch 48-64. Adjust
    downward on H100 (80 GB) — drop cap to 32 and base_limit to 160k.
    """
    return min(cap, max(1, base_limit // max(max_input_length, 1)))


# ---- feature extraction at the perturbation point, single layer ----

def extract_features(model, tokenizer, pairs, layer: int):
    """
    Extract hidden states at `layer` at the perturbation point (mean of the
    middle third of the CoT continuation) for each pair. Returns features
    aligned with subtype labels.

    Mirrors the perturbation-point feature extraction used to train the
    layer-21 logistic-regression direction in the single-direction baseline,
    so the resulting bases live in the same space as the existing
    error-propagation direction.
    """
    import numpy as np
    import torch
    from tqdm import tqdm

    inputs_text, prompt_lens, valid = [], [], []
    for pair in pairs:
        perturbed = pair.get("perturbed_cot", "")
        if not perturbed:
            continue
        messages = [
            {"role": "user", "content": pair["prompt"]},
            {"role": "assistant", "content": perturbed},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": pair["prompt"]}],
            tokenize=False, add_generation_prompt=True)
        plen = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
        inputs_text.append(text)
        prompt_lens.append(plen)
        valid.append(pair)

    feats, labels = [], []
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"

    tlens = [len(tokenizer.encode(t, truncation=True, max_length=2048)) for t in inputs_text]
    order = sorted(range(len(inputs_text)), key=lambda i: tlens[i])

    pbar = tqdm(total=len(order), desc=f"Extract features layer={layer}")
    i = 0
    subtype_to_idx = {"TYPE_A": 0, "TYPE_B": 1, "TYPE_C": 2}
    while i < len(order):
        peek = order[min(i + 15, len(order) - 1)]
        bsz = _adaptive_batch_size(tlens[peek])
        idxs = order[i:i + bsz]
        i += len(idxs)

        batch = tokenizer(
            [inputs_text[k] for k in idxs], return_tensors="pt", truncation=True,
            max_length=2048, padding=True).to(model.device)

        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)

        h = out.hidden_states[layer + 1]  # +1 for embedding layer
        attn = batch["attention_mask"]
        seq_len = attn.shape[1]

        for b, k_idx in enumerate(idxs):
            actual = attn[b].sum().item()
            pad_off = seq_len - actual
            cot_start = pad_off + prompt_lens[k_idx]
            cot_end = seq_len
            if cot_start >= cot_end:
                continue

            cot_h = h[b, cot_start:cot_end]
            cot_len = cot_h.shape[0]
            mid = cot_len // 2
            w = max(1, cot_len // 6)
            feat = cot_h[max(0, mid - w):min(cot_len, mid + w)].float().mean(dim=0).cpu().numpy()

            subtype = valid[k_idx].get("subtype", "")
            if subtype in subtype_to_idx:
                feats.append(feat)
                labels.append(subtype_to_idx[subtype])

        del out
        torch.cuda.empty_cache()
        pbar.update(len(idxs))

    pbar.close()
    tokenizer.padding_side = old_pad
    X = np.array(feats)
    y = np.array(labels)
    logging.info(f"Extracted {len(X)} feature vectors; subtype counts: {Counter(y.tolist())}")
    return X, y


# ---- basis construction ----

def _orthonormalize_rows(M, eps=1e-10):
    """Modified Gram-Schmidt on rows of M; drop near-zero rows."""
    import numpy as np
    out = []
    for v in M:
        for u in out:
            v = v - (v @ u) * u
        n = float(np.linalg.norm(v))
        if n > eps:
            out.append(v / n)
    return np.array(out)


def build_basis_M2(X, y, k_max: int):
    """
    M2 spec basis:
      row 0     = unit (μ_C − μ_non-C)
      rows 1..k-1 = top-(k-1) PCs of the *within-Type-C* covariance computed
                   on residuals  (x_C − μ_C) − ((x_C − μ_C)·d_mean) d_mean.

    Removing the mean-difference component before within-class PCA prevents
    the mean-diff axis from re-appearing as the top PC (would otherwise give
    a redundant rank-1 basis).
    """
    import numpy as np
    label_C = (y == 2)
    if label_C.sum() < 10:
        raise RuntimeError(f"Too few TYPE_C examples for M2 basis: {label_C.sum()}")

    mu_C = X[label_C].mean(axis=0)
    mu_other = X[~label_C].mean(axis=0)
    d_mean = mu_C - mu_other
    n_mean = float(np.linalg.norm(d_mean))
    if n_mean < 1e-10:
        raise RuntimeError("mean-difference is degenerate (~zero)")
    d_mean = d_mean / n_mean

    XC = X[label_C] - mu_C  # within-class centering
    proj = XC @ d_mean[:, None] * d_mean[None, :]  # project onto d_mean
    residuals = XC - proj  # remove d_mean component

    # Top-(k_max-1) PCs of residuals
    U, S, Vt = np.linalg.svd(residuals, full_matrices=False)
    pcs = Vt[: k_max - 1]  # (k_max-1, hidden_dim), unit-norm rows from SVD

    basis = np.vstack([d_mean[None, :], pcs])  # (k_max, hidden_dim)
    basis = _orthonormalize_rows(basis)
    if basis.shape[0] < k_max:
        raise RuntimeError(f"M2 basis only spans {basis.shape[0]} dims < k_max={k_max}")
    logging.info(f"M2 basis built: shape={basis.shape}, top-{min(5, k_max-1)} singular values "
                 f"of within-C residuals: {S[:5].tolist()}")
    return basis[:k_max]


def build_basis_probe(X, y, k_max: int):
    """
    Robustness basis: 3 weight directions from an L2-regularized 3-class
    multinomial logistic-regression probe, orthonormalized; padded with PCs
    of features (after removing the 3 probe directions) when k_max > 3.
    """
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # sklearn 1.5+ removed the `multi_class` kwarg — multinomial is now the
    # default when there are >2 classes. lbfgs solver handles it natively.
    clf = LogisticRegression(
        max_iter=3000, C=1.0, random_state=42,
        solver="lbfgs", class_weight="balanced",
    )
    clf.fit(Xs, y)
    logging.info(f"3-class probe accuracy on training set: {clf.score(Xs, y):.3f}")

    # Bring back to raw feature space: weight / scale, then unit-normalize
    raw_W = clf.coef_ / scaler.scale_[None, :]
    raw_W = raw_W / np.linalg.norm(raw_W, axis=1, keepdims=True).clip(min=1e-10)
    base = _orthonormalize_rows(raw_W)  # up to 3 rows

    if k_max <= base.shape[0]:
        return base[:k_max]

    # Fill remainder with top PCs of features after projecting out base rows
    Xc = X - X.mean(axis=0, keepdims=True)
    for u in base:
        Xc = Xc - (Xc @ u[:, None]) * u[None, :]
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt[: k_max - base.shape[0]]
    full = np.vstack([base, pcs])
    full = _orthonormalize_rows(full)
    if full.shape[0] < k_max:
        raise RuntimeError(f"probe basis only spans {full.shape[0]} dims < k_max={k_max}")
    logging.info(f"probe basis built: shape={full.shape} (3 probe dirs + {k_max-3} fill PCs)")
    return full[:k_max]


# ---- multi-direction steering hook ----

class MultiDirSteeringHook:
    """
    Forward-hook that adds  α · Σᵢ dᵢ  to the layer output at every
    non-prompt token. Replaces the single-direction BatchSteeringHook from
    12_causal_intervention.py:367. Masking semantics are identical (steer
    CoT/continuation tokens only; the autoregressive seq_len=1 path always
    steers because by construction those tokens are past the prompt).

    ON DOSE-RESPONSE COMPARABILITY WITH THE SINGLE-DIRECTION REFERENCE:
    The single-direction experiment used  hidden += α · d  with `d` of
    unit norm, so the intervention magnitude at strength α is exactly
    |α|. Here, with k orthonormal rows in the basis,
    `directions.sum(dim=0)` has magnitude √k, so the intervention
    magnitude at strength α is √k · |α|. This means α=10 in the k=4
    sweep delivers ~2× the residual-stream perturbation of α=10 in the
    single-direction sweep. Report this scaling explicitly in §5.4 so
    reviewers can interpret the dose-response numbers. We keep the sum
    (rather than normalizing to unit norm) because (i) it matches the
    "every direction contributes equally with coefficient α" semantics
    most consistent with the M2-basis interpretation, and (ii)
    normalizing collapses multi-direction back to a single direction
    (the sum's unit vector), which defeats the experiment.
    """

    def __init__(self, directions, alpha: float, prompt_lengths):
        # `directions`: torch.Tensor of shape (k, hidden_dim), unit rows
        # We pre-sum here so the hook does one fused add per token.
        self.directions = directions
        self.summed = directions.sum(dim=0)  # (hidden_dim,)
        self.alpha = alpha
        self.prompt_lengths = prompt_lengths
        self.handle = None

    def __call__(self, module, input, output):
        import torch
        # transformers 4.x returned `tuple(hidden_states, ...)` from decoder
        # layers; 5.x returns the tensor directly. Detect & handle both.
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output  # (batch, seq, hidden_dim)
        batch, seq_len, dim = hidden.shape
        if seq_len == 1:
            hidden = hidden + self.alpha * self.summed
        else:
            mask = torch.zeros(batch, seq_len, 1, device=hidden.device, dtype=hidden.dtype)
            for i, pl in enumerate(self.prompt_lengths):
                if i < batch and pl < seq_len:
                    mask[i, pl:, 0] = 1.0
            hidden = hidden + (self.alpha * self.summed).unsqueeze(0).unsqueeze(0) * mask
        if is_tuple:
            return (hidden,) + output[1:]
        return hidden

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle:
            self.handle.remove()


# ---- preprocess (pre-tokenize prefixes once for reuse across α / basis) ----

def preprocess_198set(pairs, tokenizer):
    """Pre-tokenize each pair's perturbed prefix for forced continuation."""
    out = []
    for pair in pairs:
        prompt = pair["prompt"]
        prefix = pair.get("prefix") or pair.get("perturbed_cot", "")
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        plen = tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]

        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": prefix}],
            tokenize=False, add_generation_prompt=False)
        if text.endswith("<end_of_turn>\n"):
            text = text[: -len("<end_of_turn>\n")]
        elif text.endswith("<end_of_turn>"):
            text = text[: -len("<end_of_turn>")]

        tlen = len(tokenizer.encode(text, truncation=True, max_length=2048))
        out.append({**pair, "_input_text": text, "_prompt_length": plen, "_token_length": tlen})
    return out


# ---- batched steered generation ----

def _generate_batch(model, tokenizer, batch_pairs, hook_directions, alpha, layer,
                    max_new_tokens):
    import torch

    texts = [p["_input_text"] for p in batch_pairs]
    old_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                       max_length=2048, padding=True).to(model.device)
    tokenizer.padding_side = old_pad

    attn = inputs["attention_mask"]
    seq_len = attn.shape[1]
    adj_plens = []
    for i, pair in enumerate(batch_pairs):
        actual = attn[i].sum().item()
        pad_off = seq_len - actual
        adj_plens.append(pad_off + pair["_prompt_length"])

    hook = MultiDirSteeringHook(hook_directions, alpha, adj_plens)
    hook.register(model.model.layers[layer])

    results = []
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                use_cache=True, pad_token_id=tokenizer.pad_token_id)
        in_len = inputs["input_ids"].shape[1]
        for i, pair in enumerate(batch_pairs):
            resp = tokenizer.decode(out[i][in_len:], skip_special_tokens=True).strip()
            ext = extract_answer(resp, pair.get("source", "gsm8k"))
            correct = check_correct(ext, pair)
            results.append({
                "id": pair["id"],
                "source": pair.get("source", ""),
                "alpha": alpha,
                "layer": layer,
                "extracted_answer": ext,
                "reference_answer": pair.get("reference_answer", ""),
                "is_correct": correct,
                "baseline_subtype": pair.get("subtype", ""),
                "baseline_label_3class": pair.get("label_3class", ""),
                "perturbation_point": pair.get("perturbation_point", ""),
                "perturbation_strategy": pair.get("perturbation_strategy", ""),
                # Carried through for 2b_subclassify (reads perturbation_description
                # and correct_letter via extract_perturbed_values, plus original /
                # perturbed step text for downstream judge prompts).
                "perturbation_description": pair.get("perturbation_description", ""),
                "correct_letter": pair.get("correct_letter", ""),
                "original_step": pair.get("original_step", ""),
                "perturbed_step": pair.get("perturbed_step", ""),
                "question": pair.get("question", ""),
                "prompt": pair.get("prompt", ""),
                "response_length": len(resp),
                "continuation": resp,
            })
    finally:
        hook.remove()
    del out
    return results


def run_dose_response(model, tokenizer, pairs, basis_full, k, alphas, layer):
    """Single dose-response sweep with top-k rows of basis_full summed."""
    import torch
    from tqdm import tqdm

    directions = torch.tensor(basis_full[:k], dtype=model.dtype).to(model.device)

    results = []
    order = sorted(range(len(pairs)), key=lambda i: pairs[i].get("_token_length", 512))
    for alpha in alphas:
        pbar = tqdm(total=len(order), desc=f"k={k} α={alpha:+.1f}")
        i = 0
        while i < len(order):
            peek = order[min(i + 15, len(order) - 1)]
            bsz = _adaptive_batch_size(pairs[peek].get("_token_length", 512))
            idxs = order[i:i + bsz]
            i += len(idxs)
            batch_pairs = [pairs[j] for j in idxs]
            batch_results = _generate_batch(
                model, tokenizer, batch_pairs, directions, float(alpha),
                layer, max_new_tokens=256)
            for r in batch_results:
                r["k"] = k
                results.append(r)
            pbar.update(len(idxs))
        pbar.close()
    return results


# ---- deterministic 198-set selection ----

def select_198set(clear_pairs, n_per_type, exclude_first):
    """
    Same protocol as the single-direction experiment: drop the first
    `exclude_first` pairs (used for basis/direction training to avoid leakage)
    then take the first `n_per_type` of each subtype, in the JSON's natural
    order. Deterministic; matches scripts/12_causal_intervention.py:657-668.
    """
    pool = clear_pairs[exclude_first:]
    set198 = []
    for st in ["TYPE_A", "TYPE_B", "TYPE_C"]:
        sub = [p for p in pool if p.get("subtype") == st][:n_per_type]
        set198.extend(sub)
    return set198


# ---- rule-based subclassification of steered continuations ----

def rule_classify(rec, sub2b):
    """
    Run the same dataset-specific rule-based subclassifier used by the main
    pipeline (matched to scripts/2b_subclassify.py). 2b only exposes
    classify_gsm8k and classify_mmlu; per exp_1_3_confound_numerical.py:488
    BBH is routed through classify_mmlu (same option-letter logic). The
    behavior field is set following the same convention as exp_1_1:
    self_corrects when the regex answer is correct, propagates_error when
    we extracted *some* answer that doesn't match, else unclear.
    """
    src = rec.get("source", "")
    cont = rec.get("continuation", "")
    pair_like = {
        **rec,
        "continuation": cont,
        "perturbed_answer": rec.get("extracted_answer"),
        "perturbed_completion": cont,
        "behavior": (
            "self_corrects" if rec.get("is_correct")
            else ("propagates_error" if rec.get("extracted_answer") else "unclear")),
    }
    if src == "gsm8k":
        subtype, reason = sub2b.classify_gsm8k(pair_like)
    else:
        # MMLU and BBH both use the option-letter classifier (matches main pipeline).
        subtype, reason = sub2b.classify_mmlu(pair_like)
    label_3class = {
        "TYPE_A": "silent_bypass", "TYPE_B": "self_correction",
        "TYPE_C": "error_propagation",
    }.get(subtype, "unclear")
    return subtype, reason, label_3class


# ---- stage: extract ----

def stage_extract(args):
    import numpy as np

    blob = json.load(open(IN_PAIRS))
    all_pairs = blob if isinstance(blob, list) else blob.get("records", blob)
    clear = [p for p in all_pairs if p.get("label_3class") in
             ("silent_bypass", "self_correction", "error_propagation")]
    logging.info(f"Loaded {len(all_pairs)} pairs; {len(clear)} clear-labeled")

    # Held-out set for basis training (matches single-direction experiment's
    # direction-extraction split: first `direction_examples` pairs).
    train = clear[:args.direction_examples]
    logging.info(f"Basis training set: {len(train)} pairs from clear[:{args.direction_examples}]")

    model, tokenizer = load_model()
    X, y = extract_features(model, tokenizer, train, args.target_layer)

    basis_M2 = build_basis_M2(X, y, K_MAX)
    basis_probe = build_basis_probe(X, y, K_MAX)

    np.savez(
        OUT_BASES,
        basis_M2=basis_M2,
        basis_probe=basis_probe,
        layer=np.array([args.target_layer]),
        n_train=np.array([len(X)]),
        n_per_type=np.array([
            int((y == 0).sum()), int((y == 1).sum()), int((y == 2).sum())]),
    )
    logging.info(f"Saved bases -> {OUT_BASES}")
    print(f"\nBasis shapes: M2={basis_M2.shape}  probe={basis_probe.shape}")
    print(f"Layer = {args.target_layer}; trained on {len(X)} examples "
          f"(A={int((y==0).sum())}, B={int((y==1).sum())}, C={int((y==2).sum())})")


# ---- stage: steer ----

def stage_steer(args):
    import numpy as np

    if not OUT_BASES.exists():
        raise SystemExit(f"{OUT_BASES} not found — run --stage extract first.")
    bases = np.load(OUT_BASES)
    basis_M2 = bases["basis_M2"]
    basis_probe = bases["basis_probe"]
    saved_layer = int(bases["layer"][0])
    if saved_layer != args.target_layer:
        logging.warning(f"Loaded bases trained at layer {saved_layer}; "
                        f"--target-layer={args.target_layer} differs")

    blob = json.load(open(IN_PAIRS))
    all_pairs = blob if isinstance(blob, list) else blob.get("records", blob)
    clear = [p for p in all_pairs if p.get("label_3class") in
             ("silent_bypass", "self_correction", "error_propagation")]

    set198 = select_198set(clear, args.n_per_type, args.direction_examples)
    counts = Counter(p["subtype"] for p in set198)
    logging.info(f"198-set selected: total={len(set198)} {dict(counts)}")
    if any(counts.get(st, 0) < args.n_per_type for st in ["TYPE_A", "TYPE_B", "TYPE_C"]):
        logging.warning("Some types under-represented in 198-set; check IN_PAIRS")

    sub2b = _load_module("2b_subclassify.py", "subclassify_2b")

    model, tokenizer = load_model()
    set198 = preprocess_198set(set198, tokenizer)

    # Resume: skip (basis, k, α) tuples already recorded
    records = []
    if OUT_SUBCLASSIFIED.exists():
        try:
            ck = json.load(open(OUT_SUBCLASSIFIED))
            if ck.get("status") == "in_progress":
                records = ck["records"]
                logging.info(f"Resuming: {len(records)} records already on disk")
        except Exception:
            pass
    done_keys = {(r["basis"], r["k"], r["alpha"], r["id"], r.get("perturbation_point", ""))
                 for r in records}

    bases_map = {"M2": basis_M2, "probe": basis_probe}
    k_values = [int(k) for k in args.k_values.split(",")] if args.k_values else K_VALUES
    alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else ALPHAS

    t0 = time.time()
    for basis_name in args.bases.split(","):
        basis_name = basis_name.strip()
        if basis_name not in bases_map:
            logging.warning(f"Unknown basis '{basis_name}'; skipping")
            continue
        for k in k_values:
            # Skip if all entries for this (basis,k) are already done
            need = [(p["id"], p.get("perturbation_point", "")) for p in set198]
            already = sum(1 for a in alphas for nid, npt in need
                          if (basis_name, k, float(a), nid, npt) in done_keys)
            total = len(alphas) * len(set198)
            if already >= total:
                logging.info(f"basis={basis_name} k={k}: all {total} already done, skipping")
                continue
            logging.info(f"=== basis={basis_name}  k={k}  ({already}/{total} pre-done) ===")
            rs = run_dose_response(
                model, tokenizer, set198, bases_map[basis_name], k, alphas,
                args.target_layer)
            for r in rs:
                key = (basis_name, k, r["alpha"], r["id"], r.get("perturbation_point", ""))
                if key in done_keys:
                    continue
                r["basis"] = basis_name
                # Rule-based classification of this steered continuation.
                subtype, reason, label_3class = rule_classify(r, sub2b)
                r["subtype"] = subtype
                r["subtype_reason"] = reason
                r["label_3class"] = label_3class
                records.append(r)
                done_keys.add(key)

            # Checkpoint after every (basis, k)
            json.dump({"timestamp": datetime.now().isoformat(),
                       "n": len(records), "status": "in_progress",
                       "records": records}, open(OUT_SUBCLASSIFIED, "w"))
            logging.info(f"checkpoint: {len(records)} records, "
                         f"elapsed {time.time()-t0:.0f}s")

    json.dump({"timestamp": datetime.now().isoformat(), "n": len(records),
               "status": "complete", "records": records},
              open(OUT_SUBCLASSIFIED, "w"), indent=2)
    logging.info(f"Wrote {len(records)} steered records -> {OUT_SUBCLASSIFIED}")
    _print_rule_based_summary(records)


def _print_rule_based_summary(records):
    print("\nRule-based steered-continuation labels by (basis, k):")
    by = {}
    for r in records:
        key = (r["basis"], r["k"])
        by.setdefault(key, Counter())[r.get("subtype", "?")] += 1
    for (basis, k) in sorted(by):
        c = by[(basis, k)]
        n = sum(c.values())
        if not n:
            continue
        print(f"  {basis} k={k}  n={n:5d}  "
              f"A={c['TYPE_A']:4d}  B={c['TYPE_B']:4d}  C={c['TYPE_C']:4d}  "
              f"NEEDS_JUDGE={c['NEEDS_JUDGE']:4d}  UNCLEAR={c['UNCLEAR']:4d}")
    print("\nNOTE: A/B (bypass vs self-correct) is only final AFTER --stage judge.")


# ---- stage: judge (item 4.4 — re-score regex-ambiguous cases) ----

def stage_judge(args):
    j9b = _load_module("9b_run_judge.py", "judge_9b")
    if not OUT_SUBCLASSIFIED.exists():
        raise SystemExit(f"{OUT_SUBCLASSIFIED} not found — run --stage steer first.")
    blob = json.load(open(OUT_SUBCLASSIFIED))
    records = blob["records"] if isinstance(blob, dict) else blob

    needs_ab = [r for r in records if r.get("subtype") == "NEEDS_JUDGE"]
    unclear = [r for r in records if r.get("subtype") == "UNCLEAR"]
    logging.info(f"Judge: {len(needs_ab)} NEEDS_JUDGE (A/B), {len(unclear)} UNCLEAR (extract)")

    import asyncio
    judged = []
    if needs_ab:
        t0 = time.time()
        ab = asyncio.run(j9b.run_judge_async(
            needs_ab, args.api_key, j9b.JUDGE_PROMPT_AB, j9b.parse_ab_response,
            model=args.judge_model, concurrency=args.concurrency, max_tokens=5))
        logging.info(f"A/B pass: {len(ab)} in {time.time()-t0:.0f}s")
        judged += ab

    if unclear:
        def parse_extract(t):
            t = t.strip()
            return None if t.upper() == "NONE" or not t else t
        t0 = time.time()
        ex = asyncio.run(j9b.run_judge_async(
            unclear, args.api_key, j9b.JUDGE_PROMPT_EXTRACT, parse_extract,
            model=args.judge_model, concurrency=args.concurrency, max_tokens=50))
        # For extract-stage we resolve to A vs C by comparing the judged
        # numeric/text answer against reference, matching exp_1_1.
        for r in ex:
            jl = r.get("judge_label")
            if not jl:
                continue
            ref = r.get("reference_answer", "")
            src = r.get("source", "")
            if src == "gsm8k":
                try:
                    eg = float(str(jl).replace(",", ""))
                    rg = float(str(ref).replace(",", ""))
                    correct = abs(eg - rg) < 1e-6
                except (ValueError, TypeError):
                    correct = False
            elif src == "mmlu":
                correct = str(jl).strip().upper() == str(r.get("correct_letter", "")).upper()
            else:
                correct = str(jl).strip().lower() == str(ref).strip().lower()
            r["judge_label"] = "A" if correct else "C"
            r["is_correct"] = correct
        logging.info(f"Extract pass: {len(ex)} in {time.time()-t0:.0f}s")
        judged += ex

    # Match judged decisions back to records via stable key
    def _key(r):
        return (r.get("basis", ""), r.get("k", 0), r.get("alpha", 0.0),
                r.get("id", ""), r.get("perturbation_point", ""))
    jmap = {_key(r): r for r in judged}
    for r in records:
        j = jmap.get(_key(r))
        if not j:
            continue
        jl = j.get("judge_label")
        if r.get("subtype") == "NEEDS_JUDGE":
            if jl == "A":
                r["subtype"], r["label_3class"] = "TYPE_A", "silent_bypass"
            elif jl == "B":
                r["subtype"], r["label_3class"] = "TYPE_B", "self_correction"
            r["judge_label"] = jl
        elif r.get("subtype") == "UNCLEAR":
            if jl == "C":
                r["subtype"], r["label_3class"] = "TYPE_C", "error_propagation"
            elif jl == "A":
                r["subtype"], r["label_3class"] = "TYPE_A", "silent_bypass"
            r["judge_label"] = jl

    json.dump(records, open(OUT_LABELED, "w"), indent=2)
    logging.info(f"Wrote {len(records)} labeled records -> {OUT_LABELED}")
    _print_flip_diagnostic(records)


# ---- summary diagnostic: per-(basis, k) flip rates by baseline type ----

def _print_flip_diagnostic(records):
    """
    Pivot table: for each (basis, k), report per-baseline-type *flip-out* rate
    at each α. A "flip" means the steered continuation's judge label differs
    from the baseline subtype. This is the right metric for item 4.4 — regex
    is_correct misses A↔B flips entirely.
    """
    print("\n" + "=" * 80)
    print("ITEM 4.2 — MULTI-DIRECTION STEERING: FLIP RATES BY (BASIS, K, BASELINE TYPE)")
    print("=" * 80)
    print("Baseline single-direction reference (paper, layer 21):")
    print("  TYPE_A: 0.0% flip across α ∈ [-10, +10]  (UB 0.41% pooled)")
    print("  TYPE_B: 0.0%-3.0% flip  (uninformative; n_per_α=66)")
    print("  TYPE_C: error rate 87.9%-97.0%  (at ceiling)")
    print("-" * 80)

    by = {}
    for r in records:
        key = (r["basis"], r["k"], r["baseline_subtype"], r["alpha"])
        by.setdefault(key, []).append(r)

    bases = sorted({r["basis"] for r in records})
    ks = sorted({r["k"] for r in records})
    alphas = sorted({r["alpha"] for r in records})

    _CLEAR_SUBTYPES = {"TYPE_A", "TYPE_B", "TYPE_C"}
    for basis in bases:
        for k in ks:
            print(f"\n  basis={basis}  k={k}")
            print(f"    {'α':>6}  {'TYPE_A flip%':>13}  {'TYPE_B flip%':>13}  {'TYPE_C flip%':>13}")
            for a in alphas:
                row = [f"{a:+6.1f}"]
                for st in ["TYPE_A", "TYPE_B", "TYPE_C"]:
                    rs = by.get((basis, k, st, a), [])
                    # Exclude NEEDS_JUDGE / UNCLEAR / "" from the flip-rate
                    # denominator. Counting them as flips inflates the rate
                    # whenever the judge stage hasn't (yet) resolved them.
                    clear = [r for r in rs if r.get("subtype", "?") in _CLEAR_SUBTYPES]
                    if not clear:
                        row.append(f"{'—':>13}")
                        continue
                    flips = sum(1 for r in clear if r["subtype"] != st)
                    pct = 100.0 * flips / len(clear)
                    # Show denominator over total so missing-judge cells are visible.
                    row.append(f"{pct:5.1f}% (n={len(clear):3d}/{len(rs):3d})")
                print("    " + "  ".join(row))
    print("=" * 80)
    print("Interpretation:")
    print("  * If TYPE_A flip stays near 0 and TYPE_C flip stays near 0 for both bases")
    print("    at every k ∈ {2,4,8}: null confirmed; rescope §5.4 narrowly.")
    print("  * If TYPE_A flip rises ≥10pp OR TYPE_C flip rises ≥10pp at any (basis,k,α):")
    print("    multi-direction is informative; report and re-frame §5.4 conclusion.")
    print("=" * 80)


# ---- main ----

def main():
    ap = argparse.ArgumentParser(
        description="Multi-direction activation steering (item 4.2 + 4.4)")
    ap.add_argument("--stage", choices=["extract", "steer", "judge", "all"],
                    default="steer", help="pipeline stage")
    ap.add_argument("--target-layer", type=int, default=TARGET_LAYER)
    ap.add_argument("--direction-examples", type=int, default=DIRECTION_EXAMPLES,
                    help="held-out set size for basis training (must match the "
                         "single-direction exp to keep the 198-set non-overlapping)")
    ap.add_argument("--n-per-type", type=int, default=N_PER_TYPE)
    ap.add_argument("--alphas", type=str, default=None,
                    help="comma-separated; defaults to the 11-point dose-response")
    ap.add_argument("--k-values", type=str, default=None,
                    help="comma-separated; defaults to 2,4,8")
    ap.add_argument("--bases", type=str, default="M2,probe",
                    help="comma-separated subset of {M2, probe}")
    ap.add_argument("--api-key", type=str, default=None)
    ap.add_argument("--judge-model", type=str, default="claude-haiku-4-5-20251001")
    ap.add_argument("--concurrency", type=int, default=40)
    args = ap.parse_args()

    if args.stage in ("extract", "all"):
        stage_extract(args)
    if args.stage in ("steer", "all"):
        stage_steer(args)
    if args.stage in ("judge", "all"):
        if not args.api_key:
            raise SystemExit("--api-key required for the judge stage")
        stage_judge(args)


if __name__ == "__main__":
    main()
