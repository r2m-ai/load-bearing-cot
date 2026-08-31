#!/bin/bash
# finalize_v1: Full Phase 2 pipeline (GSM8K + MMLU + BBH)
#
# Hardware: H100 SXM 80GB, 26 vCPU, 251GB RAM, 40GB container, 100GB volume
# Backend: vLLM (primary) with HF fallback
#
# Usage: bash scripts/run_finalize_v1.sh
# Estimated: ~1.5 hrs total

set -e

if [ -z "$TMUX" ] && command -v tmux &>/dev/null; then
    echo "Launching in tmux..."
    tmux new-session -d -s cot-lies "cd $(pwd) && bash scripts/run_finalize_v1.sh; echo 'Done!'; read"
    tmux attach -t cot-lies
    exit 0
fi

LOGDIR="logs"
mkdir -p "$LOGDIR" data/raw data/processed/hidden_states figures
TS=$(date +"%Y%m%d_%H%M%S")

RESULTS_DIR="results_finalize_v1_${TS}"
mkdir -p "$RESULTS_DIR"

echo "=============================================="
echo "finalize_v1: Phase 2 Full Pipeline"
echo "Hardware: H100 SXM, 251GB RAM"
echo "Backend: vLLM (primary)"
echo "Datasets: GSM8K (full) + MMLU (5K) + BBH"
echo "Results: $RESULTS_DIR/"
echo "Started: $(date)"
echo "=============================================="

# Helper: save a checkpoint after each step
save_checkpoint() {
    local step_name=$1
    echo ""
    echo "[Checkpoint] Saving $step_name results..."

    # Copy all current processed data to results dir
    cp data/processed/*.json "$RESULTS_DIR/" 2>/dev/null || true
    cp -r figures/* "$RESULTS_DIR/" 2>/dev/null || true

    # Save a manifest
    echo "$step_name completed at $(date)" >> "$RESULTS_DIR/manifest.log"
    echo "  Files:" >> "$RESULTS_DIR/manifest.log"
    ls -lh data/processed/*.json 2>/dev/null >> "$RESULTS_DIR/manifest.log"
    ls -lh figures/*.png 2>/dev/null >> "$RESULTS_DIR/manifest.log"
    echo "" >> "$RESULTS_DIR/manifest.log"

    echo "[Checkpoint] $step_name saved to $RESULTS_DIR/"
}

# ---- Setup ----
echo ""
echo "[Setup] Checking dependencies..."
python3 -c "import vllm; print(f'vLLM {vllm.__version__}')" 2>/dev/null || {
    echo "Installing vLLM..."
    pip install -q vllm 2>&1 | tail -3
}
python3 -c "import transformers, sklearn, matplotlib; print('All deps OK')"

# ---- Step 0: Download all datasets ----
echo ""
echo "[Step 0] Downloading datasets..."
if [ ! -f "data/raw/gsm8k_full.json" ] || [ ! -f "data/raw/mmlu_full.json" ]; then
    python3 scripts/0_download_datasets.py 2>&1 | tee "$LOGDIR/${TS}_0_download.log"
else
    echo "GSM8K + MMLU already downloaded, skipping."
fi

if [ ! -f "data/raw/bbh_full.json" ]; then
    python3 scripts/11_bbh_pipeline.py --step download 2>&1 | tee "$LOGDIR/${TS}_0_bbh_download.log"
else
    echo "BBH already downloaded, skipping."
fi
save_checkpoint "step0_download"

# ---- Step 1a: GSM8K CoT ----
echo ""
echo "[Step 1a] GSM8K CoT generation (full, ~8K)..."
python3 scripts/1_generate_cot.py --gsm8k-only --batch-size 32 2>&1 | tee "$LOGDIR/${TS}_1a_gsm8k_cot.log"
# Save GSM8K results separately
cp data/processed/cot_responses.json "$RESULTS_DIR/gsm8k_cot_responses.json" 2>/dev/null
cp data/processed/faithful_cot.json "$RESULTS_DIR/gsm8k_faithful_cot.json" 2>/dev/null
save_checkpoint "step1a_gsm8k_cot"

# ---- Step 1b: MMLU CoT ----
echo ""
echo "[Step 1b] MMLU CoT generation (5K)..."
python3 scripts/1_generate_cot.py --mmlu-only --max-mmlu 5000 --batch-size 32 2>&1 | tee "$LOGDIR/${TS}_1b_mmlu_cot.log"
cp data/processed/cot_responses.json "$RESULTS_DIR/mmlu_cot_responses.json" 2>/dev/null
cp data/processed/faithful_cot.json "$RESULTS_DIR/after_mmlu_faithful_cot.json" 2>/dev/null
save_checkpoint "step1b_mmlu_cot"

# ---- Step 1c: BBH CoT ----
echo ""
echo "[Step 1c] BBH CoT generation..."
python3 scripts/11_bbh_pipeline.py --step cot --batch-size 32 2>&1 | tee "$LOGDIR/${TS}_1c_bbh_cot.log"
cp data/processed/bbh_cot_responses.json "$RESULTS_DIR/" 2>/dev/null
cp data/processed/bbh_faithful_cot.json "$RESULTS_DIR/" 2>/dev/null
cp data/processed/faithful_cot.json "$RESULTS_DIR/all_faithful_cot.json" 2>/dev/null
save_checkpoint "step1c_bbh_cot"

# ---- Step 2: Perturbation ----
echo ""
echo "[Step 2] Continuation-based perturbation (all datasets)..."
python3 scripts/2_create_perturbations.py --batch-size 32 --perturbation-points "early,middle,late" 2>&1 | tee "$LOGDIR/${TS}_2_perturbations.log"
cp data/processed/all_continuation_pairs.json "$RESULTS_DIR/" 2>/dev/null
cp data/processed/faithful_unfaithful_pairs.json "$RESULTS_DIR/" 2>/dev/null
save_checkpoint "step2_perturbation"

# ---- Step 2b: Sub-classification ----
echo ""
echo "[Step 2b] Sub-classification..."
python3 scripts/2b_subclassify.py 2>&1 | tee "$LOGDIR/${TS}_2b_subclassify.log"
cp data/processed/subclassified_pairs.json "$RESULTS_DIR/" 2>/dev/null
save_checkpoint "step2b_subclassify"

# ---- Step 10: Fused probe ----
echo ""
echo "[Step 10] Fused MLP + linear probe (42 layers)..."
python3 scripts/10_mlp_layer_sweep.py --max-examples 2000 --probe-types "linear,mlp" 2>&1 | tee "$LOGDIR/${TS}_10_probe.log"
cp figures/layer_sweep.png figures/mlp_gain.png figures/table6_probe_comparison.csv "$RESULTS_DIR/" 2>/dev/null
save_checkpoint "step10_probe"

# ---- Step 7: Behavioral analysis ----
echo ""
echo "[Step 7] Behavioral analysis..."
python3 scripts/7_behavioral_analysis.py 2>&1 | tee "$LOGDIR/${TS}_7_behavioral.log"
cp figures/figure5_behavioral.png figures/table4_behavioral.csv "$RESULTS_DIR/" 2>/dev/null
save_checkpoint "step7_behavioral"

# ---- Final archive ----
echo ""
echo "[Archive] Creating final results archive..."
cp -r "$LOGDIR"/*${TS}* "$RESULTS_DIR/" 2>/dev/null
tar czf "/workspace/finalize_v1_results_${TS}.tar.gz" "$RESULTS_DIR/" 2>/dev/null

echo ""
echo "=============================================="
echo "finalize_v1 COMPLETE! $(date)"
echo "=============================================="
echo ""
echo "All results saved to:"
echo "  Pod: $RESULTS_DIR/"
echo "  Archive: /workspace/finalize_v1_results_${TS}.tar.gz"
echo ""
echo "To download:"
echo "  scp -P <port> -i ~/.ssh/id_ed25519 root@<ip>:/workspace/finalize_v1_results_${TS}.tar.gz ."
echo ""
echo "Next (run locally):"
echo "  1. tar xzf finalize_v1_results_${TS}.tar.gz"
echo "  2. python3 scripts/9b_run_judge.py --api-key <key>"
echo "  3. python3 scripts/7_behavioral_analysis.py"
echo ""
echo "Data inventory:"
ls -lh "$RESULTS_DIR/"*.json 2>/dev/null | wc -l
echo " JSON files saved"
ls -lh "$RESULTS_DIR/"*.png 2>/dev/null | wc -l
echo " figures saved"
ls -lh "$RESULTS_DIR/"*.csv 2>/dev/null | wc -l
echo " tables saved"
ls -lh "$RESULTS_DIR/"*.log 2>/dev/null | wc -l
echo " logs saved"
