#!/bin/bash
# Run the full experimental pipeline on a GPU instance.
# Usage: bash scripts/run_all.sh [--max-gsm8k N] [--max-mmlu N]
#
# Runs inside tmux so it survives SSH disconnects.
# To reattach: tmux attach -t cot-lies
#
# Logs are saved to logs/ directory with timestamps.

set -e

# Auto-launch inside tmux if not already in a tmux session
if [ -z "$TMUX" ] && command -v tmux &>/dev/null; then
    echo "Launching inside tmux session 'cot-lies'..."
    echo "If disconnected, reattach with: tmux attach -t cot-lies"
    tmux new-session -d -s cot-lies "cd $(pwd) && bash scripts/run_all.sh $*; echo 'Pipeline complete! Press enter to exit.'; read"
    tmux attach -t cot-lies
    exit 0
fi

LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default batch sizes for H100 80GB
BATCH_SIZE_GEN=${BATCH_SIZE_GEN:-32}
BATCH_SIZE_VERIFY=${BATCH_SIZE_VERIFY:-48}

echo "=============================================="
echo "CoT Lies Detection — Full Pipeline (H100)"
echo "Started: $(date)"
echo "Batch sizes: gen=$BATCH_SIZE_GEN, verify=$BATCH_SIZE_VERIFY"
echo "Logs: $LOGDIR/"
echo "=============================================="

# Step 0: Download datasets (skip if already downloaded)
if [ ! -f "data/raw/gsm8k_full.json" ] || [ ! -f "data/raw/mmlu_full.json" ]; then
    echo ""
    echo "[Step 0] Downloading datasets..."
    python3 scripts/0_download_datasets.py 2>&1 | tee "$LOGDIR/${TIMESTAMP}_0_download.log"
else
    echo ""
    echo "[Step 0] Datasets already downloaded, skipping."
fi

# Sample sizes for this run
MAX_MMLU=${MAX_MMLU:-1000}
MAX_PERTURBATIONS=${MAX_PERTURBATIONS:-1000}
MAX_HIDDEN=${MAX_HIDDEN:-1000}

# Step 1: Generate CoT (all GSM8K, limited MMLU)
echo ""
echo "[Step 1] Generating CoT (GSM8K=all, MMLU=$MAX_MMLU, batch=$BATCH_SIZE_GEN)..."
python3 scripts/1_generate_cot.py --batch-size "$BATCH_SIZE_GEN" --max-mmlu "$MAX_MMLU" "$@" 2>&1 | tee "$LOGDIR/${TIMESTAMP}_1_generate_cot.log"

# Step 2: Create perturbations (limited)
echo ""
echo "[Step 2] Creating perturbations (max=$MAX_PERTURBATIONS, batch=$BATCH_SIZE_VERIFY)..."
python3 scripts/2_create_perturbations.py --batch-size "$BATCH_SIZE_VERIFY" --max-examples "$MAX_PERTURBATIONS" 2>&1 | tee "$LOGDIR/${TIMESTAMP}_2_perturbations.log"

# Step 3: Extract hidden states (limited)
echo ""
echo "[Step 3] Extracting hidden states (max=$MAX_HIDDEN, layer-mode=key)..."
python3 scripts/3_extract_hidden_states.py --max-examples "$MAX_HIDDEN" --layer-mode key 2>&1 | tee "$LOGDIR/${TIMESTAMP}_3_hidden_states.log"

# Step 4: Logit lens analysis
echo ""
echo "[Step 4] Running logit lens analysis..."
python3 scripts/4_logit_lens_analysis.py 2>&1 | tee "$LOGDIR/${TIMESTAMP}_4_logit_lens.log"

# Step 5: Train linear probe
echo ""
echo "[Step 5] Training linear probe..."
python3 scripts/5_train_probe.py 2>&1 | tee "$LOGDIR/${TIMESTAMP}_5_train_probe.log"

# Step 6: Evaluation and ablations
echo ""
echo "[Step 6] Running evaluation and ablations..."
python3 scripts/6_probe_eval.py 2>&1 | tee "$LOGDIR/${TIMESTAMP}_6_eval.log"

echo ""
echo "=============================================="
echo "Pipeline complete! $(date)"
echo "=============================================="
echo ""
echo "Results:"
echo "  Figures: figures/"
echo "  Data:    data/processed/"
echo "  Logs:    $LOGDIR/"
echo ""
echo "To download results: scp -r ubuntu@<ip>:~/cot-lies-detection/{figures,data/processed,logs} ."
