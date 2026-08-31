#!/bin/bash
# v3: Re-run with continuation-based perturbation + sub-classification + reframed probe.
# Script 1 results reused.
#
# Usage: bash scripts/rerun_from_step2.sh

set -e

if [ -z "$TMUX" ] && command -v tmux &>/dev/null; then
    echo "Launching inside tmux session 'cot-lies'..."
    tmux new-session -d -s cot-lies "cd $(pwd) && bash scripts/rerun_from_step2.sh; echo 'Done! Press enter to exit.'; read"
    tmux attach -t cot-lies
    exit 0
fi

LOGDIR="logs"
mkdir -p "$LOGDIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

MAX_EXAMPLES=${MAX_EXAMPLES:-1000}
MAX_HIDDEN=${MAX_HIDDEN:-1000}
BATCH_SIZE=${BATCH_SIZE:-32}

echo "=============================================="
echo "CoT Lies Detection — v3 (reframed probe)"
echo "Started: $(date)"
echo "=============================================="

if [ ! -f "data/processed/faithful_cot.json" ]; then
    echo "ERROR: data/processed/faithful_cot.json not found!"
    exit 1
fi
echo "Using existing faithful_cot.json"

# Step 2: Continuation-based perturbation (v3: improved MMLU strategies)
echo ""
echo "[Step 2] Continuation-based perturbation..."
python3 scripts/2_create_perturbations.py \
    --max-examples "$MAX_EXAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --perturbation-points "early,middle,late" \
    2>&1 | tee "$LOGDIR/${TIMESTAMP}_2_perturbations.log"

# Step 2b: Sub-classify into Type A / Type B / Type C
echo ""
echo "[Step 2b] Sub-classifying unfaithful examples..."
python3 scripts/2b_subclassify.py 2>&1 | tee "$LOGDIR/${TIMESTAMP}_2b_subclassify.log"

# Step 3: Extract hidden states
echo ""
echo "[Step 3] Extracting hidden states..."
python3 scripts/3_extract_hidden_states.py \
    --max-examples "$MAX_HIDDEN" \
    --layer-mode key \
    2>&1 | tee "$LOGDIR/${TIMESTAMP}_3_hidden_states.log"

# Step 4: Logit lens analysis
echo ""
echo "[Step 4] Logit lens analysis..."
python3 scripts/4_logit_lens_analysis.py 2>&1 | tee "$LOGDIR/${TIMESTAMP}_4_logit_lens.log"

# Step 5: Original probe (for comparison)
echo ""
echo "[Step 5] Original probe (original vs perturbed)..."
python3 scripts/5_train_probe.py 2>&1 | tee "$LOGDIR/${TIMESTAMP}_5_train_probe.log"

# Step 5b: Reframed probe (within-perturbed, 3-class)
echo ""
echo "[Step 5b] Reframed probe (Type A vs B vs C)..."
python3 scripts/5b_reframed_probe.py 2>&1 | tee "$LOGDIR/${TIMESTAMP}_5b_reframed_probe.log"

# Step 6: Evaluation and ablations
echo ""
echo "[Step 6] Evaluation..."
python3 scripts/6_probe_eval.py 2>&1 | tee "$LOGDIR/${TIMESTAMP}_6_eval.log"

# Step 7: Behavioral analysis
echo ""
echo "[Step 7] Behavioral analysis (position gradient, strategy effectiveness)..."
python3 scripts/7_behavioral_analysis.py 2>&1 | tee "$LOGDIR/${TIMESTAMP}_7_behavioral.log"

echo ""
echo "=============================================="
echo "Pipeline complete! $(date)"
echo "=============================================="
echo "Results: figures/, data/processed/, $LOGDIR/"
