#!/bin/bash
# v6: Fused MLP probe + 42-layer sweep.
# Single script does everything: forward pass + feature extraction + probe training.
# No intermediate .pt files saved — all in memory.
#
# Usage: bash scripts/run_v6_gpu.sh
# Estimated: ~30 min on H100 (was ~3.5 hrs with separate extraction)

set -e

if [ -z "$TMUX" ] && command -v tmux &>/dev/null; then
    tmux new-session -d -s cot-lies "cd $(pwd) && bash scripts/run_v6_gpu.sh; echo 'Done!'; read"
    tmux attach -t cot-lies
    exit 0
fi

LOGDIR="logs"
mkdir -p "$LOGDIR"
TS=$(date +"%Y%m%d_%H%M%S")

echo "=============================================="
echo "v6: Fused MLP Probe + 42-Layer Sweep"
echo "Started: $(date)"
echo "=============================================="

if [ ! -f "data/processed/subclassified_pairs.json" ]; then
    echo "ERROR: subclassified_pairs.json not found!"
    exit 1
fi

python3 scripts/10_mlp_layer_sweep.py \
    --max-examples 1000 \
    --probe-types "linear,mlp,mlp_deep" \
    2>&1 | tee "$LOGDIR/${TS}_10_mlp_sweep.log"

echo ""
echo "=============================================="
echo "v6 complete! $(date)"
echo "=============================================="
