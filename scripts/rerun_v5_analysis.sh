#!/bin/bash
# v5: Run LLM judge + re-analyze with expanded dataset.
# Runs LOCALLY — no GPU needed.
#
# Usage: bash scripts/rerun_v5_analysis.sh --api-key <your-anthropic-key>
#
# Prerequisites:
# - data/processed/subclassified_pairs.json (from v3)
# - data/processed/hidden_states/ (from v3, only needed for probe re-training)

set -e

LOGDIR="logs"
mkdir -p "$LOGDIR"
TS=$(date +"%Y%m%d_%H%M%S")

API_KEY="$1"
if [ -z "$API_KEY" ] && [ "$1" = "--api-key" ]; then
    API_KEY="$2"
fi

if [ -z "$API_KEY" ]; then
    echo "Usage: bash scripts/rerun_v5_analysis.sh --api-key <your-anthropic-key>"
    exit 1
fi

echo "=============================================="
echo "v5: LLM Judge + Expanded Analysis"
echo "Started: $(date)"
echo "=============================================="

# Step 9a: Validate judge on known examples
echo ""
echo "[Step 9a] Validating judge prompt..."
python3 scripts/9a_validate_judge.py --api-key "$API_KEY" --n-samples 30 2>&1 | tee "$LOGDIR/${TS}_9a_validate.log"

# Check validation result
echo ""
read -p "Continue with full judging? (y/n) " confirm
if [ "$confirm" != "y" ]; then
    echo "Aborted. Review validation results and adjust prompt if needed."
    exit 0
fi

# Step 9b: Judge all NEEDS_JUDGE examples
echo ""
echo "[Step 9b] Judging 1,188 NEEDS_JUDGE examples..."
python3 scripts/9b_run_judge.py --api-key "$API_KEY" --resume 2>&1 | tee "$LOGDIR/${TS}_9b_judge.log"

# Step 7: Re-run behavioral analysis with expanded data
echo ""
echo "[Step 7] Behavioral analysis (expanded)..."
python3 scripts/7_behavioral_analysis.py 2>&1 | tee "$LOGDIR/${TS}_7_behavioral.log"

# Step 8: Re-run leakage test with expanded data
echo ""
echo "[Step 8] Leakage test (expanded)..."
python3 scripts/8_leakage_test.py --layer 2 2>&1 | tee "$LOGDIR/${TS}_8_leakage.log"

echo ""
echo "=============================================="
echo "v5 analysis complete! $(date)"
echo "=============================================="
echo ""
echo "Note: To re-run probes (Scripts 5b), hidden states from the pod are needed."
echo "If you have them locally, run: python3 scripts/5b_reframed_probe.py"
