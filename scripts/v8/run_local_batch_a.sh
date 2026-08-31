#!/bin/bash
# Batch A: Local experiments (no GPU needed)
# Run all three in parallel since they're independent
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo "V8 Batch A: Local Experiments"
echo "Started: $(date)"
echo "=========================================="

# Exp 1c: Binary replication (no API, instant)
echo ""
echo "[1/3] Exp 1c: Binary C vs non-C replication..."
python3 scripts/v8/exp1c_binary_replication.py
echo "  Done."

# Exp 3b: Perturbation uptake (no API, instant)
echo ""
echo "[2/3] Exp 3b: Perturbation uptake analysis..."
python3 scripts/v8/exp3b_perturbation_uptake.py
echo "  Done."

# Exp 1a-1b: Judge sensitivity (needs ANTHROPIC_API_KEY)
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "[3/3] Exp 1a-1b: SKIPPED (set ANTHROPIC_API_KEY to run)"
    echo "  Usage: ANTHROPIC_API_KEY=sk-... bash scripts/v8/run_local_batch_a.sh"
else
    echo ""
    echo "[3/3] Exp 1a-1b: Judge sensitivity (5 prompt variants)..."
    echo "  This will make ~60K API calls to Haiku. Estimated cost: ~\$1-3"
    python3 scripts/v8/exp1ab_judge_sensitivity.py --concurrency 40
    echo "  Done."
fi

echo ""
echo "=========================================="
echo "Batch A complete: $(date)"
echo "Results in: results/v8/"
echo "Logs in: logs/"
echo "=========================================="
ls -la results/v8/
