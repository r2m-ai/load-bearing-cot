#!/bin/bash
# Download all results from a remote GPU instance.
# Usage: bash scripts/download_results.sh <user@ip> [remote_path]
#
# Example: bash scripts/download_results.sh ubuntu@203.0.113.50 ~/cot-lies-detection

set -e

if [ -z "$1" ]; then
    echo "Usage: bash scripts/download_results.sh <user@ip> [remote_path]"
    echo "Example: bash scripts/download_results.sh ubuntu@203.0.113.50 ~/cot-lies-detection"
    exit 1
fi

REMOTE="$1"
REMOTE_PATH="${2:-~/cot-lies-detection}"
LOCAL_DIR="results_$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOCAL_DIR"

echo "Downloading results from $REMOTE:$REMOTE_PATH..."

# Download processed data (JSON files, not hidden states which are too large)
echo "[1/4] Downloading processed data..."
scp "$REMOTE:$REMOTE_PATH/data/processed/cot_responses.json" "$LOCAL_DIR/" 2>/dev/null || echo "  (not found)"
scp "$REMOTE:$REMOTE_PATH/data/processed/faithful_cot.json" "$LOCAL_DIR/" 2>/dev/null || echo "  (not found)"
scp "$REMOTE:$REMOTE_PATH/data/processed/faithful_unfaithful_pairs.json" "$LOCAL_DIR/" 2>/dev/null || echo "  (not found)"
scp "$REMOTE:$REMOTE_PATH/data/processed/hidden_states/metadata.json" "$LOCAL_DIR/" 2>/dev/null || echo "  (not found)"

# Download figures
echo "[2/4] Downloading figures..."
scp -r "$REMOTE:$REMOTE_PATH/figures/" "$LOCAL_DIR/figures/" 2>/dev/null || echo "  (not found)"

# Download logs
echo "[3/4] Downloading logs..."
scp -r "$REMOTE:$REMOTE_PATH/logs/" "$LOCAL_DIR/logs/" 2>/dev/null || echo "  (not found)"

# Download probe model
echo "[4/4] Downloading probe model..."
scp "$REMOTE:$REMOTE_PATH/data/processed/probe_model.pkl" "$LOCAL_DIR/" 2>/dev/null || echo "  (not found)"

echo ""
echo "Done! Results saved to: $LOCAL_DIR/"
ls -la "$LOCAL_DIR/"
