#!/bin/bash
# Batch B: GPU experiments (run on RunPod H100)
# Loads model once, runs Exp 2 + 3a + 3c sequentially
#
# RunPod setup: H100 80GB, 20GB container, 100GB volume at /workspace
# Model is cached to /workspace/model_cache (~18GB, persists across pods)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

echo "=========================================="
echo "V8 Batch B: GPU Experiments"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "VRAM free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=========================================="

# Install deps (fast, most already present on RunPod pytorch image)
pip install -q transformers accelerate scipy 2>/dev/null

# Ensure HF cache goes to volume (not container)
export HF_HOME=/workspace/model_cache
export TRANSFORMERS_CACHE=/workspace/model_cache
mkdir -p /workspace/model_cache 2>/dev/null || true

# Pre-download model if not cached (saves time on reruns)
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
cache = '/workspace/model_cache' if os.path.exists('/workspace') else None
kwargs = {'cache_dir': cache} if cache else {}
print('Checking model cache...')
AutoTokenizer.from_pretrained('google/gemma-2-9b-it', **kwargs)
print('Model cached.')
" 2>&1 | tail -3

# Run all GPU experiments with single model load
echo ""
echo "Running experiments 2 + 3a + 3c..."
python3 scripts/v8/run_gpu_batch_b.py --exp 2,3a,3c --max-examples 0

echo ""
echo "=========================================="
echo "Batch B complete: $(date)"
echo "Results:"
ls -lh results/v8/exp*.json
echo "=========================================="
