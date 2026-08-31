#!/bin/bash
# Setup script for H100 GPU instance (RunPod/Lambda).
# Usage: bash scripts/setup_instance.sh
#
# Assumes Ubuntu with CUDA 12.x already installed (standard on RunPod).

set -e

echo "=============================================="
echo "Setting up CoT Lies Detection (H100 optimized)"
echo "=============================================="

# System info
echo "[1/6] System info..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" 2>/dev/null || true

# Update system
echo "[2/6] Updating system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq ninja-build  # Required for flash-attn build

# Install Python packages
echo "[3/6] Installing Python dependencies..."
pip install --upgrade pip
pip install \
    transformers>=4.40.0 \
    torch>=2.0.0 \
    datasets>=2.19.0 \
    scikit-learn>=1.4.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    tqdm>=4.66.0 \
    accelerate>=0.30.0

# Install Flash Attention 2 (critical for H100 performance)
echo "[4/6] Installing Flash Attention 2 (H100 optimized)..."
pip install flash-attn --no-build-isolation 2>&1 | tail -5 || {
    echo "  WARNING: flash-attn install failed. Trying with ninja..."
    pip install ninja && pip install flash-attn --no-build-isolation 2>&1 | tail -5 || {
        echo "  Flash Attention 2 unavailable. Falling back to PyTorch SDPA."
        echo "  SDPA is still fast on H100 but ~20% slower than FA2."
    }
}

# Verify GPU + optimizations
echo "[5/6] Verifying GPU setup..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'VRAM: {props.total_memory / 1024**3:.1f} GB')
    print(f'Compute capability: {props.major}.{props.minor}')
    if props.major >= 9:
        print('H100 detected (Hopper arch) - all optimizations available')
    elif props.major >= 8:
        print('A100/A10 detected (Ampere arch) - most optimizations available')

# Check Flash Attention
try:
    import flash_attn
    print(f'Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('Flash Attention: NOT INSTALLED (will use SDPA)')

# Check torch.compile
print(f'torch.compile available: {hasattr(torch, \"compile\")}')
print(f'TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}')
"

# Pre-download model
echo "[6/6] Pre-downloading Gemma-2-9B-IT..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('google/gemma-2-9b-it')
print('Downloading model (~18GB)...')
AutoModelForCausalLM.from_pretrained('google/gemma-2-9b-it', torch_dtype=torch.bfloat16)
print('Model cached successfully!')
"

echo ""
echo "=============================================="
echo "Setup complete! H100 optimizations ready."
echo "=============================================="
echo ""
echo "Optimizations enabled:"
echo "  - Flash Attention 2 (Hopper-optimized)"
echo "  - torch.compile with reduce-overhead (CUDA graphs)"
echo "  - TF32 matmul (H100 tensor cores)"
echo "  - bfloat16 inference (native H100 support)"
echo "  - Large batch sizes (32-48)"
echo ""
echo "Run pipeline:"
echo "  bash scripts/run_all.sh"
echo ""
echo "Or individual steps:"
echo "  python3 scripts/1_generate_cot.py --batch-size 32"
echo "  python3 scripts/2_create_perturbations.py --batch-size 48"
echo "  python3 scripts/3_extract_hidden_states.py --layer-mode all"
