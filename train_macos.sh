#!/bin/bash
# macOS Training Script - Disables all multiprocessing

export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "🍎 macOS Training Script"
echo "========================"
echo "Environment variables set:"
echo "  TOKENIZERS_PARALLELISM=false"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1"
echo "  OMP_NUM_THREADS=1"
echo ""
echo "Running simple training script..."
echo ""

cd "$(dirname "$0")"
python scripts/run_train_simple.py
