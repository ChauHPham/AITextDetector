# 🚀 Training Guide

## Problem
The mutex lock error `[mutex.cc : 452] RAW: Lock blocking...` happens because:
1. HuggingFace Trainer API tries to use multiprocessing
2. macOS doesn't handle multiprocessing from tokenizers well
3. Environment variables alone aren't enough to fix it completely

## Solution

### ✅ BEST: Use the Simple Training Script (Recommended)

The simple training script avoids the Trainer API entirely:

```bash
python scripts/run_train_simple.py
```

**What it does:**
- ✅ No multiprocessing
- ✅ No threading issues  
- ✅ Direct PyTorch training loop
- ✅ Works on macOS
- ✅ Same results as Trainer API

**Output:**
- Trains for 2 epochs
- Shows progress with tqdm
- Saves model to `models/ai_detector`

### Alternative: Shell Script

```bash
bash train_macos.sh
```

This sets all environment variables and runs the simple script.

## If You Still Get Errors

### Option 1: Reduce to Tiny Dataset
```bash
python scripts/sample_dataset.py data/ai_vs_human_text.csv data/tiny.csv -n 100
# Then edit configs/default.yaml:
#   data_path: data/tiny.csv
python scripts/run_train.py
```

### Option 2: Run Outside venv
```bash
# Exit your virtualenv
deactivate

# Install system-wide
pip install --user -r requirements.txt

# Train
python scripts/run_train_simple.py
```

### Option 3: Use Colab/Cloud
If nothing works locally, train on Google Colab (free GPU):
- Upload your data to Google Drive
- Use the Colab notebook template
- Much faster training

## Key Differences

### Simple Script (`run_train_simple.py`)
- ✅ No Trainer API (no multiprocessing issues)
- ✅ Works on macOS
- ✅ Good for small-medium datasets
- ⚠️ Less efficient on large datasets

### Standard Script (`run_train.py`)
- Uses HuggingFace Trainer API
- ✅ Optimized for large datasets
- ⚠️ Multiprocessing issues on macOS

## Recommended Setup

1. **Dataset:** ✅ Downloaded (`data/ai_vs_human_text.csv`)
2. **Config:** ✅ Updated (`configs/default.yaml`)
3. **Training:** Use `run_train_simple.py`

## Start Training

```bash
python scripts/run_train_simple.py
```

Should see output like:
```
🚀 Starting training (simple mode - no multiprocessing)
============================================================

📖 Loading data from data/ai_vs_human_text.csv...
   Loaded 1,000 samples
   Distribution: {0: 493, 1: 507}
   Train: 800 | Val: 200

🤖 Loading model: roberta-base...

📊 Creating datasets...

⚙️  Training for 2 epochs...
```

Good luck! 🎉
