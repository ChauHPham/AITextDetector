# 🍎 macOS Threading Fix

## Problem
On macOS, PyTorch/transformers multiprocessing causes mutex lock blocking issues:
```
[mutex.cc : 452] RAW: Lock blocking 0x...
```

## Solution ✅

### 1. Environment Variables Set
The script now sets these BEFORE importing torch/transformers:
- `TOKENIZERS_PARALLELISM=false` - Disables tokenizer multiprocessing
- `PYTORCH_ENABLE_MPS_FALLBACK=1` - Better MPS handling
- Multiprocessing start method set to "spawn" (required on macOS)

### 2. Config Files Updated
All config files now have `dataloader_num_workers: 0`:
- ✅ `configs/default.yaml`
- ✅ `configs/m2_small.yaml`
- ✅ `configs/m2_medium.yaml`
- ✅ `configs/m2_large.yaml`

### 3. Auto-Detection Added
The training code now automatically detects macOS and sets workers to 0:
- If you're on macOS (Darwin) and workers > 0, it auto-fixes it
- Shows a warning message when it does this

### 4. Tokenizer Fixes
Both `models.py` and `datasets.py` now disable tokenizer parallelism on import

## Why This Happens

macOS uses a different multiprocessing model than Linux/Windows:
- `fork()` is not fully supported on macOS
- Multiple worker processes can cause deadlocks
- Setting workers to 0 uses the main process (slower but stable)

## Performance Impact

- **With workers=0**: Slightly slower data loading, but stable
- **With workers>0**: Faster on Linux/Windows, but crashes on macOS

For small-medium datasets (1k-50k), the difference is minimal.

## Test It

```bash
python scripts/run_train.py
```

Should now work without mutex lock errors! 🎉
