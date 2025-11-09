# 📊 Dataset Size Guide for M2 Mac

## 🎯 Quick Recommendation

**Use 10k-50k samples** for the best balance of performance and training time.

## 📈 Comparison Table

| Dataset Size | Training Time | Memory Usage | Best For | Recommendation |
|-------------|---------------|--------------|----------|----------------|
| **1k** | ~5-10 min | Low | Quick testing | ⚠️ Too small - high overfitting risk |
| **10k** | ~20-40 min | Medium | **Recommended start** | ✅ Good balance |
| **50k** | ~1-2 hours | Medium-High | **Best balance** | ✅ **RECOMMENDED** |
| **500k** | ~6-12 hours | High | Maximum performance | ⚠️ Only if you have time |

## 🚀 Recommended Workflow

### Step 1: Start Small (1k-5k)
Test your pipeline quickly:
```bash
python scripts/sample_dataset.py data/your_500k_dataset.csv data/dataset_5k.csv -n 5000
python scripts/run_train.py --config configs/m2_small.yaml --data data/dataset_5k.csv
```
**Time:** ~10 minutes  
**Purpose:** Validate your setup works

### Step 2: Scale Up (10k-50k) ⭐ RECOMMENDED
Train your production model:
```bash
python scripts/sample_dataset.py data/your_500k_dataset.csv data/dataset_50k.csv -n 50000
python scripts/run_train.py --config configs/m2_medium.yaml --data data/dataset_50k.csv
```
**Time:** ~1-2 hours  
**Purpose:** Best performance/time trade-off

### Step 3: Full Dataset (Optional)
Only if you need maximum performance:
```bash
python scripts/run_train.py --config configs/m2_large.yaml --data data/your_500k_dataset.csv
```
**Time:** ~6-12 hours  
**Purpose:** Maximum accuracy (marginal gains)

## 💡 Why 10k-50k is Best

1. **Sufficient Diversity**: Enough examples to learn patterns without overfitting
2. **Manageable Time**: 1-2 hours vs 6-12 hours for 500k
3. **Good Performance**: For AI text detection, 50k is usually enough
4. **Quick Iterations**: You can experiment with hyperparameters faster

## 🔧 M2 Mac Optimizations

Your configs are optimized for:
- **CPU training** (M2 doesn't have CUDA)
- **Unified memory** (8-24GB typical)
- **Batch size tuning** (smaller batches for larger datasets)
- **Gradient accumulation** (simulates larger batches)

## 📝 Example Commands

```bash
# Sample 10k balanced samples
python scripts/sample_dataset.py data/large_dataset.csv data/dataset_10k.csv -n 10000

# Train with medium config
python scripts/run_train.py --config configs/m2_medium.yaml --data data/dataset_10k.csv

# Or use the full dataset
python scripts/run_train.py --config configs/m2_large.yaml --data data/large_dataset.csv
```

## ⚡ Performance Tips

1. **Start with 10k** - Validate everything works
2. **Scale to 50k** - Get good performance
3. **Only use 500k** if:
   - You have 6+ hours to spare
   - You need every last % of accuracy
   - You're doing research/comparison

## 🎓 For AI Text Detection Specifically

AI text detection typically needs:
- ✅ **Diverse AI models** (GPT-3, GPT-4, Claude, etc.)
- ✅ **Diverse human writing** (essays, stories, technical, casual)
- ✅ **Balanced classes** (50/50 or close)

**10k-50k samples** with good diversity will outperform **500k samples** with poor diversity.

## 🚨 When to Use Each Size

- **1k**: ❌ Don't use for production - too small
- **10k**: ✅ Good for initial training and testing
- **50k**: ✅ **BEST CHOICE** - production ready
- **500k**: ⚠️ Only if you have time and need maximum accuracy
