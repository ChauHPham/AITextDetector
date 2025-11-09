# 🚀 Quick Start: Download Dataset

## ✅ Script Works! (Tested Successfully)

The download script works perfectly! Here are all the ways to use it:

---

## Method 1: Use the Script (Easiest) ⭐

```bash
# Download the default dataset
python scripts/download_kagglehub.py

# Or specify a different dataset
python scripts/download_kagglehub.py --dataset shamimhasan8/ai-vs-human-text-dataset
```

**Output:** Dataset saved to `data/ai_vs_human_text.csv`

---

## Method 2: Direct in Your Code (Simple)

Just copy-paste this into your Python script:

```python
import kagglehub
import pandas as pd
from pathlib import Path

# Download dataset (no API token needed!)
path = kagglehub.dataset_download("shamimhasan8/ai-vs-human-text-dataset")
print("Path to dataset files:", path)

# Load the CSV
csv_files = list(Path(path).glob("*.csv"))
df = pd.read_csv(csv_files[0])

# Save to your data directory
df.to_csv("data/dataset.csv", index=False)
```

**See:** `examples/simple_download.py` for a complete example

---

## Method 3: Use the Integrated Function

```python
from ai_text_detector.download_data import download_ai_vs_human_dataset

# Download and get the path
csv_path = download_ai_vs_human_dataset()
print(f"Dataset at: {csv_path}")

# Now use it in your training
from ai_text_detector.config import load_config
cfg = load_config("configs/default.yaml")
cfg.data_path = csv_path
```

**See:** `examples/download_and_train.py` for a complete training example

---

## Method 4: Download Any Dataset

```python
from ai_text_detector.download_data import download_kaggle_dataset

# Download any Kaggle dataset
csv_path = download_kaggle_dataset(
    "shamimhasan8/ai-vs-human-text-dataset",
    output_path="data/my_dataset.csv"
)
```

---

## 📊 What Was Downloaded

- **Dataset:** `shamimhasan8/ai-vs-human-text-dataset`
- **Size:** 1,000 samples
- **Columns:** `id`, `text`, `label`, `prompt`, `model`, `date`
- **Labels:** "AI-generated" or "Human-written"
- **Saved to:** `data/ai_vs_human_text.csv`

---

## 🎯 Next Steps

1. **Dataset is ready!** It's at `data/ai_vs_human_text.csv`
2. **Config updated!** `configs/default.yaml` already points to it
3. **Train your model:**
   ```bash
   python scripts/run_train.py
   ```

---

## 💡 Tips

- **Small dataset (1k samples):** Good for quick testing
- **Want more data?** Look for larger datasets on Kaggle
- **Already downloaded?** The script won't re-download (uses cache)
- **No API token needed!** `kagglehub` handles everything

---

## 🔍 Verify It Works

```bash
# Check the dataset
head -5 data/ai_vs_human_text.csv

# Or in Python
import pandas as pd
df = pd.read_csv("data/ai_vs_human_text.csv")
print(f"Rows: {len(df):,}")
print(df.head())
```
