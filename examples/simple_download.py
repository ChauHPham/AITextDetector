"""
Simple example: Download dataset directly in your code
Just copy-paste this into your script!
"""
import kagglehub
import pandas as pd
from pathlib import Path

# Download dataset (no API token needed!)
print("📥 Downloading dataset...")
path = kagglehub.dataset_download("shamimhasan8/ai-vs-human-text-dataset")
print(f"✅ Downloaded to: {path}")

# Find and load CSV
csv_files = list(Path(path).glob("*.csv"))
if csv_files:
    df = pd.read_csv(csv_files[0])
    print(f"✅ Loaded {len(df):,} rows")
    print(f"   Columns: {list(df.columns)}")
    
    # Save to your data directory
    output_path = "data/dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"💾 Saved to: {output_path}")
    
    # Now you can use it!
    print(f"\n🎯 Use this path in your config: {output_path}")
else:
    print("⚠️  No CSV files found")
