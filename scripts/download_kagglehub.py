"""
Download Kaggle datasets directly using kagglehub (no API token needed!)

Usage:
    python scripts/download_kagglehub.py
    
    # Or download specific dataset:
    python scripts/download_kagglehub.py --dataset shamimhasan8/ai-vs-human-text-dataset
"""
import os
import kagglehub
import pandas as pd
import glob
from pathlib import Path
import argparse

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_dataset(dataset_slug: str, output_name: str = None):
    """
    Download a Kaggle dataset using kagglehub.
    
    Args:
        dataset_slug: Kaggle dataset slug (e.g., "shamimhasan8/ai-vs-human-text-dataset")
        output_name: Optional name for the output CSV file
    """
    print(f"📥 Downloading dataset: {dataset_slug}")
    print("   (No API token needed with kagglehub!)")
    
    # Download dataset - returns path to downloaded files
    path = kagglehub.dataset_download(dataset_slug)
    print(f"✅ Downloaded to: {path}")
    
    # Find all CSV files in the downloaded directory
    csv_files = list(Path(path).glob("*.csv"))
    
    if not csv_files:
        print(f"⚠️  No CSV files found in {path}")
        print(f"   Files found: {list(Path(path).iterdir())}")
        return None
    
    print(f"\n📊 Found {len(csv_files)} CSV file(s):")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")
    
    # If multiple CSVs, try to find the main one or merge them
    if len(csv_files) == 1:
        main_csv = csv_files[0]
    else:
        # Look for common names
        main_csv = None
        for csv_file in csv_files:
            name_lower = csv_file.name.lower()
            if any(keyword in name_lower for keyword in ['train', 'main', 'dataset', 'data']):
                main_csv = csv_file
                break
        
        if not main_csv:
            # Use the largest CSV
            main_csv = max(csv_files, key=lambda p: p.stat().st_size)
            print(f"   Using largest file: {main_csv.name}")
    
    # Copy to data directory
    output_path = os.path.join(DATA_DIR, output_name or main_csv.name)
    
    # Read and save (this also normalizes the file)
    print(f"\n📝 Processing and saving to: {output_path}")
    df = pd.read_csv(main_csv)
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to: {output_path}")
    
    # If there are other CSVs, mention them
    other_csvs = [f for f in csv_files if f != main_csv]
    if other_csvs:
        print(f"\n💡 Other CSV files available in {path}:")
        for csv_file in other_csvs:
            print(f"   - {csv_file.name}")
        print(f"   You can manually copy them to {DATA_DIR} if needed")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Download Kaggle datasets using kagglehub")
    parser.add_argument(
        "--dataset",
        default="shamimhasan8/ai-vs-human-text-dataset",
        help="Kaggle dataset slug (default: shamimhasan8/ai-vs-human-text-dataset)"
    )
    parser.add_argument(
        "--output",
        help="Output filename (default: uses dataset filename)"
    )
    
    args = parser.parse_args()
    
    output_path = download_dataset(args.dataset, args.output)
    
    if output_path:
        print(f"\n🎯 Next steps:")
        print(f"   1. Update configs/default.yaml: data_path: {output_path}")
        print(f"   2. Or use: python scripts/run_train.py --data {output_path}")
        print(f"\n💡 Tip: Use scripts/sample_dataset.py to create smaller subsets for testing")

if __name__ == "__main__":
    main()
