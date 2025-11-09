"""
Simple function to download Kaggle datasets directly in your code.
No API token needed - just use kagglehub!
"""
import kagglehub
import pandas as pd
from pathlib import Path
import os

def download_kaggle_dataset(dataset_slug: str, output_path: str = None, data_dir: str = "data"):
    """
    Download a Kaggle dataset and save it to your data directory.
    
    Args:
        dataset_slug: Kaggle dataset slug (e.g., "shamimhasan8/ai-vs-human-text-dataset")
        output_path: Optional output filename (default: uses dataset filename)
        data_dir: Directory to save the dataset (default: "data")
    
    Returns:
        Path to the saved CSV file
    
    Example:
        >>> from ai_text_detector.download_data import download_kaggle_dataset
        >>> csv_path = download_kaggle_dataset("shamimhasan8/ai-vs-human-text-dataset")
        >>> print(f"Dataset saved to: {csv_path}")
    """
    print(f"📥 Downloading dataset: {dataset_slug}")
    
    # Download dataset
    download_path = kagglehub.dataset_download(dataset_slug)
    print(f"✅ Downloaded to: {download_path}")
    
    # Find CSV files
    csv_files = list(Path(download_path).glob("*.csv"))
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {download_path}")
    
    # Use the first CSV (or largest if multiple)
    if len(csv_files) > 1:
        csv_file = max(csv_files, key=lambda p: p.stat().st_size)
        print(f"📊 Multiple CSVs found, using: {csv_file.name}")
    else:
        csv_file = csv_files[0]
    
    # Create output directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Determine output path
    if output_path is None:
        output_path = os.path.join(data_dir, csv_file.name)
    elif not os.path.isabs(output_path):
        output_path = os.path.join(data_dir, output_path)
    
    # Load and save
    print(f"📝 Loading {csv_file.name}...")
    df = pd.read_csv(csv_file)
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")
    
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to: {output_path}")
    
    return output_path

# Convenience function for the specific dataset
def download_ai_vs_human_dataset(output_path: str = "data/ai_vs_human_text.csv"):
    """
    Download the AI vs Human Text dataset.
    
    Args:
        output_path: Where to save the dataset (default: "data/ai_vs_human_text.csv")
    
    Returns:
        Path to the saved CSV file
    """
    return download_kaggle_dataset(
        "shamimhasan8/ai-vs-human-text-dataset",
        output_path=output_path
    )
