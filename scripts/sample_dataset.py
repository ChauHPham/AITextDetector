"""
Helper script to intelligently sample a large dataset for training on M2 Mac.
This creates balanced subsets for quick iteration.
"""
import pandas as pd
import argparse
from pathlib import Path

def sample_dataset(input_path: str, output_path: str, n_samples: int, stratify: bool = True):
    """
    Sample a dataset while maintaining class balance.
    
    Args:
        input_path: Path to input CSV/JSONL
        output_path: Path to save sampled dataset
        n_samples: Number of samples to keep
        stratify: If True, maintain class balance
    """
    print(f"📖 Loading dataset from {input_path}...")
    
    # Load dataset
    if str(input_path).endswith(".csv"):
        df = pd.read_csv(input_path)
    elif str(input_path).endswith(".jsonl") or str(input_path).endswith(".json"):
        df = pd.read_json(input_path, lines=str(input_path).endswith(".jsonl"))
    else:
        raise ValueError(f"Unsupported format: {input_path}")
    
    print(f"📊 Original dataset size: {len(df):,} samples")
    
    # Find label column
    label_col = None
    for col in ["label", "target", "class", "is_ai"]:
        if col in df.columns:
            label_col = col
            break
    
    if label_col:
        print(f"📈 Class distribution:")
        print(df[label_col].value_counts())
    
    # Sample
    if stratify and label_col:
        # Stratified sampling to maintain balance
        sampled = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), n_samples // 2), random_state=42)
        )
        # If we need more samples, take randomly
        if len(sampled) < n_samples:
            remaining = df[~df.index.isin(sampled.index)]
            needed = n_samples - len(sampled)
            if len(remaining) > 0:
                additional = remaining.sample(min(len(remaining), needed), random_state=42)
                sampled = pd.concat([sampled, additional])
    else:
        sampled = df.sample(min(len(df), n_samples), random_state=42)
    
    print(f"✅ Sampled dataset size: {len(sampled):,} samples")
    if label_col:
        print(f"📈 Sampled class distribution:")
        print(sampled[label_col].value_counts())
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if str(output_path).endswith(".csv"):
        sampled.to_csv(output_path, index=False)
    elif str(output_path).endswith(".jsonl"):
        sampled.to_json(output_path, orient="records", lines=True)
    else:
        sampled.to_csv(output_path, index=False)
    
    print(f"💾 Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample a dataset for training")
    parser.add_argument("input", help="Input dataset path")
    parser.add_argument("output", help="Output dataset path")
    parser.add_argument("-n", "--n-samples", type=int, default=10000,
                       help="Number of samples (default: 10000)")
    parser.add_argument("--no-stratify", action="store_true",
                       help="Don't maintain class balance")
    
    args = parser.parse_args()
    
    sample_dataset(
        args.input,
        args.output,
        args.n_samples,
        stratify=not args.no_stratify
    )
