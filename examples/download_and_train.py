"""
Example: Download dataset and train directly in your code
"""
from ai_text_detector.download_data import download_ai_vs_human_dataset
from sklearn.model_selection import train_test_split
from ai_text_detector.config import load_config
from ai_text_detector.datasets import DatasetLoader
from ai_text_detector.models import DetectorModel
from ai_text_detector.train import build_trainer

# Step 1: Download dataset (if not already downloaded)
print("=" * 60)
print("STEP 1: Downloading dataset...")
print("=" * 60)
csv_path = download_ai_vs_human_dataset()
print(f"\n✅ Dataset ready at: {csv_path}\n")

# Step 2: Load config and update data path
print("=" * 60)
print("STEP 2: Loading configuration...")
print("=" * 60)
cfg = load_config("configs/default.yaml")
cfg.data_path = csv_path  # Use the downloaded dataset
print(f"Using dataset: {cfg.data_path}\n")

# Step 3: Load and prepare data
print("=" * 60)
print("STEP 3: Loading and preparing data...")
print("=" * 60)
loader = DatasetLoader(cfg.base_model, max_length=cfg.max_length)
df = loader.load(cfg.data_path)
print(f"Loaded {len(df):,} samples")
print(f"Class distribution:\n{df['label'].value_counts()}\n")

# Split data
train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=cfg.seed, 
    stratify=df["label"]
)
print(f"Train: {len(train_df):,} samples")
print(f"Validation: {len(val_df):,} samples\n")

# Step 4: Initialize model
print("=" * 60)
print("STEP 4: Initializing model...")
print("=" * 60)
model = DetectorModel(cfg.base_model)
print(f"Model: {cfg.base_model}\n")

# Step 5: Build trainer
print("=" * 60)
print("STEP 5: Building trainer...")
print("=" * 60)
trainer = build_trainer(model.model, model.tokenizer, train_df, val_df, cfg)
print("✅ Trainer ready\n")

# Step 6: Train
print("=" * 60)
print("STEP 6: Training model...")
print("=" * 60)
trainer.train()

# Step 7: Save model
print("=" * 60)
print("STEP 7: Saving model...")
print("=" * 60)
model.save(cfg.save_dir)
print(f"✅ Model saved to: {cfg.save_dir}")
print("\n🎉 Training complete!")
