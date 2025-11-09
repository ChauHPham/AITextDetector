"""
Simple training script without HuggingFace Trainer API.
This avoids multiprocessing issues on macOS.
"""
import sys
import os
from pathlib import Path

# Fix macOS multiprocessing issues - MUST be before any torch/transformers imports
if sys.platform == "darwin":  # macOS
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    # Set multiprocessing start method to spawn (required on macOS)
    try:
        import multiprocessing
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Disable all parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Force CPU and disable MPS on macOS (this is the key fix!)
if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    torch.backends.mps.enabled = False
    os.environ["DEVICE"] = "cpu"

torch.set_num_threads(1)

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_length)).squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }

def train_simple():
    """Train model without HuggingFace Trainer API to avoid multiprocessing issues"""
    
    import sys
    print("🚀 Starting training (simple mode - no multiprocessing)", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()
    
    # Config
    MODEL_NAME = "roberta-base"
    DATA_PATH = "data/ai_vs_human_text.csv"
    SAVE_DIR = "models/ai_detector"
    BATCH_SIZE = 8
    EPOCHS = 2
    LR = 5e-5
    MAX_LENGTH = 256
    
    # Create output directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Load data
    print(f"\n📖 Loading data from {DATA_PATH}...", flush=True)
    sys.stdout.flush()
    df = pd.read_csv(DATA_PATH)
    
    # Normalize labels
    def normalize_label(label):
        if isinstance(label, str):
            return 1 if label.lower() in ["ai", "ai-generated"] else 0
        return int(label) if label in [0, 1] else 0
    
    df["label"] = df["label"].apply(normalize_label)
    print(f"   Loaded {len(df):,} samples")
    print(f"   Distribution: {df['label'].value_counts().to_dict()}")
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["text"].tolist(),
        df["label"].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    
    print(f"   Train: {len(train_texts):,} | Val: {len(val_texts):,}")
    
    # Load model and tokenizer
    print(f"\n🤖 Loading model: {MODEL_NAME}...")
    
    # Force CPU device on macOS
    if sys.platform == "darwin":
        device = torch.device("cpu")
        print("   Using CPU device (macOS detected)")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load with explicit device mapping
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        device_map=None  # Don't use device map, we'll handle device placement
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = model.to(device)
    print(f"   Model loaded on: {device}")
    
    # Create datasets and dataloaders (num_workers=0 to avoid multiprocessing)
    print(f"\n📊 Creating datasets...")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=LR)
    
    # Training loop
    print(f"\n⚙️  Training for {EPOCHS} epochs...")
    print("=" * 60)
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                val_loss += loss.item()
                val_correct += (outputs.logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)
                
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2%}")
        print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2%}")
        print()
    
    # Save model
    print(f"\n💾 Saving model to {SAVE_DIR}...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"✅ Model saved!")
    
    print("\n" + "=" * 60)
    print("🎉 Training complete!")
    print(f"Model saved at: {SAVE_DIR}")

if __name__ == "__main__":
    train_simple()
