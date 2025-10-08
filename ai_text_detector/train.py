import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
from typing import List
from .utils import set_seed, device_info, auto_fp16

class TextDataset(Dataset):
    def __init__(self, encodings, labels: List[int]):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def build_trainer(model, tokenizer, train_df, val_df, cfg):
    set_seed(cfg.seed)
    print("💻 Device:", device_info())

    train_enc = tokenizer(
        train_df["text"].tolist(),
        truncation=True, padding="max_length",
        max_length=cfg.max_length, return_tensors="pt"
    )
    val_enc = tokenizer(
        val_df["text"].tolist(),
        truncation=True, padding="max_length",
        max_length=cfg.max_length, return_tensors="pt"
    )

    train_ds = TextDataset(train_enc, train_df["label"].tolist())
    val_ds = TextDataset(val_enc, val_df["label"].tolist())

    use_fp16 = auto_fp16(cfg.fp16)

    args = TrainingArguments(
        output_dir=cfg.save_dir,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        logging_steps=cfg.logging_steps,
        evaluation_strategy=cfg.eval_strategy,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        fp16=use_fp16,
        warmup_ratio=cfg.warmup_ratio,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=cfg.dataloader_num_workers,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )
    return trainer
