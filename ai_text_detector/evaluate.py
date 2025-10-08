import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

def evaluate(model, tokenizer, df, max_length=256):
    enc = tokenizer(
        df["text"].tolist(),
        truncation=True, padding="max_length",
        max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**enc)
    preds = outputs.logits.argmax(dim=1).cpu().numpy()
    y = df["label"].to_numpy()
    print("Accuracy:", round(accuracy_score(y, preds), 4))
    print("F1 (macro):", round(f1_score(y, preds, average="macro"), 4))
    print("\nReport:\n", classification_report(y, preds, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y, preds))
