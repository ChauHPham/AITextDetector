from typing import Tuple, List
import pandas as pd
from transformers import AutoTokenizer

SUPPORTED_TEXT_COLUMNS = ["text", "content", "body", "essay", "prompt"]

# Try common label column names; map to 0 (human), 1 (ai)
LABEL_MAPPINGS = {
    "label": None,           # already 0/1 or string
    "target": None,
    "class": None,
    "is_ai": None
}

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Find text column
    text_col = None
    for c in SUPPORTED_TEXT_COLUMNS:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"Could not find a text column among: {SUPPORTED_TEXT_COLUMNS}")

    df = df.rename(columns={text_col: "text"})

    # Find label column
    label_col = None
    for c in LABEL_MAPPINGS.keys():
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        # attempt heuristic: columns named like 'human'/'ai'
        for c in df.columns:
            if str(c).lower() in ("ai", "human", "source"):
                label_col = c
                break
    if label_col is None:
        raise ValueError("Could not find a label column. Expected one of: "
                         f"{list(LABEL_MAPPINGS.keys())} or something like ['ai','human','source'].")

    # Normalize labels (0=human, 1=ai)
    def to01(v):
        if isinstance(v, str):
            v_low = v.strip().lower()
            if v_low in ("ai", "machine", "generated", "gpt", "llm", "chatgpt"):
                return 1
            if v_low in ("human", "person", "authored", "real"):
                return 0
        try:
            iv = int(v)
            if iv in (0, 1):
                return iv
        except Exception:
            pass
        # fallback: treat non-human as AI
        return 1

    df["label"] = df[label_col].apply(to01)
    df = df[["text", "label"]].dropna()
    df = df[df["text"].astype(str).str.strip() != ""]
    return df

class DatasetLoader:
    def __init__(self, model_name="roberta-base", max_length: int = 256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def load(self, path) -> pd.DataFrame:
        if str(path).endswith(".csv"):
            df = pd.read_csv(path)
        elif str(path).endswith(".jsonl") or str(path).endswith(".json"):
            df = pd.read_json(path, lines=str(path).endswith(".jsonl"))
        else:
            raise ValueError(f"Unsupported file format: {path}")
        return _normalize_columns(df)

    def tokenize(self, texts: List[str]):
        return self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
