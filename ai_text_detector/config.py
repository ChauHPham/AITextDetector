import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml

@dataclass
class Config:
    data_path: str = "data/dataset.csv"
    base_model: str = "roberta-base"
    save_dir: str = "models/ai_detector"
    max_length: int = 256
    batch_size: int = 8
    num_epochs: int = 2
    lr: float = 5e-5
    weight_decay: float = 0.01
    logging_steps: int = 25
    eval_strategy: str = "epoch"
    seed: int = 42
    gradient_accumulation_steps: int = 1
    fp16: Optional[bool] = None  # if None, auto based on cuda
    load_in_8bit: bool = False   # optional if you later add bitsandbytes
    warmup_ratio: float = 0.0
    save_total_limit: int = 2
    save_steps: int = 0          # 0 -> follow eval/save strategy
    dataloader_num_workers: int = 2

def load_config(path: Optional[str]) -> Config:
    if path is None:
        return Config()
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}
    cfg = Config(**{**Config().__dict__, **raw})
    return cfg
