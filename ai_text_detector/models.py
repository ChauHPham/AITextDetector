import os

# Disable tokenizer parallelism and MPS on macOS
if os.getenv("TOKENIZERS_PARALLELISM") is None:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoModelForSequenceClassification, AutoTokenizer

class DetectorModel:
    def __init__(self, model_name="roberta-base", num_labels=2):
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str):
        model = AutoModelForSequenceClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        obj = cls.__new__(cls)
        obj.model_name = path
        obj.model = model
        obj.tokenizer = tokenizer
        return obj
