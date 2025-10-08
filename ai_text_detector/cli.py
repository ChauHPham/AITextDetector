import argparse
from sklearn.model_selection import train_test_split
from .config import load_config
from .datasets import DatasetLoader
from .models import DetectorModel
from .train import build_trainer
from .evaluate import evaluate

def train_command(args):
    cfg = load_config(args.config)
    loader = DatasetLoader(model_name=cfg.base_model, max_length=cfg.max_length)
    df = loader.load(args.data)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=cfg.seed, stratify=df["label"])

    model = DetectorModel(model_name=cfg.base_model)
    trainer = build_trainer(model.model, model.tokenizer, train_df, val_df, cfg)
    trainer.train()
    model.save(cfg.save_dir)
    print(f"✅ Training complete. Model saved to: {cfg.save_dir}")

def eval_command(args):
    cfg = load_config(args.config)
    model = DetectorModel.load(args.model_path)
    loader = DatasetLoader(model_name=model.model_name, max_length=cfg.max_length)
    df = loader.load(args.data)
    evaluate(model.model, model.tokenizer, df, max_length=cfg.max_length)

def main():
    parser = argparse.ArgumentParser(
        prog="ai-detector",
        description="Detect whether text is AI- or human-written."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train
    p_train = subparsers.add_parser("train", help="Train a new detector model.")
    p_train.add_argument("--data", required=True, help="Path to dataset CSV/JSON/JSONL.")
    p_train.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    p_train.set_defaults(func=train_command)

    # Evaluate
    p_eval = subparsers.add_parser("eval", help="Evaluate a trained model.")
    p_eval.add_argument("--model-path", required=True, help="Path to saved model dir.")
    p_eval.add_argument("--data", required=True, help="Path to dataset CSV/JSON/JSONL.")
    p_eval.add_argument("--config", default="configs/default.yaml", help="YAML config path.")
    p_eval.set_defaults(func=eval_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
