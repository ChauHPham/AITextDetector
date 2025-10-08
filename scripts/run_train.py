from sklearn.model_selection import train_test_split
from ai_text_detector.config import load_config
from ai_text_detector.datasets import DatasetLoader
from ai_text_detector.models import DetectorModel
from ai_text_detector.train import build_trainer

if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    loader = DatasetLoader(cfg.base_model, max_length=cfg.max_length)
    df = loader.load(cfg.data_path)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=cfg.seed, stratify=df["label"])
    model = DetectorModel(cfg.base_model)
    trainer = build_trainer(model.model, model.tokenizer, train_df, val_df, cfg)
    trainer.train()
    model.save(cfg.save_dir)
    print("✅ Training complete.")
