from ai_text_detector.config import load_config
from ai_text_detector.models import DetectorModel
from ai_text_detector.datasets import DatasetLoader
from ai_text_detector.evaluate import evaluate

if __name__ == "__main__":
    cfg = load_config("configs/default.yaml")
    model = DetectorModel.load(cfg.save_dir)
    loader = DatasetLoader(model.model_name, max_length=cfg.max_length)
    df = loader.load(cfg.data_path)
    evaluate(model.model, model.tokenizer, df, max_length=cfg.max_length)
