# Data folder

Put your datasets here.

If using Kaggle:
1) Install Kaggle API: `pip install kaggle`
2) Save your token at `~/.kaggle/kaggle.json` (chmod 600)
3) Run: `python scripts/kaggle_downloader.py`
4) Point your config (`configs/default.yaml`) `data_path` to the desired CSV/JSONL, or merge to `data/dataset.csv`.
