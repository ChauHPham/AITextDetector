"""
Downloads and prepares the two Kaggle datasets you specified into `data/`:

1) LLM Detect AI Generated Text Dataset
   https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset

2) AI vs Human Text
   https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text

Prereqs:
- Install Kaggle API: `pip install kaggle`
- Place your Kaggle API token at ~/.kaggle/kaggle.json (or set KAGGLE_USERNAME/KAGGLE_KEY env vars)
"""

import os
import zipfile
import glob
import pandas as pd
import subprocess

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

def kaggle_download(dataset, outdir):
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", outdir, "--force"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def unzip_all(outdir):
    for z in glob.glob(os.path.join(outdir, "*.zip")):
        print("Unzipping:", z)
        with zipfile.ZipFile(z, "r") as f:
            f.extractall(outdir)

def main():
    # 1) Sunil Thite dataset
    kaggle_download("sunilthite/llm-detect-ai-generated-text-dataset", DATA_DIR)
    # 2) Shane Gerami dataset
    kaggle_download("shanegerami/ai-vs-human-text", DATA_DIR)

    unzip_all(DATA_DIR)

    print("\n✅ Downloaded and unzipped. Please inspect files in `data/` and pick the right CSVs.")
    print("If needed, you can concatenate them yourself or point --data to a specific one.")
    print("Example to merge (edit column names as necessary):")
    print("   python - <<'PY'\n"
          "import pandas as pd\n"
          "import glob\n"
          "dfs=[]\n"
          "for p in glob.glob('data/*.csv'):\n"
          "    try:\n"
          "        df=pd.read_csv(p)\n"
          "        dfs.append(df)\n"
          "    except Exception as e:\n"
          "        print('Skip', p, e)\n"
          "pd.concat(dfs, ignore_index=True).to_csv('data/dataset.csv', index=False)\n"
          "print('Wrote data/dataset.csv')\n"
          "PY")

if __name__ == "__main__":
    main()
