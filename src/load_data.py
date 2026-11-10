from pathlib import Path
import pandas as pd

FILES = {"train": "mitbih_train.csv", "test": "mitbih_test.csv"}

def load_raw(data_dir="data"):
    """Return {'train': DataFrame, 'test': DataFrame} exactly as stored."""
    frames = {}
    for split, name in FILES.items():
        path = Path(data_dir) / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Download the dataset first.")
        frames[split] = pd.read_csv(path, header=None)
    return frames

if __name__ == "__main__":
    raw = load_raw()
    for split, df in raw.items():
        print(f"{split}: {df.shape[0]} rows, {df.shape[1]} columns")
    print(raw["train"].iloc[:3, [0, 1, 2, 185, 186, 187]].to_string())