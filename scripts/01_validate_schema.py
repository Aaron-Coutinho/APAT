from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/patents_raw.csv")
REQUIRED = {"patent_id", "year", "title", "abstract"}

def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError("patents_raw.csv not found")

    df = pd.read_csv(RAW_PATH)
    missing = REQUIRED - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    print("Schema validation passed.")

if __name__ == "__main__":
    main()
