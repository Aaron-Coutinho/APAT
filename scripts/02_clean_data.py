from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/patents_raw.csv")
OUT_PATH = Path("data/processed/patents_clean.csv")
REPORT_PATH = Path("data/processed/CLEANING_REPORT.md")

def main():
    df = pd.read_csv(RAW_PATH)

    before = len(df)

    df = df.drop_duplicates(subset=["patent_id"])
    df["year"] = df["year"].astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)

    after = len(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    with open(REPORT_PATH, "w") as f:
        f.write(f"Rows before: {before}\n")
        f.write(f"Rows after: {after}\n")
        f.write(f"Duplicates removed: {before - after}\n")

    print("Cleaning complete.")

if __name__ == "__main__":
    main()
