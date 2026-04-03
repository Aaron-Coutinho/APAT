import pandas as pd
import os
import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "patents_raw.csv")
CLEAN_CSV_PATH = os.path.join(os.path.dirname(__file__), "patents_clean.csv")

# Columns from Lens.org we actually need — everything else is dropped
REQUIRED_COLUMNS = [
    "Title",
    "Abstract",
    "Applicants",
    "Publication Year",
    "CPC Classifications"
]

OPTIONAL_COLUMNS = [
    "Lens ID",
    "Publication Date",
    "Owners",
    "Document Type",
    "Cited by Patent Count",
    "Cites Patent Count",
    "Simple Family Size",
    "IPCR Classifications",
    "Jurisdiction"
]


def clean_text(text: str) -> str:
    """Removes extra whitespace, newlines, and non-ASCII characters."""
    if not isinstance(text, str):
        return ""
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip()


def process_raw_csv() -> bool:
    """
    Reads the raw Lens.org CSV, cleans it, and writes patents_clean.csv.
    Returns True if successful, False if raw file not found.
    """
    if not os.path.exists(RAW_CSV_PATH):
        logging.error(
            f"Raw CSV not found at '{RAW_CSV_PATH}'. "
            f"Please place your Lens.org export as 'patents_raw.csv' inside the data/ folder."
        )
        return False

    logging.info(f"Loading raw CSV from {RAW_CSV_PATH}...")
    df = pd.read_csv(RAW_CSV_PATH, low_memory=False)
    rows_before = len(df)
    logging.info(f"Loaded {rows_before} rows.")

    # --- Step 1: Normalize column names (strip whitespace) ---
    df.columns = [c.strip() for c in df.columns]

    # --- Step 2: Keep only useful columns ---
    keep_cols = [c for c in REQUIRED_COLUMNS + OPTIONAL_COLUMNS if c in df.columns]
    df = df[keep_cols]

    # --- Step 3: Drop rows missing critical fields ---
    # Title and Abstract are non-negotiable — app can't work without them
    df = df[df['Title'].notna() & (df['Title'].str.strip() != "")]
    df = df[df['Abstract'].notna() & (df['Abstract'].str.strip() != "")]

    # --- Step 4: Drop duplicates (same Title + same Year) ---
    df = df.drop_duplicates(subset=['Title'], keep='first')

    # --- Step 5: Clean text fields ---
    df['Title'] = df['Title'].apply(clean_text)
    df['Abstract'] = df['Abstract'].apply(clean_text)

    if 'Applicants' in df.columns:
        df['Applicants'] = df['Applicants'].apply(clean_text)
    
    if 'CPC Classifications' in df.columns:
        df['CPC Classifications'] = df['CPC Classifications'].apply(clean_text)

    # --- Step 6: Normalize Publication Year ---
    if 'Publication Year' in df.columns:
        df['Publication Year'] = pd.to_numeric(df['Publication Year'], errors='coerce')
        # Drop rows with invalid or suspiciously old years
        df = df[df['Publication Year'].between(1990, 2026)]
        df['Publication Year'] = df['Publication Year'].astype(int)

    # --- Step 7: Fill remaining NaN values ---
    df.fillna("", inplace=True)

    # --- Step 8: Reset index ---
    df = df.reset_index(drop=True)

    rows_after = len(df)

    # --- Step 9: Write clean CSV ---
    df.to_csv(CLEAN_CSV_PATH, index=False)

    # --- Step 10: Write cleaning report ---
    report_path = os.path.join(os.path.dirname(__file__), "../CLEANING_REPORT.md")
    with open(report_path, "w") as f:
        f.write(f"# Data Cleaning Report\n\n")
        f.write(f"- **Rows before cleaning:** {rows_before}\n")
        f.write(f"- **Rows after cleaning:** {rows_after}\n")
        f.write(f"- **Rows removed:** {rows_before - rows_after}\n")
        f.write(f"- **Columns kept:** {list(df.columns)}\n")
        f.write(f"- **Year range:** {df['Publication Year'].min()} – {df['Publication Year'].max()}\n")

    logging.info(f"Cleaning complete. {rows_before} → {rows_after} rows saved to {CLEAN_CSV_PATH}")
    return True


def ensure_clean_data_exists() -> bool:
    """
    Called on app startup.
    If patents_clean.csv already exists → skip processing.
    If not → process the raw CSV and generate it.
    """
    if os.path.exists(CLEAN_CSV_PATH):
        logging.info("patents_clean.csv already exists. Skipping data processing.")
        return True

    logging.info("patents_clean.csv not found. Starting data processing pipeline...")
    return process_raw_csv()


if __name__ == "__main__":
    # Can also be run standalone: python data/process_data.py
    success = process_raw_csv()
    if success:
        print("Done. patents_clean.csv is ready.")
    else:
        print("Failed. Check that patents_raw.csv is in the data/ folder.")