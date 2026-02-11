from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

CLEAN_PATH = Path("data/processed/patents_clean.csv")
OUT_PATH = Path("outputs/tables/trend_table.csv")

def main():
    df = pd.read_csv(CLEAN_PATH)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=200
    )

    vectorizer.fit(df["abstract"])
    features = vectorizer.get_feature_names_out()

    year_map = {}

    for year in sorted(df["year"].unique()):
        text = " ".join(df[df["year"] == year]["abstract"])
        vec = vectorizer.transform([text]).toarray().ravel()
        year_map[year] = vec

    trend_df = pd.DataFrame(year_map, index=features)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    trend_df.to_csv(OUT_PATH)

    print("Keyword extraction complete.")

if __name__ == "__main__":
    main()
