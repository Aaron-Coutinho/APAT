import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


DATA_PATH = Path("patents.csv")
OUT_BAR = Path("trend_top_keywords_by_year.png")
OUT_HEAT = Path("keyword_heatmap.png")


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"patent_id", "year", "title", "abstract"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    df["year"] = df["year"].astype(str)
    df["abstract"] = df["abstract"].fillna("").astype(str)
    return df


def extract_vectorizer(df: pd.DataFrame, max_features: int = 200):
    # ngrams gives meaningful phrases like "edge computing", "machine learning"
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=1,
    )
    vectorizer.fit(df["abstract"])
    return vectorizer


def build_trend_table(df: pd.DataFrame, vectorizer: TfidfVectorizer) -> pd.DataFrame:
    features = vectorizer.get_feature_names_out()
    year_keyword_map = {}

    for year in sorted(df["year"].unique()):
        text = " ".join(df.loc[df["year"] == year, "abstract"])
        tfidf_vec = vectorizer.transform([text]).toarray().ravel()
        year_keyword_map[year] = {
            feat: float(tfidf_vec[i]) for i, feat in enumerate(features)
        }

    trend_df = pd.DataFrame(year_keyword_map).fillna(0.0)
    return trend_df


def top_keywords_per_year(trend_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    # Keep only the union of top_n keywords from each year for clean plotting
    keep = set()
    for year in trend_df.columns:
        top = trend_df[year].sort_values(ascending=False).head(top_n).index
        keep.update(top)
    return trend_df.loc[sorted(keep)]


def plot_top_keywords_bar(trend_df_small: pd.DataFrame, out_path: Path):
    ax = trend_df_small.T.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Top Technology Keywords by Year (from Patent Abstracts)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Keyword Importance (TF-IDF)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_heatmap(trend_df_small: pd.DataFrame, out_path: Path):
    # Heatmap: rows=keywords, cols=years
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(trend_df_small))))
    im = ax.imshow(trend_df_small.values, aspect="auto")

    ax.set_title("Keyword Trend Heatmap (TF-IDF)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Keywords")

    ax.set_xticks(range(len(trend_df_small.columns)))
    ax.set_xticklabels(trend_df_small.columns)

    ax.set_yticks(range(len(trend_df_small.index)))
    ax.set_yticklabels(trend_df_small.index)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    try:
        df = load_data(DATA_PATH)
        vectorizer = extract_vectorizer(df)
        trend_df = build_trend_table(df, vectorizer)

        # Keep only top terms for readable plots
        trend_df_small = top_keywords_per_year(trend_df, top_n=8)

        print("Sample extracted keywords (top overall by latest year):")
        latest_year = sorted(trend_df.columns)[-1]
        print(trend_df[latest_year].sort_values(ascending=False).head(15))

        plot_top_keywords_bar(trend_df_small, OUT_BAR)
        plot_heatmap(trend_df_small, OUT_HEAT)

        print(f"\nSaved bar chart to {OUT_BAR}")
        print(f"Saved heatmap to {OUT_HEAT}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
