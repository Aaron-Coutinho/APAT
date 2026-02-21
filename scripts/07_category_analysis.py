from pathlib import Path
import pandas as pd

TREND_PATH = Path("outputs/tables/trend_table.csv")
OUT_DIR = Path("outputs/tables")
OUT_CATEGORY_TREND = OUT_DIR / "category_trend.csv"
OUT_CATEGORY_SUMMARY = OUT_DIR / "category_summary.csv"

# Simple rule-based mapping (fast + explainable)
CATEGORY_RULES = {
    "AI/ML": ["ai", "learning", "model", "neural", "tf", "nlp"],
    "Cloud/Distributed": ["cloud", "microservices", "distributed", "orchestration", "workload", "scaling"],
    "Edge/IoT": ["edge", "iot", "sensor", "latency", "routing", "gateway"],
    "Security": ["encryption", "quantum", "authentication", "intrusion", "secure", "cyber"],
    "Sustainability": ["carbon", "energy", "cooling", "water", "efficient", "footprint", "green"],
}

def keyword_to_category(keyword: str) -> str:
    kw = keyword.lower()
    for cat, tokens in CATEGORY_RULES.items():
        if any(t in kw for t in tokens):
            return cat
    return "Other"

def main():
    trend_df = pd.read_csv(TREND_PATH, index_col=0)

    # Add category column for each keyword row
    tmp = trend_df.copy()
    tmp["category"] = [keyword_to_category(k) for k in tmp.index]

    # Category trend: sum of keyword tf-idf per year within each category
    years = [c for c in trend_df.columns]
    category_trend = tmp.groupby("category")[years].sum().sort_values(by=years[-1], ascending=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    category_trend.to_csv(OUT_CATEGORY_TREND)

    # Category summary: growth + competition index
    base_year = years[0]
    latest_year = years[-1]

    summary = pd.DataFrame({
        "category": category_trend.index,
        "base_year": base_year,
        "latest_year": latest_year,
        "base_value": category_trend[base_year].values,
        "latest_value": category_trend[latest_year].values,
    })

    summary["growth"] = summary["latest_value"] - summary["base_value"]
    # Competition proxy: average presence across years
    summary["competition_index"] = category_trend[years].mean(axis=1).values

    summary = summary.sort_values(by="growth", ascending=False)
    summary.to_csv(OUT_CATEGORY_SUMMARY, index=False)

    print("Saved:")
    print(f"- {OUT_CATEGORY_TREND}")
    print(f"- {OUT_CATEGORY_SUMMARY}")

if __name__ == "__main__":
    main()