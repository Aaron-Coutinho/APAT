import pandas as pd
from pathlib import Path

TREND_TABLE_PATH = Path("outputs/tables/trend_table.csv")
OUT_PATH = Path("outputs/tables/white_space_candidates.csv")


def detect_white_space(trend_df: pd.DataFrame,
                       base_year="2024",
                       latest_year="2026",
                       low_threshold=0.05,
                       growth_threshold=0.10,
                       latest_min=0.10):

    candidates = []

    for keyword in trend_df.index:
        base_val = trend_df.loc[keyword, base_year]
        latest_val = trend_df.loc[keyword, latest_year]
        growth = latest_val - base_val

        if (
            base_val < low_threshold and
            growth > growth_threshold and
            latest_val > latest_min
        ):
            candidates.append({
                "keyword": keyword,
                "base_year_value": base_val,
                "latest_year_value": latest_val,
                "growth": growth
            })

    return pd.DataFrame(candidates).sort_values(by="growth", ascending=False)


def main():
    trend_df = pd.read_csv(TREND_TABLE_PATH, index_col=0)
    white_space_df = detect_white_space(trend_df)
    white_space_df.to_csv(OUT_PATH, index=False)
    print(f"White space candidates saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
