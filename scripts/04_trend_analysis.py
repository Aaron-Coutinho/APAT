from pathlib import Path
import pandas as pd

TREND_PATH = Path("outputs/tables/trend_table.csv")

def main():
    trend_df = pd.read_csv(TREND_PATH, index_col=0)

    latest_year = sorted(trend_df.columns)[-1]

    print("Top keywords in latest year:")
    print(trend_df[latest_year].sort_values(ascending=False).head(10))

if __name__ == "__main__":
    main()
