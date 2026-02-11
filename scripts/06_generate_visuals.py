from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

TREND_PATH = Path("outputs/tables/trend_table.csv")
BAR_OUT = Path("outputs/figures/trend_top_keywords_by_year.png")
HEAT_OUT = Path("outputs/figures/keyword_heatmap.png")

def main():
    trend_df = pd.read_csv(TREND_PATH, index_col=0)

    # Keep top 8 keywords overall
    top_keywords = trend_df.sum(axis=1).sort_values(ascending=False).head(8).index
    small_df = trend_df.loc[top_keywords]

    # Bar plot
    ax = small_df.T.plot(kind="bar", figsize=(10,5))
    plt.tight_layout()
    BAR_OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(BAR_OUT)
    plt.close()

    # Heatmap
    fig, ax = plt.subplots(figsize=(8,5))
    im = ax.imshow(small_df.values, aspect="auto")
    ax.set_xticks(range(len(small_df.columns)))
    ax.set_xticklabels(small_df.columns)
    ax.set_yticks(range(len(small_df.index)))
    ax.set_yticklabels(small_df.index)
    fig.colorbar(im)
    plt.tight_layout()
    plt.savefig(HEAT_OUT)
    plt.close()

    print("Visualizations generated.")

if __name__ == "__main__":
    main()
